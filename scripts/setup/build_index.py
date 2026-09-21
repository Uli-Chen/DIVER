#!/usr/bin/env python3
"""Prepare or rebuild a local GraphRAG index from the published text corpus."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from extraction.paths import graph_root


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare(dataset, root):
    manifest = json.loads((PROJECT / 'data/FILES_SHA256.json').read_text())['files']
    sources = {name: record for name, record in manifest.items() if name.startswith(dataset + '/')}
    if not sources:
        raise ValueError('No input corpus for ' + dataset)
    for name, record in sources.items():
        content = (PROJECT / 'data' / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Corpus hash mismatch: ' + name)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PROJECT / 'data' / dataset, root / 'input')
    config = PROJECT / 'configs/indexing' / dataset
    shutil.copy2(config / 'settings.yaml', root / 'settings.yaml')
    shutil.copytree(config / 'prompts', root / 'prompts')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file()}
    write(root / 'PREPARED.json', {'dataset': dataset, 'graphrag': '2.7.2', 'hashes': hashes})
    print(json.dumps({'prepared': str(root), 'documents': len(sources), 'api_requests': 0}))


def check_prepared(root):
    record = json.loads((root / 'PREPARED.json').read_text())
    for name, digest in record['hashes'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared input/configuration changed: ' + name)
    return record


def index(root, dry_run=False):
    from dotenv import load_dotenv
    load_dotenv(PROJECT / '.env')
    check_prepared(root)
    if not dry_run:
        import httpx
        from extraction.request_audit import RequestAudit

        class IndexAudit(RequestAudit):
            def prepare_request(self, request):
                request = super().prepare_request(request)
                if not request.url.path.endswith('/chat/completions'):
                    return request
                body = json.loads(request.content)
                if request.url.host == 'api.deepseek.com':
                    body['thinking'] = {'type': 'disabled'}
                else:
                    body['enable_thinking'] = False
                headers = request.headers.copy()
                headers.pop('content-length', None)
                return httpx.Request(request.method, request.url, headers=headers,
                                     content=json.dumps(body).encode(), extensions=request.extensions)

        audit = IndexAudit(root / 'requests.jsonl')
        audit.journal_starts = True
        audit.install()
    # Some transitive NLTK imports reject a working directory containing .venv.
    previous_cwd, previous_argv, connect = Path.cwd(), sys.argv, socket.socket.connect
    try:
        os.chdir(tempfile.gettempdir())
        sys.argv = ['graphrag', 'index', '--root', str(root)]
        if dry_run:
            # GraphRAG's regular preflight makes live model calls even with dry-run.
            sys.argv.extend(['--dry-run', '--skip-validation'])
            def offline_only(*args, **kwargs):
                raise RuntimeError('Network disabled during offline index configuration check')
            socket.socket.connect = offline_only
        runpy.run_module('graphrag', run_name='__main__')
    finally:
        socket.socket.connect = connect
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def supervise(root, session):
    result = {'status': 'failed', 'started': time.time()}
    try:
        with (root / 'build.log').open('w') as log:
            process = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '_index', '--root', str(root)], stdout=log, stderr=subprocess.STDOUT)
        result['exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError('GraphRAG indexing failed; inspect build.log and requests.jsonl')
        import lancedb
        for name in ['entities.parquet', 'relationships.parquet', 'text_units.parquet']:
            if not (root / 'output' / name).is_file():
                raise RuntimeError('Missing index output: ' + name)
        table = lancedb.connect(str(root / 'output/lancedb')).open_table('default-entity-description')
        dimensions = table.schema.field('vector').type.list_size
        if dimensions != 4096:
            raise RuntimeError(f'This benchmark requires 4096 dimensions, got {dimensions}')
        result.update(status='complete', embedding_dimensions=dimensions)
    except BaseException:
        result['traceback'] = traceback.format_exc()
    finally:
        result['finished'] = time.time()
        write(root / 'BUILD_STATUS.json', result)
        subprocess.run(['rmux', 'kill-session', '-t', session], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'launch', '_supervise', '_index'])
    parser.add_argument('--dataset', choices=['novel', 'medical', 'agriculture'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--session')
    args = parser.parse_args()
    if args.root is None and args.dataset is None:
        parser.error('Supply --dataset or --root')
    root = args.root.expanduser().resolve() if args.root else graph_root(args.dataset)
    if args.action == 'prepare':
        if args.dataset is None:
            parser.error('prepare requires --dataset')
        prepare(args.dataset, root)
    elif args.action == 'check':
        index(root, dry_run=True)
    elif args.action == '_index':
        index(root)
    elif args.action == '_supervise':
        if not args.session:
            parser.error('_supervise requires --session')
        supervise(root, args.session)
    else:
        if not args.session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.session):
            parser.error('launch requires a unique --session using letters, numbers, - or _')
        record = check_prepared(root)
        if record['dataset'] not in args.session:
            parser.error('--session must include the dataset name and a run identifier')
        if (root / 'BUILD_LAUNCH.json').exists():
            raise RuntimeError('This index already has a launch record; use a new root for a new run')
        subprocess.run(['rmux', '-V'], check=True)
        command = shlex.join([sys.executable, '-u', str(Path(__file__).resolve()), '_supervise',
                              '--root', str(root), '--session', args.session])
        # The supervisor and GraphRAG child live in this dedicated session.
        subprocess.run(['rmux', 'new-session', '-d', '-s', args.session, '-n', 'index', command], check=True)
        write(root / 'BUILD_LAUNCH.json', {'session': args.session, 'root': str(root), 'command': command})
        print('rmux attach -t ' + args.session)


if __name__ == '__main__':
    main()
