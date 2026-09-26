#!/usr/bin/env python3
"""Prepare or rebuild a local GraphRAG index from the published text corpus."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
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
    # Cost accounting uses provider usage; a remote price map is unnecessary.
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    if dry_run:
        # Validate GraphRAG's configuration without importing or running the
        # indexing pipeline, which is unnecessary for this offline check.
        connect = socket.socket.connect
        def offline_only(*args, **kwargs):
            raise RuntimeError('Network disabled during index configuration check')
        try:
            socket.socket.connect = offline_only
            from graphrag.config.load_config import load_config
            config = load_config(root)
            if config.local_search.max_context_tokens != 12000:
                raise ValueError('DIVER requires a 12000-token retrieval context')
        finally:
            socket.socket.connect = connect
        print(json.dumps({'status': 'passed', 'api_requests': 0}))
        return
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
    previous_cwd, previous_argv = Path.cwd(), sys.argv
    try:
        os.chdir(tempfile.gettempdir())
        sys.argv = ['graphrag', 'index', '--root', str(root)]
        runpy.run_module('graphrag', run_name='__main__')
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def run(root):
    """Build and validate the index in the foreground, with durable logs."""
    check_prepared(root)
    result = {'status': 'failed', 'started': time.time()}
    write(root / 'BUILD_STATUS.json', {**result, 'status': 'running'})
    try:
        print('Building index; logs: ' + str(root / 'build.log'), flush=True)
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
    if result['status'] != 'complete':
        raise RuntimeError('Index build failed; inspect ' + str(root / 'build.log'))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'run', '_index'])
    parser.add_argument('--dataset', choices=['novel', 'medical', 'agriculture'])
    parser.add_argument('--root', type=Path)
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
    else:
        run(root)


if __name__ == '__main__':
    main()
