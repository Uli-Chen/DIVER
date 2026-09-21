#!/usr/bin/env python3
"""Verify a source-and-raw-corpus release; generated results are always excluded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/FILES_SHA256.json'
SECRET_PATTERN = re.compile(rb'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def candidates():
    output = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    return sorted({os.fsdecode(p) for p in output.split(b'\0') if p and os.path.lexists(ROOT / os.fsdecode(p))})


def known_secrets():
    path = ROOT / '.env'
    if not path.exists():
        return []
    found = []
    for line in path.read_text().splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        name, value = line.split('=', 1)
        value = value.strip().strip('\"\'')
        if any(word in name.lower() for word in ('key', 'token', 'secret', 'password')) and len(value) >= 16 and '${' not in value:
            found.append(value.encode())
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-manifest', action='store_true', help='Raw corpus hashes are always verified')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths, secrets = candidates(), known_secrets()
    expected = json.loads(MANIFEST.read_text())['files']
    errors, raw_files, total = [], {}, 0
    attributes = subprocess.check_output(['git', 'check-attr', '-z', '--stdin', 'filter'],
        input=b''.join(os.fsencode(p) + b'\0' for p in paths), cwd=ROOT).split(b'\0')
    lfs = [os.fsdecode(attributes[i]) for i in range(0, len(attributes)-1, 3) if attributes[i+2] == b'lfs']
    if lfs:
        errors.append({'reason': 'This release must not depend on LFS', 'paths': lfs})
    for rel in paths:
        path = ROOT / rel
        if rel.startswith(('.local_archive/', 'paper/', 'tmp/', 'result/')) or path.name in {'.env', 'provider.env'}:
            errors.append({'path': rel, 'reason': 'local/generated/private artifact in upload scope'})
        if rel.startswith('baselines/') and any(part in {'runs', 'runtime', 'cache'} for part in Path(rel).parts):
            errors.append({'path': rel, 'reason': 'baseline output/runtime in upload scope'})
        if path.is_symlink():
            if os.path.isabs(os.readlink(path)) or not path.exists() or not path.resolve().is_relative_to(ROOT):
                errors.append({'path': rel, 'reason': 'nonportable or broken link'})
            continue
        content = path.read_bytes()
        total += len(content)
        if len(content) > 95 * 1024 * 1024:
            errors.append({'path': rel, 'reason': 'file over 95 MiB'})
        if SECRET_PATTERN.search(content) or any(value in content for value in secrets):
            errors.append({'path': rel, 'reason': 'possible credential; value omitted'})
        if rel.startswith('data/') and path.suffix == '.txt':
            raw_files[rel.removeprefix('data/')] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    if raw_files != expected:
        changed = sorted(k for k in raw_files.keys() | expected.keys() if raw_files.get(k) != expected.get(k))
        errors.append({'reason': 'raw corpus differs from manifest', 'paths': changed})
    result = {'status': 'failed' if errors else 'passed', 'candidate_files': len(paths),
              'total_bytes': total, 'raw_corpus_files': len(raw_files),
              'raw_corpus_bytes': sum(e['bytes'] for e in raw_files.values()),
              'lfs_files': len(lfs), 'errors': errors}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
