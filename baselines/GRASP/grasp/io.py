from pathlib import Path
import hashlib, json, os

ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=ROOT.parents[1]


def inside(path):
    p=Path(path).absolute()
    if not p.resolve().is_relative_to(ROOT):raise PermissionError('Output outside implementation root')
    return p


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    p=inside(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp')
    with t.open('w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def tree_hash(path):
    root=Path(path)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def implementation_hashes():
    paths=[ROOT/'grasp.py',*list((ROOT/'grasp').glob('*.py')),*list((ROOT/'prompts').glob('*.txt'))]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
