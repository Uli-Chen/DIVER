"""Check that the published layout can freeze one baseline without local state."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/_shared'))
import run as harness


def test_single_method_freeze_without_dotenv_or_encoder(tmp_path, monkeypatch):
    project = tmp_path / 'checkout'
    (project / 'src').mkdir(parents=True)
    (project / 'src/example.py').write_text('VALUE = 1\n')
    (project / 'baselines/AGEA').mkdir(parents=True)
    (project / 'baselines/AGEA/example.py').write_text('VALUE = 2\n')
    (project / 'baselines/__init__.py').write_text('')
    for method in harness.METHODS:
        shutil.copytree(ROOT / 'baselines' / method, project / 'baselines' / method,
                        ignore=shutil.ignore_patterns('Information', 'imgs', 'runs', '__pycache__'))
    graph = tmp_path / 'graph'
    (graph / 'output').mkdir(parents=True)
    (graph / 'prompts').mkdir()
    for name in ['entities.parquet', 'relationships.parquet']:
        (graph / 'output' / name).write_bytes(b'fixture graph bytes')
    (graph / 'prompts/local.txt').write_text('{context_data}')
    (graph / 'settings.yaml').write_text(yaml.safe_dump({
        'models': {'default_chat_model': {}},
        'vector_store': {'default_vector_store': {}}, 'cache': {}, 'reporting': {},
    }))
    monkeypatch.setattr(harness, 'PROJECT', project)
    monkeypatch.setattr(harness, 'credentials', lambda dataset: None)
    from extraction import paths
    monkeypatch.setattr(paths, 'graph_root', lambda dataset: graph)
    for key in ['GRAPHRAG_CHAT_MODEL', 'GRAPHRAG_API_BASE', 'GRAPHRAG_EMBEDDING_MODEL']:
        monkeypatch.setenv(key, 'offline-fixture')
    monkeypatch.setenv('AGEA_THINKING_CONTROL_STYLE', 'enable_thinking')
    out = project / 'baselines/TGTB/runs/test'
    harness.prepare(out, 2, [42], None, 'novel', ('TGTB',))
    manifest = harness.check(out)
    assert manifest['methods'] == ['TGTB']
    assert not (project / '.env').exists()
    assert (out / 'workspace/baselines/TGTB/attack.py').exists()
    assert (out / 'workspace/baselines/_shared/common.py').exists()
    assert list(harness.summarize(out)['methods']) == ['TGTB']
    env = {**os.environ, 'BASELINE_PROJECT': str(project), 'BASELINE_SOURCE': str(out / 'workspace')}
    help_result = subprocess.run([sys.executable, str(out / 'workspace/baselines/_shared/run.py'), '--help'],
                                 env=env, cwd=tmp_path, capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    assert '--root' in help_result.stdout
