"""Rebuilding starts from verified raw text and refuses changed prepared inputs."""
import importlib.util
from pathlib import Path

import pytest
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_index', ROOT / 'scripts/setup/build_index.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_raw_corpus_prepare_and_tamper_detection(tmp_path):
    root = tmp_path / 'medical'
    builder.prepare('medical', root)
    record = builder.check_prepared(root)
    assert record['dataset'] == 'medical'
    assert len(list((root / 'input').glob('*.txt'))) == 44
    assert not (root / 'output').exists()
    assert not (root / '.env').exists()
    with pytest.raises(FileExistsError):
        builder.prepare('medical', root)
    (root / 'settings.yaml').write_text('changed')
    with pytest.raises(ValueError, match='Prepared input/configuration changed'):
        builder.check_prepared(root)


def test_check_cannot_make_model_preflight_calls(tmp_path, monkeypatch):
    root = tmp_path / 'medical'
    builder.prepare('medical', root)
    original_connect = socket.socket.connect
    def inspect_invocation(*args, **kwargs):
        assert '--dry-run' in sys.argv and '--skip-validation' in sys.argv
        with pytest.raises(RuntimeError, match='Network disabled'):
            socket.socket.connect(None, ('example.invalid', 443))
    monkeypatch.setattr(builder.runpy, 'run_module', inspect_invocation)
    builder.index(root, dry_run=True)
    assert socket.socket.connect is original_connect
