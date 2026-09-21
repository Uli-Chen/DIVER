"""Constrain generated files to baselines before importing third-party libraries."""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('BASELINE_PROJECT', BASE.parents[1])).resolve()
# Frozen execution sets a separate source root, but credentials remain read-only.
SOURCE = Path(os.environ.get('BASELINE_SOURCE', PROJECT))

def setup():
    sys.dont_write_bytecode = True
    for key, suffix in {
        'XDG_CACHE_HOME': 'cache/xdg', 'HF_HOME': 'cache/huggingface',
        'SENTENCE_TRANSFORMERS_HOME': 'cache/sentence_transformers',
        'TORCH_HOME': 'cache/torch', 'TIKTOKEN_CACHE_DIR': 'cache/tiktoken',
        'TMPDIR': 'tmp', 'MPLCONFIGDIR': 'cache/matplotlib',
        'NUMBA_CACHE_DIR': 'cache/numba', 'UV_CACHE_DIR': 'cache/uv',
    }.items():
        path = BASE / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                      TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_TELEMETRY='1',
                      ANONYMIZED_TELEMETRY='false', OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    sys.path.insert(0, str(SOURCE / 'src'))
    sys.path.insert(0, str(SOURCE))

setup()
