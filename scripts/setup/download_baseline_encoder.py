#!/usr/bin/env python3
"""Download the pinned local encoder used by IKEA and GRASP (no model API calls)."""
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    snapshot_download('sentence-transformers/all-mpnet-base-v2',
                      revision='e8c3b32edf5434bc2275fc9bab85f82640a19130',
                      local_dir=root / 'baselines/_shared/cache/mpnet',
                      allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*'])
