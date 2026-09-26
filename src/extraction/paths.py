"""Portable paths for new runs; historical manifests are never rewritten."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def data_root():
    return Path(os.environ.get('DIVER_DATA_ROOT', PROJECT / 'result/inputs')).expanduser().resolve()


def graph_root(dataset):
    if dataset not in {'novel', 'medical', 'agriculture'}:
        raise ValueError(f'Unknown dataset: {dataset}')
    return data_root() / dataset
