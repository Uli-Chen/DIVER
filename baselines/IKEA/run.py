"""Run this baseline through the shared GraphRAG harness."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'baselines/_shared'), str(ROOT)]
from run import main

if __name__ == '__main__':
    main(default_method='IKEA')
