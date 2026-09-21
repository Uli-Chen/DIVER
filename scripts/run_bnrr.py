#!/usr/bin/env python3
"""Official BNRR entry point; each dispatched experiment owns an rmux session."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extraction.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
