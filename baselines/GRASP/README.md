# GRASP

Paper-based reimplementation of Song et al., arXiv:2602.06495v2. This is not
author-released code and does not claim to reproduce the published scores.
The attack implementation is in `grasp/`; templates are in `prompts/`.

`configs/benchmark/{novel,medical,agriculture}.json` adapts the method to our
whole-graph comparison: seed 42, 100 rounds including discovery, FIFO frontier,
16,384 output tokens and the common directed-endpoint evaluator. The existing
indices have no explicit ground-truth relation-type column. The targeted
paper-profile template instead requires a compatible typed index and an
explicit target cohort; its main-paper output budget is 2,048 tokens.

The implementation includes semantic context-frame selection, all 13 scheduler
policy rows, Good–Turing singleton counts, momentum, template reweighting and
soft reset. Complete records from multiple response blocks are retained;
truncated output contributes only complete records. IDs are not checked against
ground truth. The attacker receives response text only; truth is read offline.

Implementation choices not specified by the paper: all-mpnet-base-v2 encoder,
five-turn novelty window, residual cap five, deterministic frame-order tie
breaking, and the whole-graph frontier/discovery wrapper. The stopping warm-up
uses five completed observations. Local endpoint settings do not establish
byte-for-byte equivalence with the paper's defended victim.

## Reproduce

Install the `baselines` extra and run `scripts/setup/download_baseline_encoder.py`
from the repository root. The current execution wrapper requires macOS
`sandbox-exec` and `rmux`; offline unit tests do not require the sandbox.
Graph and source paths in the supplied configs are repository-relative.

```sh
python -B -m unittest discover -s baselines/GRASP/tests
python -B baselines/GRASP/grasp.py inspect --config baselines/GRASP/configs/benchmark/novel.json
python -B baselines/GRASP/grasp.py prepare \
  --config baselines/GRASP/configs/benchmark/novel.json \
  --run baselines/GRASP/runs/novel_reproduction_001
```

Inspection and preparation make no model requests. Preparation freezes source,
model and index hashes. After inspecting the prepared configuration, record
explicit approval for that exact manifest, using the run's absolute path:

```python
import hashlib, json
from pathlib import Path
run = Path('baselines/GRASP/runs/novel_reproduction_001').resolve()
(run / 'USER_APPROVAL.json').write_text(json.dumps({
    'approved': True,
    'user_instruction': 'Run this prepared GRASP reproduction.',
    'run': str(run),
    'manifest_sha256': hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest(),
}))
```

```sh
python -B baselines/GRASP/grasp.py launch \
  --run baselines/GRASP/runs/novel_reproduction_001 \
  --approval baselines/GRASP/runs/novel_reproduction_001/USER_APPROVAL.json \
  --execute-online
```

The launcher uses a private rmux socket, prints the attach command, and saves it
in `LAUNCH.json`. Replies, request ledgers, costs and terminal status persist
independently of the session. It closes only its own session on completion.
`validate`, `replay` and `evaluate` work on a prepared run without model calls;
use `--resume` only with the matching frozen code and configuration.
Generated `runs/` and `runtime/` files are excluded from this release.
