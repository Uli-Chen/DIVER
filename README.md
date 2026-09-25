# DIVER

Reproducibility code for topology-sensitive GraphRAG reconstruction. The
implementation uses the historical name **BNRR**; DIVER is the paper's method.

## Contents

```text
src/         DIVER controller, runtime and graph-recovery evaluator
baselines/   AGEA, GRASP, TGTB, PIDE and IKEA implementations
configs/     Attack prompts, indexing configuration and target prompts
data/        Original text corpora: Novel, Medical and Agriculture
scripts/     Experiment launcher, preflight and release checks
  setup/     Index builder and optional encoder/embedding tools
tests/       Offline regression tests
```

Only original text data is included. Generated graphs, embeddings, experiment
results, logs, model caches, manuscript drafts and local archives are excluded.
See [data/README.md](data/README.md) for document counts and checksums.

## Installation

Use Python 3.10 and `uv`:

```sh
uv sync --frozen --extra baselines --extra test
source .venv/bin/activate
cp .env.example .env
```

Fill in your chat and embedding provider credentials in `.env`. The baseline
extra supplies IKEA and GRASP's local sentence encoder. Git LFS is not needed.

## Rebuild the target indices

For each of `novel`, `medical`, and `agriculture`, prepare and inspect a new
local index root. These two commands make no model requests:

```sh
python scripts/setup/build_index.py prepare --dataset novel
python scripts/setup/build_index.py check --dataset novel
```

Then launch indexing in a dedicated `rmux` session (this calls the configured
chat and embedding APIs):

```sh
python scripts/setup/build_index.py launch --dataset novel --session index-novel-001
rmux attach -t index-novel-001
```

Repeat with dataset-specific session names. Install `rmux` and put it on PATH.
Outputs default to `result/inputs/{dataset}/`, which is ignored by Git. Set
`DIVER_DATA_ROOT` to another parent directory if needed. Use a fresh directory
for each rebuild; existing indices are never overwritten. An index can also be
prepared with `--root /path/to/new/index` and launched with the same `--root`.

The builder preserves the input/configuration hashes, full build log, request
ledger and final `BUILD_STATUS.json`. Its session closes after the worker exits
and diagnostics are saved. Read `BUILD_STATUS.json` before starting attacks.
Indexing uses GraphRAG 2.7.2, the bundled prompts, 1200-token chunks with 100-token
overlap, and the configured 4096-dimensional embedding model. Chat requests
explicitly disable thinking for the supplied DashScope/DeepSeek providers.
The indexing concurrency is capped at 8. Rebuilt graphs and fresh API responses
can differ from historical paper runs; exact saved-result replay is not included.

## Run DIVER

After rebuilding all three indices:

```sh
python scripts/preflight.py
python scripts/run_bnrr.py launch \
  --dataset novel --seeds 42 43 44 --rounds 100 \
  --root tmp/novel_reproduction_001 --session DIVER-novel-reproduction-001
rmux attach -t DIVER-novel-reproduction-001
```

Use `medical` or `agriculture` for the other datasets, and `--rounds 1000` for
a larger query budget. Each launch freezes code/configuration and stores
responses, graph deltas, metrics and accounting under its new `tmp/` directory.
The session closes after its workers finish. Resume through the frozen
workspace entrypoint rather than changed source code.

Defaults are seeds 42/43/44, 100 scheduled rounds including initialization,
rank admission 90% to 10%, exposure penalty 0.5, and exploration mixture 0.2.
Provider failures and skipped rounds remain in the budget. Novel/Agriculture
use the configured DashScope target; Medical uses the separate DeepSeek
credentials. All methods must evaluate against the same rebuilt graph.
The historical Agriculture table selected original seed 43 and rerun1 seeds
42/44; it is descriptive selected performance, not an unselected three-seed
validation. New reproductions should retain every attempt.

[Baseline instructions](baselines/README.md) describe each method's entrypoint.
IKEA and GRASP need the pinned local encoder:

```sh
python scripts/setup/download_baseline_encoder.py
```

GRASP's online execution wrapper currently requires macOS `sandbox-exec`.

## Offline checks

```sh
pytest tests baselines/_shared/test_baselines.py baselines/GRASP/tests -q
python scripts/check_release.py --verify-manifest
```

These checks do not launch model experiments. Tests with unpublished historical
fixtures are skipped. The release check verifies raw-data hashes, upload scope
and credential patterns. Local `result/`, `tmp/`, `paper/` and `.local_archive/`
are excluded from publication.
