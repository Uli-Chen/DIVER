# Baseline implementations

| Method | Implementation | Entry point | Provenance |
|---|---|---|---|
| AGEA | `AGEA/` | `AGEA/graphrag/run_agea.py` | `AGEA/UPSTREAM.md` |
| GRASP | `GRASP/grasp/` | `GRASP/grasp.py` | `GRASP/README.md` |
| TGTB | `TGTB/attack.py`, `TGTB/upstream/` | `TGTB/run.py` | `TGTB/UPSTREAM.json` |
| PIDE | `PIDE/attack.py`, `PIDE/prompt.py`, `PIDE/upstream/` | `PIDE/run.py` | `PIDE/UPSTREAM.json` |
| IKEA | `IKEA/attack.py`, `IKEA/upstream/` | `IKEA/run.py` | `IKEA/UPSTREAM.json` |

`_shared/` holds common GraphRAG transport, response-to-graph measurement,
request accounting and scheduling. Each method's attack policy lives in its
own named directory. No baseline experiment outputs are published.

TGTB uses the official prompt constructor. PIDE's original repository omits
its prompt constructor, so our benchmark uses the exact PIDE formatter from
the pinned official IKEA implementation (`PIDE/prompt.py`). IKEA loads the
original mutation/feedback definitions without the upstream GPU/private-data
entrypoint. These are adaptations to our common GraphRAG benchmark, not claims
to reproduce the papers' original datasets or scores.

Install the `baselines` extra as described in the root README. IKEA and GRASP
also require a local `sentence-transformers/all-mpnet-base-v2` encoder; place
it in `_shared/cache/mpnet` (excluded from Git), or pass `--model-path` for
IKEA / edit `frame_encoder.model_path` for GRASP. Download the same pinned encoder revision with:

```sh
uv run --frozen --extra baselines python scripts/setup/download_baseline_encoder.py
```

## TGTB, PIDE and IKEA

After configuring `.env`, replace `TGTB` below with the desired method:

```sh
python baselines/TGTB/run.py prepare --dataset novel --seeds 42 43 44 --rounds 100 \
  --root baselines/TGTB/runs/novel_reproduction_001
python baselines/TGTB/run.py launch \
  --root baselines/TGTB/runs/novel_reproduction_001 --run-id novel-reproduction-001
python baselines/TGTB/run.py status --root baselines/TGTB/runs/novel_reproduction_001
python baselines/TGTB/run.py summarize --root baselines/TGTB/runs/novel_reproduction_001
```

For IKEA add `--model-path baselines/_shared/cache/mpnet` to `prepare`.
Each launcher prepares and launches only its named method. Each experiment
uses its own rmux session and keeps durable logs. Use a new run directory;
resume through its frozen workspace. Failed/truncated target slots remain in
the budget. TGTB/PIDE retrieval uses the anchor; IKEA uses its generated question.
A response-only converter measures the output and does not guide the attack.

The historical comparison selected one whole seed by final directed-edge F1,
then node F1, then smaller seed ID; new runs retain all candidate seeds and
costs. Do not mix per-metric best seeds or claim average-seed performance.
