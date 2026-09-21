# AGEA

The native implementation is in `graphrag/run_agea.py`, with graph/query memory
in this directory. See `UPSTREAM.md` for the upstream revision and compatibility
patches. REACH also reuses its GraphRAG transport adapter.

The native runner reads datasets from `AGEA_GRAPHRAG_ROOT`, defaulting to
`result/inputs`. First rebuild the index as described in the root README.
For a new run, copy it to a dedicated local target to isolate logs and cache:

```sh
mkdir -p baselines/AGEA/runs/novel_reproduction_001/graphs
cp -R result/inputs/novel baselines/AGEA/runs/novel_reproduction_001/graphs/
export AGEA_GRAPHRAG_ROOT="$PWD/baselines/AGEA/runs/novel_reproduction_001/graphs"
rmux -V
rmux new-session -s agea-novel-reproduction-001
```

Inside that session, run from the repository root with the configured environment:

```sh
uv run --frozen python baselines/AGEA/graphrag/run_agea.py \
  --dataset novel --turns 100 --random-seed 42 \
  --disable-graph-filter --disable-api-thinking \
  --query-generator-model deepseek-v4-flash
```

Keep the session while the worker runs. After its outputs and exit status are
saved, close only `agea-novel-reproduction-001`. Generated results under `runs/`
are excluded from the release. For Medical, map the separate Medical provider
credentials to the `AGEA_*` and `GRAPHRAG_*` chat variables before starting the
native runner; the shared embedding configuration remains unchanged.
