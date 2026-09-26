# DIVER

Anonymous reproduction package for **DIVER: Knowledge Theft from GraphRAG System via Structural Diversity-Aware Querying**.

DIVER reconstructs a GraphRAG knowledge graph from responses. It combines the Balanced Non-Redundant Ratio (BNRR), neighborhood query exposure, and a budget-dependent rank-admission schedule to select queries. The `bnrr` names in the code refer to this method.

## Installation

Use Python 3.10 and `uv`, from this directory:

```sh
uv sync --frozen
source .venv/bin/activate
cp .env.example .env
```

Set the two API keys in `.env`. The six `PROVIDER_*` fields configure the chat and embedding endpoints, credentials, and models. All datasets and the query writer share the same chat provider. Use an OpenAI-compatible endpoint serving the supplied models; the embedding endpoint must return 4,096-dimensional vectors. Thinking is disabled automatically for the supplied DeepSeek model.

## Build the target graphs

The package includes the 76 source documents for Novel (20), Medical (44), and Agriculture (12), their checksums, and the indexing prompts. See [data/README.md](data/README.md) for data provenance. Historical graph indices, embeddings, and experimental results are not included.

Build each index once, then keep it fixed across seeds and query budgets:

```sh
for dataset in novel medical agriculture; do
  python scripts/setup/build_index.py prepare --dataset "$dataset"
  python scripts/setup/build_index.py check --dataset "$dataset"
  python scripts/setup/build_index.py run --dataset "$dataset"
done
```

`prepare` and `check` make no model requests; `run` calls the chat and embedding APIs. The default index location is `result/inputs/<dataset>/`. A successful build writes `BUILD_STATUS.json` with `"status": "complete"`. Set `DIVER_DATA_ROOT` before both indexing and extraction to use another index parent directory. Use a new index directory for an independent rebuild.

The indexing configuration uses GraphRAG 2.7.2, DeepSeek-V4-Flash, Qwen3-Embedding-8B with 4,096-dimensional vectors, and 1,200-token chunks with 100-token overlap. Rebuilding invokes external models and can change graph contents and evaluation denominators. This package supports fresh experiments; it cannot guarantee exact reproduction of the paper's saved numerical results.

## Run the experiments

The main experiment uses seeds 42, 43, and 44 on all three datasets, with **100 scheduled rounds including initialization**:

```sh
for dataset in novel medical agriculture; do
  python scripts/run_bnrr.py prepare \
    --dataset "$dataset" --seeds 42 43 44 --rounds 100 \
    --root "tmp/main/${dataset}_100r"
  python scripts/run_bnrr.py run --root "tmp/main/${dataset}_100r"
done
```

Use Linux for the complete GraphRAG workflow. The runner operates in the foreground and reports progress. Seeds run concurrently within each dataset. `prepare` requires a new run directory and records the configuration and input hashes. Keep the source and target indices unchanged while a prepared experiment runs.

The paper's 1,000-round experiment uses Novel and seed 42:

```sh
python scripts/run_bnrr.py prepare \
  --dataset novel --seeds 42 --rounds 1000 --root tmp/extended/novel_1000r
python scripts/run_bnrr.py run --root tmp/extended/novel_1000r
```

Start a fresh trajectory for each budget: the admission schedule depends on the configured horizon, so a 100-round run is not a prefix of the 1,000-round protocol.

The supplied configuration follows Sections 3 and Appendices B/D:

| Setting | Value |
|---|---|
| Rank admission | Linear from 0.9 to 0.1 over post-initialization rounds; fractional boundary ties |
| Neighbor exposure increment, alpha | 0.5 |
| Full-pool anchor-mixture weight, rho | 0.2 |
| Query writer | DeepSeek-V4-Flash; exploration temperature 0.3, exploitation temperature 0.2; 1,024 output tokens |
| Victim | DeepSeek-V4-Flash; temperature 0; 16,384 output tokens |
| Retrieved context | 12,000 tokens |
| Response processing | Deterministic record parser; thinking disabled; no additional LLM filter |

Failed or skipped scheduled rounds consume the budget. Accepted exploitation queries update exposure using the pre-query neighborhood even when they reveal no new records. The seed fixes controller randomness; external model responses can still vary.

An interrupted extraction worker can resume its last committed round:

```sh
python scripts/run_bnrr.py worker --root tmp/main/novel_100r --seed 42 --resume
python scripts/run_bnrr.py summarize --root tmp/main/novel_100r
```

Use the original source, indices, and configuration. Summarize only after all selected seeds finish. Initialization failures require a new experiment directory.

## Results and evaluation

Each experiment directory contains:

- `RESULTS.json`: final per-seed results and arithmetic means with sample standard deviations.
- `RESULTS.md`: the aggregate result table.
- `runs/FULL_seed<seed>/metrics.jsonl`: round-by-round node and directed-edge precision, recall, and F1.
- `runs/FULL_seed<seed>/final_graph.graphml`: the recovered graph.
- `EFFICIENCY.json`: recorded token usage, including initialization.
