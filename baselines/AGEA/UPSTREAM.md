# AGEA provenance

- Upstream: https://github.com/shuashua0608/AGEA
- Imported revision: `c9c27dd15d55fb2a64fef89d2f9f7cda80038ffe`
- Upstream branch at import: `main`
- Vendored on: 2026-08-08

The upstream `.git` directory is not included. The reproduction graph inputs are stored under `result/inputs/{dataset}/`.
Pass the matching graph root and output paths to the runner.

## Local compatibility changes

This snapshot includes the pre-existing local changes used by the experiments:

- provider-neutral OpenAI-compatible and Azure client construction;
- Paratera-compatible, case-sensitive model resolution;
- explicit query-generation error propagation instead of silent fallback;
- GraphRAG response/parser robustness and leakage-metric fixes;
- provider configuration and usage documentation.

These changes predate the repository reorganization and are retained so saved
and future runs use the same behavior. Comparisons must describe this directory
as the vendored AGEA baseline with compatibility patches, not an untouched
upstream checkout.
