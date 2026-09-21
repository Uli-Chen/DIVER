# Shared baseline runtime

`run.py` handles transport, accounting, durable round commits, evaluation and
rmux supervision. `algorithms.py` imports each named method; attack policies
are implemented in `../TGTB`, `../PIDE` and `../IKEA`. `common.py` supplies
shared static-query state and the loader for pinned upstream definitions.
`legacy_prompts/` supports retained historical-controller replay tests.

Use a named method entrypoint as documented in [../README.md](../README.md).
