# Original benchmark text

| Dataset | Documents | UTF-8 bytes |
|---|---:|---:|
| Novel | 20 | 4,834,606 |
| Medical | 44 | 1,055,012 |
| Agriculture | 12 | 8,882,765 |
| Total | 76 | 14,772,383 |

These are the original text inputs used to construct this benchmark's GraphRAG
indices. Each file's decoded text was checked against the corresponding original
`documents.parquet` record before publication. Original filenames and raw bytes
are retained. `FILES_SHA256.json` records SHA-256 and byte length for every file.
The question-answer sets are not used by the extraction benchmark and are not
included. No generated indices, embeddings or experiment results are included.

Agriculture was obtained from the DIGIMON dataset distributed by
[JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG), archive `datasets.tar.gz`
(SHA-256 `1f49235d5a34022de292abc437c43280d557573c77ef244d859fd536a059b4a6`),
member `datasets/agriculture/Corpus.json`. The corpus contains 12 documents;
`Reclaiming Our Food` is record 3. Novel and Medical retain the existing
benchmark corpus text used by the original indexed documents.

Rebuild local graphs and vectors with `scripts/setup/build_index.py`, following
the root README. The same rebuilt indices must be shared across all methods.
