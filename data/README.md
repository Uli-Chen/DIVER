# Benchmark corpora

| Dataset | Documents | UTF-8 bytes |
|---|---:|---:|
| Novel | 20 | 4,834,606 |
| Medical | 44 | 1,055,012 |
| Agriculture | 12 | 8,882,765 |
| Total | 76 | 14,772,383 |

These are the source texts used to construct the GraphRAG targets. Original
filenames and bytes are retained; `FILES_SHA256.json` records each file's SHA-256
and byte length. The index preparation command verifies these checksums.

Agriculture comes from the DIGIMON dataset distributed by
[JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG), archive `datasets.tar.gz`
(SHA-256 `1f49235d5a34022de292abc437c43280d557573c77ef244d859fd536a059b4a6`),
member `datasets/agriculture/Corpus.json`. Novel and Medical retain the benchmark
texts corresponding to the original indexed documents. Source-text author names
and third-party attribution are part of the datasets.

Follow the root README to rebuild the indices. Generated graphs, embeddings,
and historical experimental results are not included.
