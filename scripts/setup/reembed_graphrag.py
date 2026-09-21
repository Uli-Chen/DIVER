#!/usr/bin/env python3
"""Build an isolated embedding-only GraphRAG copy; never modify the source graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

PROJECT = Path(__file__).resolve().parents[2]
MODEL = "qwen3.7-text-embedding-flash"
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(source: Path, destination: Path, batch_size: int = 4, local_search_only: bool = False):
    import lancedb
    import pyarrow as pa
    import yaml
    from dotenv import load_dotenv
    from openai import OpenAI

    source, destination = source.resolve(), destination.resolve()
    if not 1 <= batch_size <= 20:
        raise ValueError("Batch size must be between 1 and 20")
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be separate from the original GraphRAG workspace")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing index")
    load_dotenv(PROJECT / ".env", override=True)
    if os.environ.get("GRAPHRAG_EMBEDDING_MODEL") != MODEL or os.environ.get("GRAPHRAG_EMBEDDING_API_BASE") != BASE:
        raise ValueError("Embedding configuration does not match the requested migration")
    source_db = lancedb.connect(str(source / "output/lancedb"))
    tables = sorted(source_db.table_names())
    if local_search_only:
        if "default-entity-description" not in tables:
            raise ValueError("Required local-search entity vector table is absent")
        tables = ["default-entity-description"]
    if not tables:
        raise ValueError("Source has no vector tables")
    destination.mkdir(parents=True)
    output = destination / "output"
    output.mkdir()
    parquet_hashes = {}
    for path in sorted((source / "output").glob("*.parquet")):
        parquet_hashes[path.name] = digest(path)
        (output / path.name).symlink_to(path)
    shutil.copytree(source / "prompts", destination / "prompts")
    settings = yaml.safe_load((source / "settings.yaml").read_text())
    settings["models"]["default_chat_model"].update({"model": "${GRAPHRAG_CHAT_MODEL}",
        "api_base": "${GRAPHRAG_API_BASE}", "api_key": "${GRAPHRAG_API_KEY}", "max_retries": 2, "request_timeout": 120})
    settings["models"]["default_embedding_model"].update({"model": MODEL,
        "api_base": "${GRAPHRAG_EMBEDDING_API_BASE}", "api_key": "${GRAPHRAG_EMBEDDING_API_KEY}", "max_retries": 2, "request_timeout": 120})
    settings["vector_store"]["default_vector_store"]["db_uri"] = str(output / "lancedb")
    (destination / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    manifest = {"status": "building", "model": MODEL, "api_base": BASE, "dimensions": 1024,
        "source": str(source), "parquet_sha256": parquet_hashes, "tables": {}, "requests": 0,
        "tokens": 0, "created": time.time(), "only_embeddings_changed": True, "batch_size": batch_size,
        "scope": "local_search_entity_descriptions" if local_search_only else "all_vector_tables"}
    manifest_path = destination / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    target_db = lancedb.connect(str(output / "lancedb"))
    schema = pa.schema([pa.field("id", pa.string()), pa.field("text", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), 1024)), pa.field("attributes", pa.string())])
    with OpenAI(api_key=os.environ["GRAPHRAG_EMBEDDING_API_KEY"], base_url=BASE, timeout=120, max_retries=0) as client:
        for name in tables:
            records = source_db.open_table(name).to_arrow().select(["id", "text", "attributes"]).to_pylist()
            if len({row["id"] for row in records}) != len(records):
                raise ValueError(f"Duplicate vector document IDs in {name}")
            table = target_db.create_table(name, schema=schema)
            batch_seconds = []
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                if any(not isinstance(r["text"], str) or not r["text"].strip() for r in batch):
                    raise ValueError("Cannot silently replace an empty embedding document")
                started = time.monotonic()
                response = client.embeddings.create(model=MODEL, input=[r["text"] for r in batch], encoding_format="float")
                vectors = sorted(response.data, key=lambda item: item.index)
                if [v.index for v in vectors] != list(range(len(batch))):
                    raise ValueError("Embedding response indices do not match batch")
                for row, item in zip(batch, vectors):
                    if len(item.embedding) != 1024 or not all(math.isfinite(x) for x in item.embedding):
                        raise ValueError("Invalid embedding vector")
                    row["vector"] = item.embedding
                table.add(pa.Table.from_pylist(batch, schema=schema))
                batch_seconds.append(time.monotonic() - started)
                manifest["requests"] += 1
                manifest["tokens"] += response.usage.total_tokens
                manifest["progress"] = {"table": name, "embedded_rows": offset + len(batch)}
                manifest_path.write_text(json.dumps(manifest, indent=2))
            # Compare unchanged document identity/text/attributes, not vectors.
            actual = table.to_arrow().select(["id", "text", "attributes"]).to_pylist()
            expected = [{k: row[k] for k in ("id", "text", "attributes")} for row in records]
            if actual != expected:
                raise ValueError("Re-embedding changed document metadata")
            manifest["tables"][name] = {"rows": len(records), "dimensions": 1024,
                "document_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                "batch_seconds": batch_seconds}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(json.dumps({"table": name, "rows": len(records), "completed": True}), flush=True)
    for name, sha in parquet_hashes.items():
        if digest(output / name) != sha:
            raise ValueError("Shared graph changed during re-embedding")
    manifest.update(status="completed", finished=time.time())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "completed", "requests": manifest["requests"], "tokens": manifest["tokens"], "destination": str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-search-only", action="store_true")
    args = parser.parse_args()
    rebuild(args.source, args.destination, args.batch_size, args.local_search_only)
