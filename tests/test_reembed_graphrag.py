"""Embedding migration must preserve the graph and refuse overwrites."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reembed", ROOT / "scripts/setup/reembed_graphrag.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_local_embedding_migration_preserves_source(monkeypatch, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "output").mkdir(parents=True)
    (source / "prompts").mkdir()
    (source / "prompts/local.txt").write_text("unchanged victim prompt")
    (source / "settings.yaml").write_text(yaml.safe_dump({"models": {"default_chat_model": {}, "default_embedding_model": {}},
        "vector_store": {"default_vector_store": {}}}))
    pq.write_table(pa.table({"title": ["A"]}), source / "output/entities.parquet")
    before = (source / "output/entities.parquet").read_bytes()
    db = lancedb.connect(str(source / "output/lancedb"))
    db.create_table("default-entity-description", data=[{"id": "id-a", "text": "Full original description", "attributes": "{}", "vector": [1.0, 2.0]}])
    from unittest.mock import Mock
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1] * 1024)], usage=SimpleNamespace(total_tokens=8))
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    # A historical migration utility must not depend on the live workspace provider.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", module.MODEL)
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_API_BASE", module.BASE)
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_API_KEY", "test-migration-key")
    module.rebuild(source, target, batch_size=1, local_search_only=True)
    manifest = json.loads((target / "embedding_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["scope"] == "local_search_entity_descriptions"
    assert (source / "output/entities.parquet").read_bytes() == before
    assert (target / "output/entities.parquet").read_bytes() == before
    assert (target / "prompts/local.txt").read_text() == "unchanged victim prompt"
    assert client.embeddings.create.call_args.kwargs["input"] == ["Full original description"]
    assert lancedb.connect(str(target / "output/lancedb")).open_table("default-entity-description").schema.field("vector").type.list_size == 1024
    with pytest.raises(FileExistsError):
        module.rebuild(source, target)
    with pytest.raises(ValueError, match="separate"):
        module.rebuild(source, source / "child")
