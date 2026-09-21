import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
from extraction import experiment as pilot


@pytest.mark.parametrize("dataset", ["medical", "agriculture"])
@pytest.mark.parametrize("prompt_profile", ("agea_adapted_null",))
def test_domain_prepare_preserves_original_and_freezes_expected_runtime(tmp_path, monkeypatch, dataset, prompt_profile):
    import lancedb
    source = tmp_path / "indices" / dataset
    (source / "output/lancedb").mkdir(parents=True)
    (source / "prompts").mkdir()
    (source / "prompts/local_search_system_prompt.txt").write_text("{context_data}")
    settings = {"models": {"default_chat_model": {"type": "chat"}, "default_embedding_model": {"type": "embedding"}},
        "vector_store": {"default_vector_store": {"db_uri": "output/lancedb"}}}
    (source / "settings.yaml").write_text(yaml.safe_dump(settings))
    original = (source / "settings.yaml").read_bytes()
    monkeypatch.setattr(pilot, "SOURCE_GRAPH", tmp_path / "indices/novel_9")
    monkeypatch.setattr(pilot, "environment", lambda: None)
    fake = SimpleNamespace(schema=SimpleNamespace(field=lambda name: SimpleNamespace(type=SimpleNamespace(list_size=4096))))
    monkeypatch.setattr(lancedb, "connect", lambda path: SimpleNamespace(open_table=lambda name: fake))
    for name, value in {"AGEA_API_BASE": "https://example.test/v1", "GRAPHRAG_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
        "GRAPHRAG_EMBEDDING_API_BASE": "https://embedding.test/v1"}.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "run"
    pilot.prepare(out, dataset=dataset, horizon=100, methods=("FULL",), gate_policy="residual_mass", prompt_profile=prompt_profile)
    m = json.loads((out / "manifest.json").read_text())
    assert m["dataset"] == dataset and m["methods"] == ["FULL"] and m["max_workers"] == 3
    assert m["bnrr_gate_policy"] == "residual_mass" and m["horizon"] == 100
    assert m["parser_by_method"] == {"FULL": "bnrr"}
    assert m["bnrr_parser_protocol"] == "bnrr-null-record-local"
    assert m["retrieval_by_method"] == {"FULL": "bnrr-pure-retrieval-v1"}
    assert m["bnrr_prompt_profile"] == prompt_profile
    assert m["query_memory_profile"] == pilot.profile_settings(prompt_profile)["memory_profile"]
    assert m["initialization_prompt_profile"] == prompt_profile
    assert pilot.check_manifest(out) == m
    original_retrieval = m['retrieval_by_method']['FULL']
    m['retrieval_by_method']['FULL'] = 'legacy-combined'
    (out / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='extraction contract'):
        pilot.check_manifest(out)
    m['retrieval_by_method']['FULL'] = original_retrieval
    m["prompt_files"]["seed"] = "incorrect-file.txt"
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="prompt file mapping"):
        pilot.check_manifest(out)
    assert (out / "graph_root/output").resolve() == source / "output"
    model = yaml.safe_load((out / "graph_root/settings.yaml").read_text())["models"]["default_chat_model"]
    assert model["type"] == "openai_chat" and model["max_tokens"] == 16384
    assert model["temperature"] == 0 and model["top_p"] == 1
    assert (source / "settings.yaml").read_bytes() == original


def test_new_protocol_cannot_reuse_old_combined_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'environment', lambda: None)
    source = tmp_path/'old'; source.mkdir()
    (source/'manifest.json').write_text(json.dumps({'dataset': 'novel_9'}))
    with pytest.raises(ValueError, match='legacy combined-query seeds'):
        pilot.prepare(tmp_path/'new', reuse_seeds=source)
    assert not (tmp_path/'new').exists()
