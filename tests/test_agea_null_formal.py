import importlib.util
import json
from pathlib import Path
import sys

import networkx as nx
import pytest

from extraction.bnrr_prompt_profiles import load_profile, profile_settings
from extraction.bnrr_parser import parse_response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extraction import cli as formal


def test_seed_and_both_actions_use_one_reviewed_null_contract():
    library = load_profile(ROOT, formal.PROFILE)
    assert len(library.paths) == 7
    assert all(p.parent.name == "bnrr" for p in library.paths.values())
    assert profile_settings(formal.PROFILE)["memory_profile"] == "recent_exclusion_desc"
    graph = nx.MultiDiGraph([("A", "B")])
    for turn, mode, anchor, kwargs in [(1, "explore", None, {}),
            (2, "explore", None, {"explore_query": "Other subjects?"}),
            (3, "exploit", "A", {"exploit_query": "What involves A?"})]:
        rendered = formal.metered.pilot.render(library, graph, turn, mode, anchor, [], **kwargs)
        assert rendered.split("\n\n", 1)[1] == library.templates["output_contract"]
        assert "unquoted lowercase literal null" in rendered
        assert "Omit unavailable descriptions" not in rendered
        assert "complete, un-summarized descriptions" in rendered
    assert "one short identifying phrase" in library.templates["exploit_query_generator"]


def test_null_contract_example_does_not_create_null_nodes():
    body = "ENTITY: A\nDescription: null\nSource: A\nTarget: B\nDescription: null\nENTITY: null\nDescription: null\nSource: null\nTarget: B\nDescription: null"
    nodes, edges, stats = parse_response(body)
    assert [n["label"] for n in nodes] == ["A"]
    assert [(e["source"], e["target"], e["description"]) for e in edges] == [("A", "B", None)]
    assert stats["entity_records_skipped"] == stats["relationship_records_skipped"] == 1


@pytest.mark.parametrize("dataset,seeds", [
    ("agriculture", [42, 43, 44]), ("novel", [42, 44]), ("novel", [42]), ("medical", [42, 43, 44])])
@pytest.mark.parametrize("rounds", [100, 1000])
def test_formal_prepare_freezes_initialization_and_soft_controller(monkeypatch, tmp_path, dataset, seeds, rounds):
    import lancedb
    import yaml
    from types import SimpleNamespace
    source = tmp_path / "indices" / dataset
    (source / "output/lancedb").mkdir(parents=True)
    (source / "prompts").mkdir()
    (source / "prompts/local_search_system_prompt.txt").write_text("{context_data}")
    (source / "settings.yaml").write_text(yaml.safe_dump({"models": {"default_chat_model": {}},
        "vector_store": {"default_vector_store": {"db_uri": "output/lancedb"}}}))
    pilot = formal.metered.pilot
    monkeypatch.setattr(pilot, "environment", lambda: None)
    monkeypatch.setattr(pilot, "SOURCE_GRAPH", tmp_path / "indices/novel_9")
    monkeypatch.setattr(lancedb, "connect", lambda _: SimpleNamespace(open_table=lambda _: SimpleNamespace(
        schema=SimpleNamespace(field=lambda _: SimpleNamespace(type=SimpleNamespace(list_size=4096))))))
    for key in ("AGEA_API_BASE", "GRAPHRAG_EMBEDDING_API_BASE", "GRAPHRAG_EMBEDDING_MODEL"):
        monkeypatch.setenv(key, "test")
    if dataset == "medical":
        monkeypatch.setenv("AGEA_API_BASE", "https://api.deepseek.com")
    root = tmp_path / f"{dataset}_{rounds}r"
    formal.prepare(root, dataset, seeds, rounds=rounds)
    manifest, _ = formal.check(root)
    assert manifest["initialization_prompt_profile"] == formal.PROFILE
    assert manifest["seeds"] == seeds and manifest["horizon"] == rounds
    controller = pilot.make_controller(manifest, seeds[0])
    assert controller.horizon == rounds
    from extraction.control.rank_bnrr import RankBnrrController, specification, fraction_at
    assert isinstance(controller, RankBnrrController)
    assert controller.direction == "tighten"
    assert manifest["rank_schedule"] == specification("tighten")
    graph = nx.star_graph(["A", "B", "C", "D", "E"])
    first = controller.decide(graph, 2)
    assert first["rank_fraction"] == .9 and first["rank_quota"] == 5
    last = pilot.make_controller(manifest, seeds[0]).decide(graph, rounds)
    assert last["rank_fraction"] == .1 and last["rank_quota"] == 1
    if rounds == 1000:
        assert fraction_at(100, rounds, "tighten") > fraction_at(100, 100, "tighten")
    assert manifest["max_workers"] == len(seeds)
    assert [r["seed"] for r in formal.progress(root)] == seeds
    assert manifest["coverage_protocol"] == "bnrr-soft-self-onehop"
    if dataset == "medical":
        assert manifest["thinking_control"] == {"thinking": {"type": "disabled"}}
        wrong_provider = dict(manifest, chat_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
        (root / "manifest.json").write_text(json.dumps(wrong_provider))
        with pytest.raises(RuntimeError, match="chat_api_base"):
            formal.check(root)
    changed = dict(manifest, initialization_prompt_profile="legacy")
    (root / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="initialization_prompt_profile"):
        formal.check(root)


@pytest.mark.parametrize("seeds", [[], [42, 42], [42, 45], [42, 43, 44, 42]])
def test_seed_selection_rejects_invalid_cohorts(seeds):
    with pytest.raises(ValueError):
        formal.validate_seeds(seeds)


def test_dispatch_only_opens_selected_seed_windows(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(formal, "check", lambda root: ({"seeds": [42, 44]}, {}))
    monkeypatch.setattr(formal.subprocess, "run", lambda command, **kwargs: calls.append(command))
    formal.dispatch(tmp_path / "novel_100r", "novel-two-seeds")
    windows = [command[command.index("-n") + 1] for command in calls if command[1] == "new-window"]
    assert windows == ["seed42", "seed44"]


def test_probe_can_reuse_valid_reply_with_duplicate_relationship(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import openai
    from extraction.request_audit import RequestAudit
    out = tmp_path / "availability"
    out.mkdir()
    library = load_profile(ROOT, formal.PROFILE)
    prompt = ("Evidence: ENTITY ALPHA has no supplied description. ENTITY BETA has no supplied description. "
        "A directed relationship ALPHA to BETA is supplied, with no relationship description.\n\n" + library.templates["output_contract"])
    (out / "prompt.txt").write_text(prompt)
    relation = "Relationships:\n  - Source: ALPHA\n  - Target: BETA\n  - Description: null\n"
    (out / "response.txt").write_text("ENTITY: ALPHA\nDescription: null\n" + relation + "ENTITY: BETA\nDescription: null\n" + relation)
    row = {"request_index": 1, "kind": "chat", "status": 200, "model": "deepseek-v4-flash",
        "host": "api.deepseek.com", "finish_reasons": ["stop"], "usage": {"prompt_tokens": 386, "completion_tokens": 66},
        "max_tokens": 1024, "thinking_disabled": True, "reasoning_content_nonempty": False}
    (out / "requests.jsonl").write_text(json.dumps(row) + "\n")
    for key, value in {"AGEA_API_BASE": "https://api.deepseek.com", "GRAPHRAG_CHAT_MODEL": "deepseek-v4-flash",
                       "GRAPHRAG_EMBEDDING_API_KEY": "test", "GRAPHRAG_EMBEDDING_API_BASE": "https://embedding.test",
                       "GRAPHRAG_EMBEDDING_MODEL": "test"}.items():
        monkeypatch.setenv(key, value)
    calls = []
    class EmbeddingOnlyClient:
        def __init__(self, **kwargs):
            calls.append(kwargs["base_url"])
            self.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[SimpleNamespace(embedding=[0] * 4096)]))
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(openai, "OpenAI", EmbeddingOnlyClient)
    monkeypatch.setattr(RequestAudit, "install", lambda self: None)
    monkeypatch.setattr(formal.metered, "environment", lambda: None)
    formal.availability(tmp_path, reuse_response=True)
    assert read_json(out / "RESULT.json")["status"] == "passed"
    assert calls == ["https://embedding.test"]


def read_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("rounds", [100, 1000])
def test_launch_freezes_source_and_dispatches_frozen_entry(monkeypatch, tmp_path, rounds):
    project = tmp_path / 'project'
    for folder in ('src', 'scripts', 'configs', 'baselines/AGEA'):
        (project/folder).mkdir(parents=True)
        (project/folder/'marker.txt').write_text(folder)
    for name in ('pyproject.toml', 'uv.lock', '.env'):
        (project/name).write_text('private-test-credential' if name=='.env' else name)
    monkeypatch.setattr(formal, 'PROJECT', project)
    calls = []
    monkeypatch.setattr(formal.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    cohort = project/'tmp/new_run'
    formal.launch(cohort, 'novel', [42], rounds=rounds)
    assert calls[0][0] == ['rmux', '-V']
    assert (cohort/'workspace/.env').is_symlink()
    import tarfile
    with tarfile.open(cohort/'source_snapshot.tar.gz') as archive:
        assert '.env' not in archive.getnames()
    for command, kwargs in calls[1:]:
        assert str(cohort/'workspace/scripts/run_bnrr.py') in command
        assert kwargs['cwd'] == cohort/'workspace'
    assert calls[-1][0][-1] == 'bnrr-novel-new_run'
    prepare_command = calls[1][0]
    assert prepare_command[prepare_command.index('--rounds') + 1] == str(rounds)
    assert str(cohort/f'novel_{rounds}r') in prepare_command
    assert str(cohort/f'novel_{rounds}r') in calls[-1][0]
    assert read_json(cohort/'COHORT.json')['rounds'] == rounds
    with pytest.raises(FileExistsError):
        formal.launch(cohort, 'novel', [42])


@pytest.mark.parametrize('rounds', [None, True, 1, 0, -1, 1000.0])
def test_invalid_round_budget_rejected(rounds):
    with pytest.raises(ValueError, match='Round budget'):
        formal.validate_rounds(rounds)
