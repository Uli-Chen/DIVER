import json
from pathlib import Path

import networkx as nx
import pytest

from extraction.bnrr_parser import parse_response, extraction_finish_reasons
from extraction.models import CandidateBatch
from extraction.metrics.graph import merge_batch

VALID = "ENTITY: A\nDescription: null\nRelationships:\nSource: A\nTarget: B\nDescription: A supports B.\n"


@pytest.mark.parametrize("value", ["null", "None", "(none)", "[N/A]", "", "No description provided.", "[Data: Entities (1)]"])
def test_null_description_keeps_named_nodes_and_edges(value):
    nodes, edges, stats = parse_response(f"ENTITY: A\nDescription: {value}\nSource: A\nTarget: B\nDescription: {value}")
    assert nodes[0]["description"] is edges[0]["description"] is None
    assert stats["parse_status"] == "parsed"
    assert json.loads(json.dumps(nodes))[0]["description"] is None


@pytest.mark.parametrize("bad", [
    "Source: null\nTarget: B\nDescription: conflicting positive prose.",
    "Source: A\nTarget: null\nDescription: null",
    "Source: A",
    "Target: B\nDescription: orphan target.",
    "Source: X\nTarget: Y\nDescription: one\nDescription: two",
    "Source: X, Target: Y Description: malformed",
])
def test_bad_record_does_not_discard_neighbors_or_inherit_across_boundary(bad):
    nodes, edges, stats = parse_response(VALID + "### Next block\n" + bad + "\nENTITY: C\nSource: C\nTarget: D\nDescription: null")
    assert {(e["source"], e["target"]) for e in edges} == {("A", "B"), ("C", "D")}
    assert {n["label"] for n in nodes} == {"A", "C"}
    assert stats["parse_status"] == "partial"
    assert stats["relationship_records_skipped"] == 1


def test_numbered_explanations_cannot_become_entities_or_leak_descriptions():
    text = "1. **As a Target:** explanation\nSource: X\nTarget: Y\nDescription: edge evidence\n"
    text += "2. **As a Source:** explanation\n1. **ENTITY:** A\nDescription: null\nRelationships:\nSource: A\nTarget: B\nDescription: AB"
    nodes, edges, _ = parse_response(text)
    assert [n["label"] for n in nodes] == ["A"]
    assert nodes[0]["description"] is None
    assert len(edges) == 2


def test_names_preserve_punctuation_unicode_and_real_unknown():
    names = ['AM("ENTITY"', 'UNKNOWN', 'NA', '中文实体', '(Nonesuch)']
    nodes, _, _ = parse_response("\n".join(f"ENTITY: {n}\nDescription: null" for n in names))
    assert [n["label"] for n in nodes] == [n.upper() for n in names]
    nodes, edges, stats = parse_response("ENTITY: null\nSource: null\nTarget: B\nDescription: null")
    assert not nodes and not edges and stats["round_skipped"]


def test_missing_description_and_multiline_values_are_scoped():
    nodes, edges, _ = parse_response("ENTITY: A\nSource: A\nTarget: B\nENTITY: C\nDescription: C first\nC second\nSource: C\nTarget: D\nDescription: D first\nD second")
    assert nodes[0]["description"] is None and edges[0]["description"] is None
    assert nodes[1]["description"] == "C first\nC second"
    assert edges[1]["description"] == "D first\nD second"


def test_scoped_source_inheritance_stops_after_bad_record():
    _, edges, stats = parse_response(VALID + "Target: C\nDescription: null\nTarget: null\nDescription: null\nTarget: D\nDescription: unrelated\nSource: E\nTarget: F")
    assert {(e["source"], e["target"]) for e in edges} == {("A", "B"), ("A", "C"), ("E", "F")}
    assert stats["relationship_records_skipped"] == 2


@pytest.mark.parametrize("line", [
    "Source: A, Target: B, Description: null",
    "- **Source:** A → Target: B — Some evidence.",
])
def test_inline_explicit_endpoints(line):
    _, edges, stats = parse_response(line)
    assert [(e["source"], e["target"]) for e in edges] == [("A", "B")]
    assert stats["parse_status"] == "parsed"


@pytest.mark.parametrize("text", ["", "prose only", VALID])
def test_length_always_skips_whole_reply(text):
    nodes, edges, stats = parse_response(text, finish_reasons=["length"])
    assert not nodes and not edges
    assert stats["round_skipped"] and stats["skip_reason"] == "output_length"


def test_empty_sentinel_and_unparseable_reply_are_distinct():
    assert parse_response("NO_SUPPORTED_RECORDS")[2]["parse_status"] == "explicit_empty"
    assert parse_response("unstructured prose")[2]["parse_status"] == "skipped"
    assert parse_response(VALID + "NO_SUPPORTED_RECORDS")[2]["parse_status"] == "partial"


def test_duplicate_and_graphml_preserve_real_description(tmp_path):
    nodes, edges, _ = parse_response(VALID + "ENTITY: A\nDescription: Real entity\nSource: A\nTarget: B\nDescription: null\nSource: B\nTarget: C\nDescription: null")
    batch = CandidateBatch.from_records(nodes, edges)
    graph = nx.MultiDiGraph()
    merge_batch(graph, batch)
    assert graph.nodes["A"]["description"] == "Real entity"
    assert graph["A"]["B"]["related_to"]["description"] == "A supports B."
    merge_batch(graph, CandidateBatch.from_records([], [{"source": "A", "target": "B", "description": None}]))
    assert graph["A"]["B"]["related_to"]["description"] == "A supports B."
    nx.write_graphml(graph, tmp_path / "graph.graphml")
    saved = nx.read_graphml(tmp_path / "graph.graphml", force_multigraph=True)
    assert saved.has_edge("B", "C", "related_to")
    assert saved["B"]["C"]["related_to"].get("description") is None


def test_completion_uses_current_turn_last_successful_extraction(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [dict(turn=1, stage="extraction", status=200, finish_reasons=["length"]),
        dict(turn=2, stage="query_generation", status=200, finish_reasons=["length"]),
        dict(turn=2, stage="extraction", status=200, finish_reasons=["length"]),
        dict(turn=2, stage="extraction", status=200, finish_reasons=["stop"]),
        dict(turn=2, stage="extraction", status=500, finish_reasons=[])]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert extraction_finish_reasons(path, 2) == ["stop"]


@pytest.mark.parametrize("body", [VALID, ""])
def test_adapter_saved_length_skips_without_requery(monkeypatch, tmp_path, body):
    from extraction.backends import graphrag as g
    calls = []
    def live(**kwargs):
        calls.append(kwargs)
        (tmp_path / "requests.jsonl").write_text(json.dumps(dict(turn=1, stage="extraction", status=200, finish_reasons=["length"])))
        return body, {}
    monkeypatch.setattr(g, "_run_graphrag_local_search", live)
    with g.AgeaGraphRagAdapter(graph_root=str(tmp_path), data_dir=str(tmp_path), run_dir=tmp_path, extraction_parser="bnrr") as adapter:
        result = adapter.query("test", 1)
        replay = adapter.query("test", 1, resume_saved_response=True)
    assert result.stats["skip_reason"] == replay.stats["skip_reason"] == "output_length"
    assert not result.batch.nodes and not replay.batch.edges
    assert len(calls) == 1


def test_actual_novel9_parser_regressions():
    root = Path(__file__).resolve().parents[1] / "tmp/bnrr_soft_agea_prompt_novel9_30r_20260907/novel9_30r/runs"
    from extraction.backends.graphrag import extract_actual_llm_response
    failed = root / "FULL_seed42/turn_logs/llm_responses/first_llm_response_query_2.txt"
    if not failed.exists():
        pytest.skip("Local historical fixture")
    nodes, edges, stats = parse_response(failed.read_text(), extract_body=extract_actual_llm_response)
    assert len(nodes) == 19 and len(edges) == 55
    assert stats["relationship_records_skipped"] == 3
    raw = root / "FULL_seed43/turn_logs/llm_responses/first_llm_response_query_4.txt"
    nodes, _, _ = parse_response(raw.read_text(), extract_body=extract_actual_llm_response)
    assert not any(n["label"].startswith(("AS A TARGET:", "AS A SOURCE:")) for n in nodes)
