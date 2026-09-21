import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "baselines" / "AGEA" / "tools" / "sync_wandb.py"
SPEC = importlib.util.spec_from_file_location("sync_agea_wandb", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
agea_turn_record = MODULE.agea_turn_record
query_diagnostics = MODULE.query_diagnostics


def test_agea_turn_record_maps_native_and_uniform_metrics() -> None:
    record = agea_turn_record(
        {
            "mode": "exploit",
            "seeds_used": ["ENTITY"],
            "novelty": 0.25,
            "nodes_added_to_graph": 2,
            "edges_added_to_graph": 3,
            "total_nodes_in_graph": 12,
            "total_edges_in_graph": 15,
            "cumulative_metrics": {
                "precision_nodes": 80.0,
                "leakage_rate_nodes": 20.0,
                "precision_edges": 60.0,
                "leakage_rate_edges": 10.0,
            },
        },
        {"node_f1": 0.4, "edge_pair_f1": 0.2},
        7,
    )
    assert record["turn"] == 7
    assert record["anchor"] == "ENTITY"
    assert record["batch"]["native_node_precision"] == 0.8
    assert record["batch"]["native_edge_recall"] == 0.1
    assert record["evaluation"]["node_f1"] == 0.4


def test_agea_query_diagnostics_counts_zero_gain() -> None:
    diagnostics = query_diagnostics(
        [
            {"nodes_added_to_graph": 1, "edges_added_to_graph": 0},
            {"nodes_added_to_graph": 0, "edges_added_to_graph": 0},
        ]
    )
    assert diagnostics == {
        "total_queries": 2,
        "zero_gain_turns": 1,
        "zero_gain_rate": 0.5,
    }
