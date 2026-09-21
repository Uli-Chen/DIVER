import math
import networkx as nx
import pytest
from extraction.control.bnrr import BnrrController, nearest_rank

def star():
    return nx.star_graph(["A", "B", "C", "D", "E"])


def test_score_scale_and_no_truth_input():
    p = BnrrController(30).snapshot(star())
    assert p["A"]["bnrr"] == 4
    assert all(p[v]["bnrr"] == 1 for v in "BCDE")
    assert math.isclose(BnrrController(30).snapshot(nx.complete_graph(list("ABCD")))["A"]["bnrr"], math.sqrt(3))


def test_seed_and_empty_pool():
    c = BnrrController(30)
    assert c.decide(star(), 1)["mode"] == "explore"
    assert c.calibration_round is None
    assert c.decide(nx.Graph(), 2)["mode"] == "explore"
    assert c.calibration_round is None


def test_quantiles_and_invalid_config():
    assert nearest_rank([1, 1, 1, 1, 4], .9) == 4
    for kwargs in ({"horizon": 0}, {"horizon": 30, "rho": -1}, {"horizon": 30, "q_lo": .99}):
        with pytest.raises(ValueError):
            BnrrController(**kwargs)


def test_residual_mass_gate_is_bnrr_only_and_unsaturated():
    graph = star()
    modes = set()
    for seed in range(30):
        c = BnrrController(30, seed=seed, gate_policy="residual_mass")
        d = c.decide(graph, 2)
        assert d["eligible"] == ["A"]
        assert d["total_bnrr_mass"] == 8
        assert d["eligible_fresh_bnrr_mass"] == 4
        assert d["exploit_probability"] == .5
        assert (d["mode"] == "exploit") == (d["gate_draw"] < .5)
        assert BnrrController(30, seed=seed).decide(graph, 2) == d
        modes.add(d["mode"])
    assert modes == {"explore", "exploit"}



def test_exposure_uses_prequery_neighbors_and_skip_never_charges():
    graph = nx.path_graph(['A', 'B'])
    c = BnrrController(100)
    first = c.decide(graph, 2)
    assert first['mode'] == 'exploit'
    c.skip(first['anchor'])
    assert c.exposure == {} and c.query_counts == {}
    second = c.decide(graph, 2)
    assert second == first
    graph.add_edge(first['anchor'], 'NEW')
    c.complete(first['anchor'])
    assert c.exposure[first['anchor']] == 1
    assert c.exposure[first['pre_query_neighbors'][0]] == .5
    assert 'NEW' not in c.exposure
    assert first['anchor'] in c.snapshot(graph)
    with pytest.raises(ValueError, match='pending'):
        c.complete(first['anchor'])
    third = c.decide(graph, 3)
    assert third['sampling_weights'][first['anchor']] == .5
    assert third['sampling_weights']['NEW'] == 1


def test_prepared_controller_honors_frozen_quantiles_and_mixture():
    from extraction.experiment import make_controller
    controller = make_controller({'horizon': 100, 'q_hi': .8, 'q_lo': .2, 'rho': .1,
        'neighbor_penalty': .25}, 44)
    assert (controller.q_hi, controller.q_lo, controller.rho, controller.neighbor_penalty) == (.8, .2, .1, .25)
