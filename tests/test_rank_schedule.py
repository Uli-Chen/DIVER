from fractions import Fraction
import json
import math
import networkx as nx
import pytest
from extraction.control.rank_bnrr import RankBnrrController, admission, fraction_at, specification
from extraction import experiment as pilot
from test_bnrr_resume import rig, InjectedCrash


@pytest.mark.parametrize('direction,start,end', [('relax',.1,.9),('tighten',.9,.1)])
def test_endpoints_and_monotonicity(direction,start,end):
    fractions=[fraction_at(t,100,direction) for t in range(2,101)]
    assert float(fractions[0])==start and float(fractions[-1])==end
    assert all((b-a)*(1 if direction=='relax' else -1)>0 for a,b in zip(fractions,fractions[1:]))
    assert fraction_at(2,2,direction)==fractions[0]


def test_all_ties_receive_equal_fraction_not_all_full_admission():
    membership,k,boundary,weight=admission({str(n):1. for n in range(50)},Fraction(1,10))
    assert k==5 and boundary==1 and weight==.1
    assert set(membership.values())=={.1}
    assert admission({},Fraction(1,10))==({},0,None,0.)
    # Exact fraction arithmetic at a ceil boundary.
    assert admission({str(n):1. for n in range(30)},Fraction(1,10))[1]==3


def novel_seed_graph():
    graph=nx.Graph()
    # Degree multiset / BNRR: 45 ones plus 2,3,7,31, matching Novel initialization.
    for hub,degree in [('A',31),('B',7),('C',3),('D',2)]:
        for i in range(degree):graph.add_edge(hub,f'{hub}_{i}')
    graph.add_edge('E','F')
    return graph


@pytest.mark.parametrize('direction,k,p', [('relax',5,.5),('tighten',45,84/88)])
def test_initial_graph_admission_and_gate(direction,k,p):
    c=RankBnrrController(100,direction=direction)
    d=c.decide(novel_seed_graph(),2)
    assert d['fresh_size']==49 and d['rank_quota']==k
    assert math.isclose(d['effective_eligible_size'],k)
    assert math.isclose(d['exploit_probability'],p)
    assert d['total_bnrr_mass']==88
    assert d['admission_weights']['A']==1
    expected=(k-4)/45
    assert math.isclose(d['admission_weights']['E'],expected)
    assert d['eligible_sampling_weights']['E']==expected


def test_weighted_gate_and_anchor_draw_use_different_memberships():
    from extraction.control.bnrr import keyed_rng, weighted_choice
    graph=novel_seed_graph();seen=set()
    for seed in range(100):
        c=RankBnrrController(100,seed=seed,direction='relax')
        c.exposure={'A':3.,'E':1.}
        d=c.decide(graph,2)
        g,w,pool=d['admission_weights'],d['sampling_weights'],d['fresh']
        expected=sum(g[v]*w[v]*s['bnrr'] for v,s in pool.items())/sum(s['bnrr'] for s in pool.values())
        assert math.isclose(d['exploit_probability'],expected)
        if d['mode']=='exploit':
            seen.add(d['branch'])
            labels=list(pool) if d['branch']=='random_soft' else d['eligible']
            weights=w if d['branch']=='random_soft' else d['eligible_sampling_weights']
            assert d['anchor']==weighted_choice(labels,weights,keyed_rng(seed,2,'anchor'))
            before=dict(c.exposure);c.skip(d['anchor']);assert c.exposure==before
    assert seen=={'random_soft','rank_eligible_soft'}


def test_empty_pool_and_moderation_override_preserve_contract():
    c=RankBnrrController(100)
    assert c.decide(nx.Graph(),2)['mode']=='explore'
    d=c.decide(novel_seed_graph(),3,moderation_excluded={'A','B','C'})
    assert d['branch']=='moderation_random_anchor' and d['anchor'] not in {'A','B','C'}
    assert d['exploit_probability']==1 and 0<d['bnrr_exploit_probability']<1
    c.complete(d['anchor'])
    assert c.exposure[d['anchor']]==1


@pytest.mark.parametrize('direction',['relax','tighten'])
@pytest.mark.parametrize('phase',['response_saved','graph_delta.jsonl','commit'])
def test_rank_worker_resume_matches_uninterrupted(rig,direction,phase):
    state,prepare=rig
    state['manifest']['rank_schedule']=specification(direction)
    state['record_local']=False
    base,recovered=prepare('base'),prepare('recovered')
    pilot.worker(base,'FULL',42)
    state.update(fail=phase,fired=False,writer_calls=[],extraction_calls=[])
    with pytest.raises(InjectedCrash):pilot.worker(recovered,'FULL',42)
    pilot.worker(recovered,'FULL',42,resume=True)
    for name in ['history.jsonl','query_generation.jsonl','decisions.jsonl','graph_delta.jsonl','final_graph.graphml']:
        assert (base/'runs/FULL_seed42'/name).read_bytes()==(recovered/'runs/FULL_seed42'/name).read_bytes(),name
    assert pilot.audit_run(recovered,'FULL',42)['rounds']==6


@pytest.mark.parametrize('direction',['relax','tighten'])
@pytest.mark.parametrize('stage',['query_generation','extraction'])
def test_rank_moderation_skips_then_recovers(rig,monkeypatch,direction,stage):
    from extraction import moderation
    state,prepare=rig
    state['manifest'].update(rank_schedule=specification(direction),moderation_policy=moderation.POLICY,moderation_attempts=3)
    state.update(record_local=False,moderation_stage=stage,moderation_turns={3})
    monkeypatch.setattr(moderation.time,'sleep',lambda _:None)
    root=prepare('moderation')
    pilot.worker(root,'FULL',42)
    run=root/'runs/FULL_seed42'
    history=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()]
    decisions=[json.loads(l) for l in (run/'decisions.jsonl').read_text().splitlines()]
    assert history[2]['parse_stats']['skip_reason']=='content_moderation'
    assert decisions[3]['branch']=='moderation_random_anchor'
    assert decisions[3]['anchor']!=decisions[2]['anchor']
    assert pilot.audit_run(root,'FULL',42)['request_errors']==3


def test_new_default_and_frozen_legacy_selection_are_distinct():
    from extraction.bnrr_config import DEFAULTS
    from extraction.control.bnrr import BnrrController
    assert DEFAULTS.rank_schedule == specification('tighten')
    assert RankBnrrController(100).direction == 'tighten'
    assert type(pilot.make_controller({'horizon':100},42)) is BnrrController
    assert isinstance(pilot.make_controller({'horizon':100,'rank_schedule':DEFAULTS.rank_schedule},42),RankBnrrController)


@pytest.mark.parametrize('change',[{'direction':'unknown'},{'start_fraction':.8},{'boundary_ties':'truncate'}])
def test_invalid_frozen_rank_spec_rejected(change):
    from extraction.control.rank_bnrr import validate_spec
    with pytest.raises(ValueError):validate_spec({**specification('tighten'),**change})
