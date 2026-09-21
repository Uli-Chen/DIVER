from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import networkx as nx
import pytest

from extraction.bnrr_prompt_profiles import load_profile
from extraction.bnrr_queries import generate_exploit_query, exploration_messages, exploitation_messages
from extraction.query_anchor import canonicalize_anchor
from extraction.query_described_exclusion import exploration_context, exploitation_context

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = 'AGRICULTURE, CHANGE AND("ENTITY"'


@pytest.mark.parametrize('anchor', [ANCHOR, 'FARMER\'S MARKET', 'HONEY BEE', 'X("RELATIONSHIP"'])
def test_only_anchor_quote_escaping_and_case_are_normalized(anchor):
    escaped = anchor.replace('"', '\\"').replace("'", "\\'").lower()
    query = f'What connects {escaped} to others? Keep unrelated \\"literal\\" as is.'
    expected = f'What connects {anchor} to others? Keep unrelated \\"literal\\" as is.'
    actual, audit = canonicalize_anchor(query, anchor, [anchor])
    assert actual == expected and audit['changed']


@pytest.mark.parametrize('query', ['What connects AGRICULTURE to others?',
    'What connects AGRICULTURE, CHANGE AND("ENTITIES" to others?',
    'What connects AGRICULTURE, CHANGE AND(\\u0022ENTITY\\u0022 to others?',
    'What connects AGRICULTURE, CHANGE AND("ENTITY"EXTRA to others?',
    'What connects AGRICULTURE, CHANGE AND(\\\\"ENTITY\\\\" to others?'])
def test_renamed_truncated_or_arbitrarily_escaped_anchors_still_fail(query):
    with pytest.raises(ValueError, match='fixed anchor'):
        canonicalize_anchor(query, ANCHOR, [ANCHOR])


def test_escape_collision_with_other_observed_entity_is_rejected():
    with pytest.raises(ValueError, match='ambiguous'):
        canonicalize_anchor('What involves A\\"B?', 'A"B', ['A"B', 'A\\"B'])


def test_replay_actual_failure_and_preserve_old_protocol():
    path = ROOT/'tmp/bnrr_agriculture_recent_exclusion_100r_20260906/agriculture_100r/runs/FULL_seed44/query_generation_failure.json'
    # Self-contained fixture remains usable if temporary experiment artifacts move.
    raw = f'What are all direct relationships involving "{ANCHOR.replace(chr(34), chr(92)+chr(34))}" in both directions?'
    if path.exists():
        raw = json.loads(path.read_text())['audit']['raw_query']
    graph = nx.MultiDiGraph();graph.add_node(ANCHOR, description='')
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=raw, reasoning_content=None), finish_reason='stop')])
    query, audit = generate_exploit_query(load_profile(ROOT), graph, [],
        anchor=ANCHOR, client=client, model='test', completion_options={}, memory_profile='recent_exclusion_desc')
    assert ANCHOR in query and '\\"' not in query
    assert audit['raw_query'] == raw and audit['anchor_normalization']['changed']
    assert client.chat.completions.create.call_count == 1


def test_literal_anchor_at_response_end_retains_its_quote():
    graph = nx.MultiDiGraph();graph.add_node(ANCHOR)
    client = Mock();client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='Find relationships involving '+ANCHOR, reasoning_content=None), finish_reason='stop')])
    query, audit = generate_exploit_query(load_profile(ROOT), graph, [],
        anchor=ANCHOR,client=client,model='test',completion_options={},memory_profile='recent_exclusion_desc')
    assert query.endswith(ANCHOR) and not audit['anchor_normalization']['changed']


def test_descriptions_restore_semantics_without_truth_or_scores():
    graph = nx.MultiDiGraph()
    graph.add_node('A',description='Observed description.\nRelationships: FORBIDDEN_TAIL',degree='FORBIDDEN_DEGREE')
    for i in range(25):graph.add_edge('A',f'B{i}',description='Observed edge.')
    history = [{'turn':t,'query_intent':f'Question {t}?','returned_entity_names':['A'],
        'newly_discovered_entity_names':['A'], 'truth':'FORBIDDEN_TRUTH','novelty':'FORBIDDEN_NOVELTY'} for t in range(4)]
    context=exploration_context(graph,history)
    assert context['avoid_as_main_topic']==['A']
    assert context['observed_entities']==[{'name':'A','description':'Observed description.'}]
    exploit=exploitation_context(graph,history,'A')
    assert exploit['fixed_target']=='A' and len(exploit['partial_observed_connections'])==20
    assert 'avoid_as_main_topic' not in exploit
    library=load_profile(ROOT)
    for msgs in (exploration_messages(library,graph,history,memory_profile='recent_exclusion_desc'),
                 exploitation_messages(library,graph,history,'A',memory_profile='recent_exclusion_desc')):
        assert 'Observed description.' in str(msgs) and 'FORBIDDEN' not in str(msgs)
    legacy=load_profile(ROOT)
    for key in ('seed','output_contract','global_discovery','incident_expansion','query_generator_system'):
        assert library.templates[key]==legacy.templates[key]
