import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
from extraction import experiment as pilot

class InjectedCrash(BaseException):
    """Like process interruption, not an API exception caught by adapter retries."""


@pytest.fixture(params=["agea_null"], ids=["bnrr"])
def rig(monkeypatch, tmp_path, request):
    from extraction.backends import graphrag as b
    from extraction.metrics.graph import compute_node_scores
    from extraction.request_audit import RequestAudit
    from extraction.resume import RoundJournal
    from evaluation.graph_recovery import TruthData
    from extraction.bnrr_queries import _generate, exploration_messages, exploitation_messages
    manifest = {"horizon": 6, "bnrr_gate_policy": "residual_mass", "bnrr_prompt_profile": "agea_adapted_null",
        "initialization_prompt_profile": "agea_adapted_null", "query_memory_profile": "recent_exclusion_desc",
        "prompt_protocol": "bnrr-pure-retrieval-null", "parser_by_method": {"FULL": "bnrr"},
        "history_observation_protocol": "response-visible-v1", "coverage_protocol": "bnrr-soft-self-onehop",
        "retrieval_by_method": {"FULL": "bnrr-pure-retrieval-v1"}, "seed_retrieval_protocol": "bnrr-pure-retrieval-v1"}
    g = nx.DiGraph([("A", f"N{x}") for x in range(1, 7)])
    truth = TruthData(g, set(g), set(g.edges()), {}, compute_node_scores(g), nx.pagerank(g))
    state = {"writer_calls": [], "extraction_calls": [], "fail": None, "fired": False,
        "record_local": request.param in ("record_local", "agea_null"), "manifest": manifest,
        "moderation_stage": None, "moderation_turns": set()}
    monkeypatch.setattr(pilot, "environment", lambda: None)
    monkeypatch.setattr(pilot, "check_manifest", lambda root: manifest)
    monkeypatch.setattr(pilot, "RequestAudit", RequestAudit)
    monkeypatch.setattr(RequestAudit, "install", lambda self: state.update(audit=self))
    monkeypatch.setattr(TruthData, "load", lambda path: truth)
    monkeypatch.setenv("AGEA_API_KEY", "test")
    monkeypatch.setenv("AGEA_API_BASE", "https://example.invalid/v1")

    def fail(phase, turn):
        if state['fail'] == phase and turn == 3 and not state['fired']:
            state['fired'] = True
            raise InjectedCrash('injected crash')

    def charge(kind, cap=16384, error=None, finish='stop'):
        audit = state['audit']
        request = httpx.Request('POST', 'https://example.invalid/v1/' + ('embeddings' if kind == 'embedding' else 'chat/completions'),
            json={'enable_thinking': False, 'model': 'test', 'max_tokens': cap})
        if error == 'data_inspection_failed':
            audit.finish(audit.start(request), httpx.Response(400,
                json={'error': {'code': 'data_inspection_failed'}}))
            return
        audit.finish(audit.start(request), None if error else httpx.Response(200,
            json={'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                'choices': [{'finish_reason': finish, 'message': {'content': 'ok'}}]}), error=error)

    def moderate(stage, turn, cap):
        if state['moderation_stage'] == stage and turn in state['moderation_turns']:
            from openai import BadRequestError
            charge('chat', cap, error='data_inspection_failed')
            raise BadRequestError('Input data may contain inappropriate content.',
                response=httpx.Response(400, request=httpx.Request('POST', 'https://example.invalid')),
                body={'error': {'code': 'data_inspection_failed'}})

    def writer(library, graph, history, **kwargs):
        turn = len(history) + 1
        anchor = kwargs.get('anchor')
        mode = 'exploit' if anchor else 'explore'
        messages = (exploitation_messages(library, graph, history, anchor, memory_profile=kwargs['memory_profile']) if anchor
                    else exploration_messages(library, graph, history, memory_profile=kwargs['memory_profile']))
        state['writer_calls'].append(turn)
        moderate('query_generation', turn, 1024)
        finish = 'length' if state['record_local'] and turn == 4 else 'stop'
        charge('chat', 1024, finish=finish)
        choice = SimpleNamespace(message=SimpleNamespace(content=f'What is associated with {anchor or "subject"} in step {turn}?', reasoning_content=None), finish_reason=finish)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[choice]))))
        return _generate(messages, client=client, model=kwargs['model'], completion_options={},
            temperature=.2 if anchor else .3, action=mode, anchor=anchor,
            anchor_protocol='anchor-single-quote-escape-v1', observed_labels=graph.nodes)

    def body(turn):
        description = '' if request.param == 'empty_descriptions' and turn in (1, 3) else f'A connects to N{turn}.'
        extra = 'Source: null\nTarget: N99\nDescription: null\n' if state['record_local'] and turn == 5 else ''
        return f'ENTITY: A\nDescription: A.\nRelationships:\n  - Source: A\n  - Target: N{turn}\n  - Description: {description}\n' + extra

    def live(**kw):
        turn = kw['turn']
        fail('before_extraction', turn)
        if state['fail'] == 'transport' and turn == 3 and not state['fired']:
            charge('embedding', error='ConnectError')
            fail('transport', turn)
        state['extraction_calls'].append(turn)
        charge('embedding')
        moderate('extraction', turn, 16384)
        charge('chat', finish='length' if state['record_local'] and turn == 3 else 'stop')
        response_path = kw['response_dir'] / f'first_llm_response_query_{turn}.txt'
        context_path = kw['context_dir'] / f'retrieved_context_query_{turn}.json'
        response_path.write_text(b._format_response_artifact(kw['query'], body(turn)))
        context = {'capture_status': 'captured', 'turn': turn, 'query': kw['query']}
        if request.param:
            assert kw['retrieval_query'] == kw['query'].split('\n\n', 1)[0]
            context.update(retrieval_query=kw['retrieval_query'], generation_query=kw['query'],
                retrieval_protocol='bnrr-pure-retrieval-v1')
        context_path.write_text(json.dumps(context))
        fail('response_saved', turn)
        return body(turn), str(context_path), str(response_path)

    monkeypatch.setattr(pilot, 'generate_explore_query', writer)
    monkeypatch.setattr(pilot, 'generate_exploit_query', writer)
    monkeypatch.setattr(b, '_run_live_local_query', live)
    original_append, original_commit = RoundJournal.append, RoundJournal.commit
    def appended(self, name, row):
        original_append(self, name, row)
        fail(name, row['turn'])
    def committed(self, turn, write):
        original_commit(self, turn, write)
        fail('commit', turn)
    monkeypatch.setattr(RoundJournal, 'append', appended)
    monkeypatch.setattr(RoundJournal, 'commit', committed)
    def prepare(name):
        path = tmp_path / name
        path.mkdir()
        (path/'manifest.json').write_text(json.dumps(manifest))
        seed = path/'seed_generation/seed42'
        seed.mkdir(parents=True)
        (seed/'response.txt').write_text(body(1))
        if request.param:
            (seed/'retrieval_query.txt').write_text(pilot.load_profile(pilot.PROJECT).templates['seed'])
        return path
    return state, prepare


@pytest.mark.parametrize('phase', ['query_generation.jsonl', 'decisions.jsonl', 'before_extraction',
    'transport', 'response_saved', 'graph_delta.jsonl', 'history.jsonl', 'metrics.jsonl', 'commit'])
def test_crash_resume_matches_uninterrupted(rig, phase):
    state, prepare = rig
    base, recovered = prepare('base'), prepare('recovered')
    pilot.worker(base, 'FULL', 42)
    state.update(fail=phase, fired=False, writer_calls=[], extraction_calls=[])
    with pytest.raises(InjectedCrash, match='injected crash'):
        pilot.worker(recovered, 'FULL', 42)
    pilot.worker(recovered, 'FULL', 42, resume=True)
    a, z = [p/'runs/FULL_seed42' for p in (base, recovered)]
    for name in ('decisions.jsonl', 'graph_delta.jsonl', 'history.jsonl', 'query_generation.jsonl', 'final_graph.graphml'):
        assert (a/name).read_bytes() == (z/name).read_bytes(), name
    for x, y in zip([json.loads(l) for l in (a/'metrics.jsonl').read_text().splitlines()],
                    [json.loads(l) for l in (z/'metrics.jsonl').read_text().splitlines()]):
        assert all(x[k] == y[k] for k in pilot.METRICS)
    assert state['writer_calls'] == [2, 3, 4, 5, 6]
    assert state['extraction_calls'] == ([2, 3, 5, 6] if state['record_local'] else [2, 3, 4, 5, 6])
    assert json.loads((z/'COMMIT.json').read_text())['round'] == 6
    result = pilot.audit_run(recovered, 'FULL', 42)
    assert result['parser_replay'] == 'pass'
    assert result['request_errors'] == int(phase == 'transport')
    requests = [json.loads(l) for l in (z/'requests.jsonl').read_text().splitlines()]
    assert [x['request_index'] for x in requests] == list(range(1, len(requests)+1))
    if state['record_local']:
        summaries = json.loads((z/'summary.json').read_text())
        assert summaries['skipped_rounds'] == 2 and summaries['partial_rounds'] == 1
        graph = nx.read_graphml(z/'final_graph.graphml')
        assert set(graph) == {'A', 'N1', 'N2', 'N5', 'N6'}
        history = [json.loads(line) for line in (z/'history.jsonl').read_text().splitlines()]
        assert history[2]['parse_stats']['skip_reason'] == 'output_length'
        assert history[3]['parse_stats']['skip_reason'] == 'query_output_length'
        before = json.loads((z/'decisions.jsonl').read_text().splitlines()[2])
        after = json.loads((z/'decisions.jsonl').read_text().splitlines()[3])
        assert before['exposure'] == after['exposure']


def test_terminal_audit_rejects_changed_retrieval_input(rig):
    state, prepare = rig
    root = prepare('retrieval_audit')
    pilot.worker(root, 'FULL', 42)
    context_path = root/'runs/FULL_seed42/turn_logs/retrieved_contexts/retrieved_context_query_2.json'
    context = json.loads(context_path.read_text())
    if 'retrieval_protocol' not in context:
        return
    context['retrieval_query'] += '\n\nUNEXPECTED FORMATTING INSTRUCTIONS'
    context_path.write_text(json.dumps(context))
    with pytest.raises(RuntimeError, match='separation mismatch'):
        pilot.audit_run(root, 'FULL', 42)


def test_seed_generation_then_shared_replay_charges_seed_once(rig):
    state, prepare = rig
    holder = prepare('seed_parent')
    root = holder/'fresh'; root.mkdir()
    (root/'manifest.json').write_bytes((holder/'manifest.json').read_bytes())
    pilot.worker(root, 'FULL', 42, seed_only=True)
    seed = root/'seed_generation/seed42'
    if json.loads((root/'manifest.json').read_text()).get('seed_retrieval_protocol'):
        assert (seed/'retrieval_query.txt').read_text() == pilot.load_profile(pilot.PROJECT).templates['seed']
    pilot.worker(root, 'FULL', 42)
    assert state['extraction_calls'] == ([1, 2, 3, 5, 6] if state['record_local'] else [1, 2, 3, 4, 5, 6])
    assert state['writer_calls'] == [2, 3, 4, 5, 6]
    assert pilot.audit_run(root, 'FULL', 42)['rounds'] == 6


def test_legacy_adoption_and_tamper_rejection(rig):
    state, prepare = rig
    root = prepare('legacy')
    state['fail'] = 'before_extraction'
    with pytest.raises(InjectedCrash): pilot.worker(root, 'FULL', 42)
    out = root/'runs/FULL_seed42'
    (out/'COMMIT.json').unlink()
    pilot.worker(root, 'FULL', 42, resume=True)
    assert pilot.audit_run(root, 'FULL', 42)['rounds'] == 6
    data = (out/'history.jsonl').read_text().replace('step 2', 'step 9')
    (out/'history.jsonl').write_text(data)
    with pytest.raises(RuntimeError, match='Committed log changed'):
        pilot.worker(root, 'FULL', 42, resume=True)


def test_lock_and_scope(tmp_path):
    from extraction.resume import RoundJournal
    journal = RoundJournal(tmp_path, 'a', resume=False)
    with pytest.raises(RuntimeError, match='lock'): RoundJournal(tmp_path, 'a', resume=False)
    journal.close()
    with pytest.raises(ValueError, match='FULL'): pilot.worker(tmp_path, 'AGEA_R', 42, resume=True)


def test_resume_after_failed_resume_keeps_verified_older_commit(tmp_path):
    from extraction.resume import RoundJournal
    (tmp_path/'config.json').write_text(json.dumps({'manifest_sha256':'initial'}))
    journal = RoundJournal(tmp_path, 'last_successful_resume', resume=False)
    for name in ['decisions.jsonl','graph_delta.jsonl','history.jsonl','metrics.jsonl']:
        journal.append(name, {'turn':1})
    journal.commit(1, pilot.write_json)
    journal.close()
    journal = RoundJournal(tmp_path, 'new_resume', resume=True,
        parent_manifest_sha='intervening_failed_resume', config_manifest_sha='initial',
        commit_manifest_sha='last_successful_resume')
    assert journal.completed == 1
    journal.close()


def test_unfinished_request_is_unknown_and_counted_once(tmp_path):
    from extraction.request_audit import RequestAudit
    from extraction.resume import restore_accounting
    audit = RequestAudit(tmp_path/'requests.jsonl')
    audit.journal_starts = True
    audit.start(httpx.Request('POST','https://example.invalid/chat/completions',
        headers={'Authorization':'Bearer do-not-save'}, json={'enable_thinking':False,'max_tokens':1024}))
    restored = RequestAudit(audit.path)
    restore_accounting(restored)
    assert (restored.count,restored.errors,restored.unknown_usage)==(1,1,1)
    assert 'do-not-save' not in (tmp_path/'request_attempts.jsonl').read_text()
    restore_accounting(restored)
    assert restored.count==1


@pytest.mark.parametrize('damage', ['duplicate','gap','torn','manifest'])
def test_invalid_resume_rejected(rig, damage):
    state, prepare = rig
    root=prepare('damaged');state['fail']='before_extraction'
    with pytest.raises(InjectedCrash):pilot.worker(root,'FULL',42)
    out=root/'runs/FULL_seed42'
    if damage=='manifest':
        config=json.loads((out/'config.json').read_text());config['manifest_sha256']='wrong'
        (out/'config.json').write_text(json.dumps(config))
    else:
        path=out/'query_generation.jsonl'
        text=path.read_text()
        if damage=='duplicate':text+=text.splitlines()[-1]+'\n'
        elif damage=='gap':text=text.replace('"turn": 3','"turn": 5')
        else:text+='{"turn":'
        path.write_text(text)
    with pytest.raises((RuntimeError,ValueError)):pilot.worker(root,'FULL',42,resume=True)


def test_repeat_resume_and_completed_no_api(rig):
    state,prepare=rig;root=prepare('repeat')
    for phase in ('before_extraction','response_saved'):
        state.update(fail=phase,fired=False)
        with pytest.raises(InjectedCrash):pilot.worker(root,'FULL',42,resume=phase=='response_saved')
    state['fail']=None
    pilot.worker(root,'FULL',42,resume=True)
    before=(list(state['writer_calls']),list(state['extraction_calls']))
    pilot.worker(root,'FULL',42,resume=True)
    assert before==(state['writer_calls'],state['extraction_calls'])
    assert state['writer_calls']==[2,3,4,5,6]
    assert pilot.audit_run(root,'FULL',42)['rounds']==6


@pytest.mark.parametrize('stage', ['extraction', 'query_generation'])
@pytest.mark.parametrize('phase', ['failure_saved', 'graph_delta.jsonl', 'commit'])
def test_moderation_skip_random_anchor_and_resume(rig, monkeypatch, stage, phase):
    from extraction import moderation
    from extraction.resume import rows
    state, prepare = rig
    state['manifest'].update(moderation_policy=moderation.POLICY, moderation_attempts=3)
    state.update(record_local=False, moderation_stage=stage, moderation_turns={3})
    monkeypatch.setattr(moderation.time, 'sleep', lambda _: None)
    base, recovered = prepare('moderation_base'), prepare('moderation_resume')
    pilot.worker(base, 'FULL', 42)
    state.update(fail=phase, fired=False, writer_calls=[], extraction_calls=[])
    original = moderation.save_failure
    def save_then_interrupt(*args, **kwargs):
        record = original(*args, **kwargs)
        if phase == 'failure_saved' and not state['fired']:
            state['fired'] = True
            raise InjectedCrash('after durable failure')
        return record
    monkeypatch.setattr(moderation, 'save_failure', save_then_interrupt)
    with pytest.raises(InjectedCrash):
        pilot.worker(recovered, 'FULL', 42)
    pilot.worker(recovered, 'FULL', 42, resume=True)
    a, b = [p/'runs/FULL_seed42' for p in (base, recovered)]
    for name in ['history.jsonl', 'decisions.jsonl', 'graph_delta.jsonl', 'query_generation.jsonl', 'final_graph.graphml']:
        assert (a/name).read_bytes() == (b/name).read_bytes(), name
    history, decisions = rows(b/'history.jsonl'), rows(b/'decisions.jsonl')
    assert history[2]['parse_stats']['skip_reason'] == 'content_moderation'
    assert history[2]['nodes_added_to_graph'] == history[2]['edges_added_to_graph'] == 0
    assert decisions[3]['branch'] == 'moderation_random_anchor'
    assert decisions[3]['anchor'] != decisions[2]['anchor']
    assert decisions[3]['exposure'] == decisions[2]['exposure']
    assert decisions[4]['branch'] != 'moderation_random_anchor'
    assert len([r for r in rows(b/'requests.jsonl') if r.get('error') == 'data_inspection_failed']) == 3
    assert state['writer_calls'].count(3) == (3 if stage == 'query_generation' else 1)
    assert state['extraction_calls'].count(3) == (3 if stage == 'extraction' else 0)
    audit = pilot.audit_run(recovered, 'FULL', 42)
    assert audit['rounds'] == 6 and audit['request_errors'] == 3
    assert rows(b/'metrics.jsonl')[2]['request_errors'] == 3
    sidecar = moderation.failure_path(b, stage, 3)
    data = json.loads(sidecar.read_text());data['request_identity'] = 'tampered'
    sidecar.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='Moderation failure evidence'):
        pilot.audit_run(recovered, 'FULL', 42)


def test_moderated_initialization_can_continue_with_empty_graph(rig, monkeypatch):
    from extraction import moderation
    from extraction.resume import rows
    import shutil
    state, prepare = rig
    state['manifest'].update(moderation_policy=moderation.POLICY, moderation_attempts=3)
    state.update(record_local=False, moderation_stage='extraction', moderation_turns={1})
    monkeypatch.setattr(moderation.time, 'sleep', lambda _: None)
    root = prepare('empty_seed')
    shutil.rmtree(root/'seed_generation/seed42')
    pilot.worker(root, 'FULL', 42, seed_only=True)
    assert (root/'seed_generation/seed42/response.txt').read_text() == ''
    state['extraction_calls'] = []
    pilot.worker(root, 'FULL', 42)
    run = root/'runs/FULL_seed42'
    assert 1 not in state['extraction_calls']
    decisions = rows(run/'decisions.jsonl')
    assert decisions[1]['branch'] == 'moderation_no_other_anchor'
    assert decisions[1]['mode'] == 'explore'
    assert pilot.audit_run(root, 'FULL', 42)['rounds'] == 6
