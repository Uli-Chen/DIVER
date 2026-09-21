import httpx
import networkx as nx
import pytest
from openai import APIError, BadRequestError
from extraction import moderation
from extraction.control.bnrr import BnrrController
from extraction.request_audit import RequestAudit, cost_summary


def test_error_classification_is_specific():
    request = httpx.Request('POST', 'https://example.invalid')
    error = BadRequestError('blocked', response=httpx.Response(400, request=request),
        body={'error': {'code': 'DataInspectionFailed'}})
    assert moderation.is_moderation_error(error)
    assert moderation.is_moderation_error(APIError('Output data may contain inappropriate content.', request=request, body=None))
    assert not moderation.is_moderation_error(ValueError('Output data may contain inappropriate content.'))
    assert not moderation.is_moderation_error(BadRequestError('invalid', response=httpx.Response(400, request=request),
        body={'error': {'code': 'InvalidApiKey'}}))


def test_uniform_fallback_excludes_consecutive_rejections():
    graph = nx.path_graph(['A','B','C','D'])
    history = [{'anchor': a, 'parse_stats': {'skip_reason': 'content_moderation'}} for a in ['A','B','C']]
    c = BnrrController(100, seed=44)
    d = moderation.choose_decision(c, graph, 4, history, True)
    assert d['anchor'] == 'D' and d['moderation_candidates'] == ['D']
    assert d['anchor_sampling'] == 'uniform' and d['exploit_probability'] == 1
    c.skip(d['anchor'])
    assert c.exposure == {}
    history.append({'anchor':'D','parse_stats': {'skip_reason':'content_moderation'}})
    d = moderation.choose_decision(c, graph, 5, history, True)
    assert d['mode'] == 'explore' and d['branch'] == 'moderation_no_other_anchor'


def test_query_retry_can_succeed_and_other_errors_remain_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(moderation.time, 'sleep', lambda _: None)
    calls = []
    def request():
        calls.append(1)
        if len(calls) < 3:
            raise APIError('Input data may contain inappropriate content.', request=httpx.Request('POST', 'https://example.invalid'), body=None)
        return 'ok'
    value, failure = moderation.call(request, tmp_path, 'query_generation', 2, {'query':'test'}, 3)
    assert value == 'ok' and failure is None and len(calls) == 3
    assert not moderation.failure_path(tmp_path, 'query_generation', 2).exists()
    def invalid():raise ValueError('bad configuration')
    with pytest.raises(ValueError, match='configuration'):
        moderation.call(invalid, tmp_path, 'extraction', 3, {}, 3)


def test_sse_moderation_error_counts_with_http_200(tmp_path):
    audit = RequestAudit(tmp_path/'requests.jsonl')
    request = httpx.Request('POST', 'https://example.invalid/chat/completions',
        json={'enable_thinking':False,'model':'test','max_tokens':16384,'stream':True})
    audit.finish(audit.start(request), httpx.Response(200, text='data: {"error":{"code":"data_inspection_failed","message":"private message"}}\n\ndata: [DONE]\n\n'))
    from extraction.resume import rows, restore_accounting
    records = rows(audit.path)
    assert records[0]['error'] == 'data_inspection_failed'
    assert 'private message' not in audit.path.read_text()
    assert audit.errors == cost_summary(records)['stages']['extraction']['errors'] == 1
    restore_accounting(audit)
    assert audit.errors == 1
