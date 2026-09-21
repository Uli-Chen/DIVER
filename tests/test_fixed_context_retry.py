import hashlib
import json
from pathlib import Path

import httpx
import pytest

from extraction.fixed_context_retry import (extract_stream_text, increase_limit,
    plain_table, reconstruct_request, unchanged_request, wire_bytes)


def test_only_output_limit_changes():
    body={'messages':[{'role':'user','content':'unchanged'}],'max_tokens':16384,
          'thinking':{'type':'disabled'},'stream_options':{'include_usage':True}}
    digest=hashlib.sha256(wire_bytes(body)).hexdigest()
    new=increase_limit(body,digest,32768)
    assert {k:v for k,v in body.items() if k!='max_tokens'}=={k:v for k,v in new.items() if k!='max_tokens'}
    assert body['max_tokens']==16384 and new['max_tokens']==32768
    with pytest.raises(ValueError):increase_limit({**body,'messages':[]},digest,32768)
    with pytest.raises(ValueError):increase_limit(body,digest,65536)


@pytest.mark.parametrize('finish',['stop','length'])
def test_stream_retains_content_and_finish(finish):
    data=[{'choices':[{'delta':{'content':'A'},'finish_reason':None}]},
          {'choices':[{'delta':{'content':'B'},'finish_reason':finish}]},
          {'usage':{'prompt_tokens':3,'completion_tokens':2}}]
    response=httpx.Response(200,request=httpx.Request('POST','https://example.invalid'),
        content=''.join('data: '+json.dumps(x)+'\n\n' for x in data)+'data: [DONE]\n\n')
    assert extract_stream_text(response)==('AB',[finish])


@pytest.mark.parametrize('event',[
    {'error':{'message':'upstream failure'}},
    {'choices':[{'delta':{'content':'A'},'finish_reason':None}]},
    {'choices':[{'delta':{'content':'A','reasoning_content':'thinking'},'finish_reason':'stop'}]},
])
def test_stream_failure_is_not_success(event):
    response=httpx.Response(200,request=httpx.Request('POST','https://example.invalid'),
        content='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
    with pytest.raises(ValueError):extract_stream_text(response)


def test_context_serialization_not_new_retrieval():
    tables={'entities':[{'id':'1','entity':'A','in_context':True},
                        {'id':'2','entity':'B','in_context':False}]}
    assert plain_table(tables,'entities','Entities')=='-----Entities-----\nid|entity\n1|A\n'
    with pytest.raises(ValueError):plain_table({'entities':[{'id':1}]},'entities','Entities')


@pytest.mark.parametrize('seed,turn',[(42,99),(43,18)])
def test_actual_medical_request_hash_roundtrip(seed,turn):
    root=Path(__file__).resolve().parents[1]/'tmp/bnrr_medical_deepseek_20260906/medical_100r'
    if not root.exists():pytest.skip('Historical fixtures are not checked into git')
    body,audit=reconstruct_request(root,seed,turn)
    assert hashlib.sha256(wire_bytes(body)).hexdigest()==audit['source_request_sha256']
    assert audit['query_and_context_exact'] and audit['api_calls']==0
    assert increase_limit(body,audit['source_request_sha256'],32768)['max_tokens']==32768


def test_same_request_retry_changes_no_bytes():
    body = {'messages': [{'role': 'user', 'content': 'Original query and context'}],
            'enable_thinking': False, 'max_tokens': 16384, 'stream_options': {'include_usage': True}}
    digest = hashlib.sha256(wire_bytes(body)).hexdigest()
    assert wire_bytes(unchanged_request(body, digest)) == wire_bytes(body)
    with pytest.raises(ValueError, match='modified'):
        unchanged_request({**body, 'max_tokens': 32768}, digest)
    for key,value in [('enable_thinking', True), ('max_tokens', 32768), ('stream_options', {})]:
        wrong = {**body, key:value}
        with pytest.raises(ValueError, match='contract'):
            unchanged_request(wrong, hashlib.sha256(wire_bytes(wrong)).hexdigest())


def test_actual_agriculture_same_request_roundtrip():
    root = Path(__file__).resolve().parents[1]/'tmp/bnrr_v7_agriculture_seed42_20260906_1553/agriculture_100r'
    if not root.exists(): pytest.skip('Historical fixtures are not checked into git')
    with pytest.raises(ValueError, match='length-truncated'):
        reconstruct_request(root,42,70)
    body,audit = reconstruct_request(root,42,70,retry_format_error=True)
    assert hashlib.sha256(wire_bytes(body)).hexdigest() == audit['source_request_sha256']
    assert unchanged_request(body,audit['source_request_sha256'])['max_tokens'] == 16384
    assert body['enable_thinking'] is False and audit['api_calls'] == 0
    assert audit['turn'] == 70
    with pytest.raises(ValueError, match='approved failed Agriculture'):
        reconstruct_request(root,42,69,retry_format_error=True)
