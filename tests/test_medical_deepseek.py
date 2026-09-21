import asyncio
import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest

from extraction.request_audit import RequestAudit
from extraction.backends import graphrag as g

ROOT=Path(__file__).resolve().parents[1]


def test_native_thinking_controls(monkeypatch):
    monkeypatch.setenv('AGEA_THINKING_CONTROL_STYLE','thinking_disabled')
    assert g._RUNNER.agent_completion_options('deepseek-v4-flash')=={'extra_body':{'thinking':{'type':'disabled'}}}
    async def capture(**kwargs): return kwargs
    result=asyncio.run(g._acompletion_without_thinking(capture,model='deepseek-v4-flash',max_tokens=16384))
    assert result['extra_body']=={'thinking':{'type':'disabled'}}
    assert result['max_tokens']==16384 and 'reasoning_effort' not in result


@pytest.mark.parametrize('body,valid',[
    ({'thinking':{'type':'disabled'}},True),
    ({'enable_thinking':False},False),
    ({'thinking':{'type':'enabled'}},False),
    ({},False)])
def test_native_wire_audit(tmp_path,body,valid):
    request=httpx.Request('POST','https://api.deepseek.com/chat/completions',json={'model':'deepseek-v4-flash',**body})
    audit=RequestAudit(tmp_path/'audit.jsonl')
    if valid:
        row=audit.start(request)
        assert row['thinking_disabled'] and row['thinking_control']=='thinking.type=disabled'
    else:
        with pytest.raises(RuntimeError): audit.start(request)


def test_medical_environment_is_process_only(tmp_path,monkeypatch):
    from extraction import medical_provider as module
    monkeypatch.setattr(module,'PROJECT',tmp_path)
    env='\n'.join(['AGEA_API_KEY=original','GRAPHRAG_API_KEY=original',
        'AGEA_API_BASE=https://original.test','GRAPHRAG_API_BASE=https://original.test',
        'tmp_medical_api_key=private-test-key','tmp_medical_api_base=https://api.deepseek.com',
        'tmp_medical_chat_model=deepseek-v4-flash','GRAPHRAG_EMBEDDING_API_KEY=embed-secret',
        'GRAPHRAG_EMBEDDING_API_BASE=https://api.siliconflow.cn/v1','GRAPHRAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B'])
    (tmp_path/'.env').write_text(env)
    monkeypatch.setattr(os,'environ',os.environ.copy())
    module.environment()
    assert os.environ['AGEA_API_BASE']=='https://api.deepseek.com'
    assert os.environ['GRAPHRAG_API_KEY']=='private-test-key'
    assert os.environ['AGEA_THINKING_CONTROL_STYLE']=='thinking_disabled'
    assert os.environ['GRAPHRAG_EMBEDDING_API_KEY']=='embed-secret'
    assert (tmp_path/'.env').read_text()==env
