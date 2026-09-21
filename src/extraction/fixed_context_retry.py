"""Reconstruct a saved extraction request, accepting only its exact wire hash.

Offline reconstruction uses saved context tables and the installed GraphRAG
serialization convention. It never selects new evidence or contacts a service.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def wire_bytes(body):
    # Same normalization as RequestAudit.prepare_request for streamed requests.
    return json.dumps(body, ensure_ascii=False).encode()


def plain_table(tables, key, title):
    records = [r for r in tables.get(key, []) if r.get('in_context', True)]
    if not records:
        return ''
    columns = [k for k in records[0] if k != 'in_context']
    if any(any(not isinstance(row[c], str) for c in columns) for row in records):
        raise ValueError('Unsupported table value; refuse approximate context replay')
    return f'-----{title}-----\n' + '|'.join(columns) + '\n' + ''.join(
        '|'.join(row[c] for c in columns) + '\n' for row in records)


def reconstruct_request(root, seed, turn, *, retry_format_error=False):
    import httpx
    import pandas as pd
    import yaml
    from openai import OpenAI
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.providers.fnllm.utils import get_openai_model_parameters_from_dict
    from .request_audit import RequestAudit
    root = Path(root)
    run = root / f'runs/FULL_seed{seed}'
    requests = [json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
    extracts = [r for r in requests if r['turn'] == turn and r['stage'] == 'extraction']
    if not extracts or extracts[-1]['status'] != 200:
        raise ValueError('No successful HTTP extraction to reconstruct')
    last = extracts[-1]
    if retry_format_error:
        manifest = json.loads((root/'manifest.json').read_text())
        status = json.loads((run/'status.json').read_text())
        failure = json.loads((run/'parse_failure.json').read_text())
        if (manifest['dataset'] != 'agriculture' or seed != 42 or
            manifest.get('retrieval_by_method', {}).get('FULL') != 'bnrr-pure-retrieval-v1' or
            status.get('status') != 'failed' or status.get('round') != turn - 1 or
            status.get('failed_round') != turn or failure.get('turn') != turn or
            failure.get('stats', {}).get('parse_status') != 'format_error' or
            last.get('finish_reasons') not in (['stop'], ['length']) or
            last['max_tokens'] != 16384 or last['model'] != 'deepseek-v4-flash' or
            last['host'] != 'dashscope.aliyuncs.com'):
            raise ValueError('Only the approved failed Agriculture seed42 format response may be retried')
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        thinking = {'enable_thinking': False}
    else:
        if last.get('finish_reasons') != ['length']:
            raise ValueError('Only a positively identified length-truncated extraction can be retried')
        if last['max_tokens'] != 16384 or last['model'] != 'deepseek-v4-flash' or last['host'] != 'api.deepseek.com':
            raise ValueError('Unexpected source provider/model/output budget')
        base_url = 'https://api.deepseek.com'
        thinking = {'thinking': {'type': 'disabled'}}
    context = json.loads((run/f'turn_logs/retrieved_contexts/retrieved_context_query_{turn}.json').read_text())
    if context['capture_status'] != 'captured' or context['turn'] != turn:
        raise ValueError('No complete saved context')
    tables = context['tables']
    if set(tables) - {'reports', 'entities', 'relationships', 'claims', 'sources'}:
        raise ValueError('Unsupported context table')
    local = plain_table(tables,'entities','Entities') + '\n\n' + '\n\n'.join([
        plain_table(tables,'relationships','Relationships'), plain_table(tables,'claims','claims')])
    tail = '\n\n'.join(part for part in (local, plain_table(tables,'sources','Sources')) if part.strip())
    reports = [{k:v for k,v in row.items() if k != 'in_context'}
               for row in tables.get('reports',[]) if row.get('in_context',True)]
    # GraphRAG serializes an empty report-batch list as "[]" in one path.
    prefixes = ['-----Reports-----\n' + pd.DataFrame(reports).to_csv(index=False,sep='|') + '\n\n'] if reports else ['', '[]\n\n']
    cfg = yaml.safe_load((root/'graph_root/settings.yaml').read_text())['models']['default_chat_model']
    cfg.update(api_key='offline-placeholder', api_base=base_url, model=last['model'])
    params = get_openai_model_parameters_from_dict(LanguageModelConfig(**cfg).model_dump())
    params['extra_body'] = thinking
    template = (root/'graph_root/prompts/local_search_system_prompt.txt').read_text()
    matches = []
    for prefix in prefixes:
        messages = [{'role':'system','content':template.format(context_data=prefix+tail,response_type='Multiple Paragraphs')},
                    {'content':context['query'],'role':'user'}]
        def capture(request):
            request = RequestAudit(Path('/unused')).prepare_request(request)
            if hashlib.sha256(request.content).hexdigest() == last['request_body_sha256']:
                matches.append(json.loads(request.content))
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content='data: [DONE]\n\n')
        with OpenAI(api_key='offline-placeholder',base_url=base_url,
                    http_client=httpx.Client(transport=httpx.MockTransport(capture))) as client:
            list(client.chat.completions.create(model=last['model'],messages=messages,stream=True,**params))
    if len(matches) != 1:
        raise ValueError('Could not reconstruct exactly one hash-identical original request; no online fallback')
    return matches[0], {'turn':turn,'source_request_sha256':last['request_body_sha256'],
        'query_and_context_exact':True,'api_calls':0,'original_request_index':last['request_index']}


def unchanged_request(body, expected_hash):
    """Validate an authorized retry without changing a single wire byte."""
    if hashlib.sha256(wire_bytes(body)).hexdigest() != expected_hash:
        raise ValueError('Fixed request was modified')
    if (body.get('enable_thinking') is not False or body.get('max_tokens') != 16384 or
            body.get('stream_options') != {'include_usage': True}):
        raise ValueError('Unchanged Agriculture retry contract mismatch')
    return body


def increase_limit(body, expected_hash, new_limit):
    if hashlib.sha256(wire_bytes(body)).hexdigest() != expected_hash:
        raise ValueError('Fixed request was modified')
    if new_limit != 32768 or body['max_tokens'] != 16384:
        raise ValueError('Only the authorized 16384 -> 32768 transition is supported')
    if body.get('thinking') != {'type':'disabled'} or body.get('stream_options') != {'include_usage':True}:
        raise ValueError('Thinking/usage contract mismatch')
    return {**body, 'max_tokens': new_limit}


def extract_stream_text(response):
    from .request_audit import response_events
    response.raise_for_status()
    events, _ = response_events(response)
    if any(event.get('error') for event in events):
        raise ValueError('Provider returned an in-stream error')
    choices = [c for event in events for c in event.get('choices',[])]
    if any((c.get('delta') or {}).get('reasoning_content') for c in choices):
        raise ValueError('Thinking was unexpectedly enabled')
    reasons = {c.get('finish_reason') for c in choices} - {None}
    text = ''.join((c.get('delta') or {}).get('content') or '' for c in choices)
    if reasons not in ({'stop'}, {'length'}) or not text.strip():
        raise ValueError('Incomplete or empty streamed extraction')
    # Preserve length-truncated responses too; the unchanged parser decides validity.
    return text, sorted(reasons)
