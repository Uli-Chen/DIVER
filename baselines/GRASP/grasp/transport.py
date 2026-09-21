"""Local victim adapter. Retrieved context never reaches the attack controller."""
import json, os, sys
from pathlib import Path
from .io import ROOT, PROJECT_ROOT, write


def credentials(config):
    from dotenv import dotenv_values
    project=PROJECT_ROOT
    values={**dotenv_values(project/'.env'),**os.environ}
    v=config['victim'];medical=config['dataset']=='medical'
    chat_key=values.get('tmp_medical_api_key' if medical else 'GRAPHRAG_API_KEY')
    chat_base=values.get('tmp_medical_api_base' if medical else 'GRAPHRAG_API_BASE','')
    chat_model=values.get('tmp_medical_chat_model' if medical else 'GRAPHRAG_CHAT_MODEL')
    if chat_base.rstrip('/')!=v['chat_provider'] or chat_model!=v['chat_model'] or not chat_key:
        raise RuntimeError('Chat credential profile does not match audited configuration')
    if values.get('GRAPHRAG_EMBEDDING_API_BASE','').rstrip('/')!=v['embedding_provider'] or values.get('GRAPHRAG_EMBEDDING_MODEL')!=v['embedding_model']:
        raise RuntimeError('Embedding profile mismatch')
    embedding_key=values.get('GRAPHRAG_EMBEDDING_API_KEY')
    if not embedding_key:raise RuntimeError('Missing embedding credential')
    os.environ.update(GRAPHRAG_API_KEY=chat_key,AGEA_API_KEY=chat_key,GRAPHRAG_API_BASE=v['chat_provider'],AGEA_API_BASE=v['chat_provider'],
        GRAPHRAG_CHAT_MODEL=v['chat_model'],AGEA_CHAT_MODEL=v['chat_model'],
        GRAPHRAG_EMBEDDING_API_KEY=embedding_key,GRAPHRAG_EMBEDDING_API_BASE=v['embedding_provider'],GRAPHRAG_EMBEDDING_MODEL=v['embedding_model'],
        AGEA_THINKING_CONTROL_STYLE=v['thinking_control_style'],AGEA_LLM_PROVIDER='openai_compatible')


class NativeVictim:
    def __init__(self,config,out):
        self.c,self.out=config,Path(out)
        credentials(config)
        sys.path.insert(0,str(Path(config['source_code'])/'src'))
        from extraction.backends.graphrag import _GraphRagAsyncRuntime, _run_graphrag_local_search
        from extraction.request_audit import RequestAudit
        from urllib.parse import urlsplit
        import httpx
        self.original_send=(httpx.Client.send,httpx.AsyncClient.send)
        expected=config['victim']
        class CheckedAudit(RequestAudit):
            journal_starts=True
            def start(self,request):
                if request.url.path.endswith(('/chat/completions','/embeddings')):
                    body=json.loads(request.content);chat=request.url.path.endswith('/chat/completions')
                    host=urlsplit(expected['chat_provider' if chat else 'embedding_provider']).hostname
                    model=expected['chat_model' if chat else 'embedding_model']
                    if request.url.host!=host or body.get('model')!=model:raise RuntimeError('Unapproved model endpoint')
                    if chat and body.get('max_tokens')!=expected['output_tokens']:raise RuntimeError('Output budget drift')
                return super().start(request)
        self.audit=CheckedAudit(self.out/'requests.jsonl')
        if self.audit.path.exists():
            from extraction.resume import restore_accounting
            restore_accounting(self.audit)
        self.audit.install()
        self.runtime=_GraphRagAsyncRuntime();self.query_fn=_run_graphrag_local_search

    def query(self,request):
        import httpx,openai
        self.audit.turn=request['turn']
        previous=self.audit.count
        v=self.c['victim']
        try:
            answer,_context=self.query_fn(config_filepath=None,data_dir=self.out/'graph_root/output',root_dir=self.out/'graph_root',
                community_level=v['community_level'],response_type=v['response_type'],streaming=False,
                query=request['generation_query'],retrieval_query=None,verbose=False,disable_api_thinking=True,runtime=self.runtime)
            # The GraphRAG engine may read the index; the attacker receives answer text only.
            answer=str(answer)
            error=None
        except (httpx.HTTPError,openai.APIError,TimeoutError) as exc:
            answer='';error=type(exc).__name__
        ledger=[json.loads(x) for x in self.audit.path.read_text().splitlines()][previous:] if self.audit.path.exists() else []
        chat=[item for item in ledger if item['kind']=='chat']
        reasons=chat[-1].get('finish_reasons',[]) if chat else []
        # A retried embedding/chat failure remains billed, but does not invalidate
        # the eventual successful reply. Only the terminal response controls parsing.
        if chat and (chat[-1].get('error') or (chat[-1].get('status') or 0)>=400):
            error=error or 'ProviderRequestFailure'
        if not answer.strip():error=error or 'EmptyResponse'
        if not ledger:raise RuntimeError('Native call produced no metered HTTP receipt')
        return {'response':answer,'finish_reasons':reasons,'error':error,'request_count':len(ledger),
                'cumulative_requests':self.audit.count,'usage_unknown':sum(not x.get('usage_complete') for x in ledger)}

    def close(self):
        import httpx
        try:self.runtime.close()
        finally:httpx.Client.send,httpx.AsyncClient.send=self.original_send
