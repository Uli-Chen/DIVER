import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from extraction.backends import graphrag as g
from extraction.separated_query import (PROTOCOL, RetrievalOnlyContext,
    local_search_separated, stream_separated_engine)


def engine_fixture(fail=False):
    from graphrag.query.structured_search.local_search.search import LocalSearch
    seen = {'retrieval': [], 'generation': []}
    class Builder:
        def build_context(self, **kwargs):
            seen['retrieval'].append(kwargs)
            return SimpleNamespace(context_chunks='ORIGINAL EVIDENCE', context_records={'entities': []})
    class Model:
        async def achat_stream(self, **kwargs):
            seen['generation'].append(kwargs)
            await asyncio.sleep(0)
            if fail:
                raise RuntimeError('model unavailable')
            yield 'ENTITY: A'
    engine = LocalSearch(model=Model(), context_builder=Builder(),
        system_prompt='ORIGINAL SYSTEM {context_data} {response_type}', response_type='Multiple Paragraphs',
        model_params={'max_tokens': 16384, 'temperature': 0, 'top_p': 1},
        context_builder_params={'top_k_mapped_entities': 10})
    return engine, seen


def test_real_local_search_keeps_format_out_of_retrieval():
    engine, seen = engine_fixture()
    builder, callbacks = engine.context_builder, engine.callbacks
    pure = 'What are the relationships involving NEW COLONY?'
    prompts = [pure + '\n\nOutput ENTITY, Source, Target records.',
               pure + '\n\nA completely different output format instruction.']
    for prompt in prompts:
        response, context = asyncio.run(stream_separated_engine(engine,
            retrieval_query=pure, generation_query=prompt))
        assert response == 'ENTITY: A' and context == {'entities': []}
    assert [x['query'] for x in seen['retrieval']] == [pure, pure]
    assert [x['prompt'] for x in seen['generation']] == prompts
    assert all(x['history'] == [{'role': 'system', 'content': 'ORIGINAL SYSTEM ORIGINAL EVIDENCE Multiple Paragraphs'}] for x in seen['generation'])
    assert all(x['model_parameters'] == {'max_tokens': 16384, 'temperature': 0, 'top_p': 1} for x in seen['generation'])
    assert engine.context_builder is builder and engine.callbacks is callbacks


def test_context_binding_restored_after_error():
    engine, seen = engine_fixture(fail=True)
    builder, callbacks = engine.context_builder, engine.callbacks
    with pytest.raises(RuntimeError, match='model unavailable'):
        asyncio.run(stream_separated_engine(engine, retrieval_query='topic', generation_query='topic plus format'))
    assert engine.context_builder is builder and engine.callbacks is callbacks
    assert len(seen['retrieval']) == len(seen['generation']) == 1


def test_two_instances_do_not_leak_queries():
    a, aa = engine_fixture(); b, bb = engine_fixture()
    async def execute():
        return await asyncio.gather(stream_separated_engine(a, retrieval_query='A?', generation_query='A? format A'),
            stream_separated_engine(b, retrieval_query='B?', generation_query='B? format B'))
    asyncio.run(execute())
    assert aa['retrieval'][0]['query'] == 'A?' and bb['retrieval'][0]['query'] == 'B?'


def test_missing_query_history_and_repeated_retrieval_rejected():
    engine, seen = engine_fixture()
    with pytest.raises(ValueError): RetrievalOnlyContext(engine.context_builder, ' ', 'format')
    builder = RetrievalOnlyContext(engine.context_builder, 'topic', 'full')
    with pytest.raises(ValueError): builder.build_context(query='different')
    with pytest.raises(ValueError): builder.build_context(query='full', conversation_history=object())
    builder.build_context(query='full')
    with pytest.raises(RuntimeError): builder.build_context(query='full')
    assert len(seen['retrieval']) == 1


def test_factory_bridge_preserves_public_api_inputs(monkeypatch, tmp_path):
    # Match the adapter's neutral import directory: NLTK otherwise mistakes a
    # repository-local virtualenv for a package shadowed by the working directory.
    monkeypatch.chdir(tmp_path)
    from graphrag.api import query as api
    engine, seen = engine_fixture()
    captured = {}
    monkeypatch.setattr(api, 'init_loggers', lambda **kwargs: None)
    monkeypatch.setattr(api, 'get_embedding_store', lambda **kwargs: captured.setdefault('store', kwargs))
    monkeypatch.setattr(api, 'load_search_prompt', lambda root, path: 'ORIGINAL FILE PROMPT')
    for name in ['read_indexer_reports', 'read_indexer_entities', 'read_indexer_text_units', 'read_indexer_relationships', 'read_indexer_covariates']:
        monkeypatch.setattr(api, name, lambda *args: args)
    def factory(**kwargs):
        captured['factory'] = kwargs
        return engine
    monkeypatch.setattr(api, 'get_local_search_engine', factory)
    config = SimpleNamespace(root_dir=tmp_path, local_search=SimpleNamespace(prompt='original.txt'),
        vector_store={'default': SimpleNamespace(model_dump=lambda: {'db_uri': '/original/index'})})
    dfs = {k: k for k in ['entities', 'communities', 'community_reports', 'text_units', 'relationships', 'covariates']}
    asyncio.run(local_search_separated(config=config, dataframes=dfs, community_level=2,
        response_type='Multiple Paragraphs', retrieval_query='topic', generation_query='topic and format'))
    params = captured['factory']
    assert params['config'] is config and params['system_prompt'] == 'ORIGINAL FILE PROMPT'
    assert params['entities'] == ('entities', 'communities', 2)
    assert params['reports'] == ('community_reports', 'communities', 2)
    assert params['relationships'] == ('relationships',) and params['covariates'] == {'claims': ('covariates',)}
    assert captured['store']['config_args']['default']['db_uri'] == '/original/index'
    assert seen['retrieval'][0]['query'] == 'topic'


def test_adapter_persists_and_validates_both_queries(monkeypatch, tmp_path):
    seen = []
    def local_search(**kwargs):
        seen.append(kwargs)
        return 'ENTITY: A', {'entities': [{'entity': 'A'}]}
    monkeypatch.setattr(g, '_run_graphrag_local_search', local_search)
    prompt, pure = 'Question?\n\nENTITY Source Target instructions', 'Question?'
    adapter = g.AgeaGraphRagAdapter(graph_root=str(tmp_path), data_dir=str(tmp_path),
        run_dir=tmp_path / 'run', extraction_parser='bnrr')
    first = adapter.query(prompt, 1, retrieval_query=pure)
    context = json.loads(Path(first.retrieved_context_path).read_text())
    assert context['query'] == context['generation_query'] == prompt
    assert context['retrieval_query'] == pure and context['retrieval_protocol'] == PROTOCOL
    assert seen[0]['query'] == prompt and seen[0]['retrieval_query'] == pure
    replay = adapter.query(prompt, 1, retrieval_query=pure, resume_saved_response=True)
    assert replay.batch.to_records() == first.batch.to_records() and len(seen) == 1
    with pytest.raises(ValueError, match='retrieval query/protocol mismatch'):
        adapter.query(prompt, 1, retrieval_query='other', resume_saved_response=True)
    with pytest.raises(ValueError, match='legacy retrieval'):
        adapter.query(prompt, 1, resume_saved_response=True)
    adapter.close()


def test_seed_replay_requires_pure_query_provenance(tmp_path):
    source = tmp_path / 'seed'; source.mkdir()
    response = source / 'response.txt'; response.write_text('ENTITY: A')
    adapter = g.AgeaGraphRagAdapter(graph_root=str(tmp_path), data_dir=str(tmp_path), run_dir=tmp_path / 'run')
    with pytest.raises(ValueError, match='provenance'):
        adapter.query('topic plus format', 1, retrieval_query='topic', response_override_path=response)
    (source / 'retrieval_query.txt').write_text('topic')
    result = adapter.query('topic plus format', 1, retrieval_query='topic', response_override_path=response)
    assert json.loads(Path(result.retrieved_context_path).read_text())['retrieval_query'] == 'topic'
    adapter.close()


def test_real_graphrag_factory_and_context_pass_only_topic_to_embedder(monkeypatch):
    from graphrag.query import factory
    from graphrag.data_model.entity import Entity
    from graphrag.data_model.relationship import Relationship
    from graphrag.vector_stores.base import VectorStoreDocument, VectorStoreSearchResult
    seen = {'embedding': [], 'generation': []}
    class Embedding:
        def embed(self, text):
            seen['embedding'].append(text)
            return [1.0, 0.0]
    class Store:
        def similarity_search_by_text(self, text, text_embedder, k):
            assert text_embedder(text) == [1.0, 0.0]
            return [VectorStoreSearchResult(document=VectorStoreDocument(id='a', text='Alpha', vector=None), score=1)]
    class Chat:
        async def achat_stream(self, **kwargs):
            seen['generation'].append(kwargs)
            yield 'ENTITY: ALPHA'
    monkeypatch.setattr(factory.ModelManager, 'get_or_create_embedding_model', lambda *args, **kwargs: Embedding())
    monkeypatch.setattr(factory.ModelManager, 'get_or_create_chat_model', lambda *args, **kwargs: Chat())
    monkeypatch.setattr(factory, 'get_tokenizer', lambda **kwargs: SimpleNamespace(
        encode=lambda text: list(text.encode()), num_tokens=lambda text: len(text.encode())))
    monkeypatch.setattr(factory, 'get_openai_model_parameters_from_config', lambda settings: {'max_tokens': 16384, 'temperature': 0})
    local = SimpleNamespace(chat_model_id='chat', embedding_model_id='embedding', text_unit_prop=.5,
        community_prop=.25, conversation_history_max_turns=5, top_k_entities=10,
        top_k_relationships=10, max_context_tokens=12000)
    config = SimpleNamespace(local_search=local, get_language_model_config=lambda name: SimpleNamespace(type='offline'))
    engine = factory.get_local_search_engine(config=config, reports=[], text_units=[],
        entities=[Entity(id='a', short_id='1', title='ALPHA', description='Alpha subject', rank=1),
                  Entity(id='b', short_id='2', title='BETA', description='Beta subject', rank=1)],
        relationships=[Relationship(id='r', short_id='1', source='ALPHA', target='BETA',
                                    description='Alpha explicitly supports Beta.', weight=1)],
        covariates={'claims': []}, description_embedding_store=Store(), response_type='Multiple Paragraphs',
        system_prompt='ORIGINAL SYSTEM\n{context_data}\n{response_type}', callbacks=[])
    for command in ['Output ENTITY and relationships.', 'Use a different formatting contract.']:
        response, context = asyncio.run(stream_separated_engine(engine,
            retrieval_query='What relates to ALPHA?', generation_query='What relates to ALPHA?\n\n' + command))
        assert response == 'ENTITY: ALPHA'
        assert context['relationships']['target'].tolist() == ['BETA']
    assert seen['embedding'] == ['What relates to ALPHA?', 'What relates to ALPHA?']
    assert len(seen['generation']) == 2
    assert seen['generation'][0]['history'] == seen['generation'][1]['history']
    assert seen['generation'][0]['prompt'] != seen['generation'][1]['prompt']
