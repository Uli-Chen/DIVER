"""Instance-local separation of retrieval content from answer instructions.

Uses the same GraphRAG local-search factory, adapters, prompt and streaming
generation path as its public API. Never patches a global GraphRAG class.
"""
from __future__ import annotations

PROTOCOL = "bnrr-pure-retrieval-v1"


class RetrievalOnlyContext:
    def __init__(self, delegate, retrieval_query, generation_query):
        if not isinstance(retrieval_query, str) or not retrieval_query.strip():
            raise ValueError("A nonempty, explicit retrieval query is required")
        self.delegate = delegate
        self.retrieval_query = retrieval_query
        self.generation_query = generation_query
        self.calls = 0

    def build_context(self, query, conversation_history=None, **kwargs):
        if query != self.generation_query:
            raise ValueError("Unexpected generation query in separated local search")
        if conversation_history is not None:
            raise ValueError("Separate retrieval must not inherit generation history")
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("A local-search attempt must build context exactly once")
        return self.delegate.build_context(query=self.retrieval_query,
            conversation_history=None, **kwargs)


async def stream_separated_engine(engine, *, retrieval_query, generation_query):
    """Preserve model/system prompt/parameters; change only context-builder input."""
    from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
    original_builder, original_callbacks = engine.context_builder, engine.callbacks
    builder = RetrievalOnlyContext(original_builder, retrieval_query, generation_query)
    context = None
    def capture(value):
        nonlocal context
        context = value
    callback = NoopQueryCallbacks()
    callback.on_context = capture
    engine.context_builder = builder
    engine.callbacks = [*original_callbacks, callback]
    try:
        chunks = []
        async for chunk in engine.stream_search(query=generation_query):
            chunks.append(chunk)
        if builder.calls != 1 or context is None:
            raise RuntimeError("Local search did not expose its single retrieved context")
        return "".join(chunks), context
    finally:
        engine.context_builder, engine.callbacks = original_builder, original_callbacks


async def local_search_separated(*, config, dataframes, community_level,
                                 response_type, generation_query, retrieval_query,
                                 verbose=False):
    # Mirror api.local_search_streaming's setup using its own supported helpers.
    from graphrag.api import query as api
    api.init_loggers(config=config, verbose=verbose, filename="query.log")
    store = api.get_embedding_store(
        config_args={k: v.model_dump() for k, v in config.vector_store.items()},
        embedding_name=api.entity_description_embedding)
    covariates = dataframes["covariates"]
    engine = api.get_local_search_engine(
        config=config,
        reports=api.read_indexer_reports(dataframes["community_reports"], dataframes["communities"], community_level),
        text_units=api.read_indexer_text_units(dataframes["text_units"]),
        entities=api.read_indexer_entities(dataframes["entities"], dataframes["communities"], community_level),
        relationships=api.read_indexer_relationships(dataframes["relationships"]),
        covariates={"claims": api.read_indexer_covariates(covariates) if covariates is not None else []},
        description_embedding_store=store,
        response_type=response_type,
        system_prompt=api.load_search_prompt(config.root_dir, config.local_search.prompt),
        callbacks=[])
    return await stream_separated_engine(engine, retrieval_query=retrieval_query,
        generation_query=generation_query)
