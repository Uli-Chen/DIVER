# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License.
"""DIVER GraphRAG retrieval, response capture, and tolerant record parsing.

The response-body separator helper is retained from the AGEA utility snapshot
(https://github.com/shuashua0608/AGEA, revision
c9c27dd15d55fb2a64fef89d2f9f7cda80038ffe), which carries the notice above.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..models import CandidateBatch
from . import llm_client as _RUNNER

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"
_graphrag_local_search_dependencies: Any = None
_graphrag_default_acompletion: Any = None
_graphrag_default_fnllm_parameter_builder: Any = None



class _GraphRagAsyncRuntime:
    """Keep GraphRAG's async HTTP clients on one event loop per adapter.

    GraphRAG's synchronous CLI helper creates a fresh event loop for every
    query, while LiteLLM caches aiohttp clients by event loop.  A long extraction run
    therefore retains one client/connector pair per turn until garbage
    collection.  This runtime owns one reusable loop and closes each query's
    LiteLLM clients on that same loop before the cache TTL can evict them.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def run(self, coroutine_factory: Callable[[], Any]) -> Any:
        if self._closed:
            raise RuntimeError("GraphRAG async runtime is closed")
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        async def run_and_release_clients() -> Any:
            try:
                return await coroutine_factory()
            finally:
                await _close_litellm_async_clients()

        return self._loop.run_until_complete(run_and_release_clients())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is None:
            return
        try:
            loop.run_until_complete(_close_graphrag_async_clients())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()



async def _close_graphrag_async_clients() -> None:
    """Close and forget GraphRAG/LiteLLM clients before their loop closes."""

    from graphrag.language_model.manager import ModelManager

    await _close_litellm_async_clients()

    # GraphRAG's singleton otherwise carries model wrappers across adapter
    # instances (and potentially across different settings files).
    manager = ModelManager()
    for name in manager.list_chat_models():
        manager.remove_chat(name)
    for name in manager.list_embedding_models():
        manager.remove_embedding(name)



async def _close_litellm_async_clients() -> None:
    """Close LiteLLM clients and remove references before cache eviction."""

    import litellm

    # OpenAI-compatible completions are cached as ``AsyncOpenAI`` instances.
    # Their async shutdown method is named ``close`` (not ``aclose``), so
    # LiteLLM's cleanup helper currently skips them.
    client_cache = litellm.in_memory_llm_clients_cache
    for client in list(client_cache.cache_dict.values()):
        await _close_async_resource(client)

    # LiteLLM's streaming handler uses this lazy module-level AsyncHTTPHandler
    # for OpenAI-compatible providers.  It is not part of
    # ``in_memory_llm_clients_cache`` and is therefore omitted by LiteLLM's
    # own cleanup helper.
    module_aclient = vars(litellm).get("module_level_aclient")
    if module_aclient is not None:
        await _close_async_resource(module_aclient)
        vars(litellm).pop("module_level_aclient", None)

    await litellm.close_litellm_async_clients()
    # LiteLLM keys clients by event-loop identity.  Once this loop is closed,
    # or a query has ended, retaining closed handlers serves no purpose.  More
    # importantly, LiteLLM's TTL eviction does not close clients itself.
    client_cache.flush_cache()

    # The global aiohttp handler is outside the client cache.  Its close method
    # intentionally retains object references, so reset them before the next
    # adapter query asks the handler to create a fresh session.
    base_handler = getattr(litellm, "base_llm_aiohttp_handler", None)
    if base_handler is not None:
        base_handler.client_session = None
        base_handler.transport = None
        base_handler.connector = None



async def _close_async_resource(resource: Any) -> None:
    """Close a cached async resource across ``aclose``/``close`` APIs."""

    import inspect

    for method_name in ("aclose", "close"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            # Cleanup remains best-effort and must not mask the query result or
            # the provider exception that caused shutdown.
            pass
        return



@dataclass
class QueryResult:
    response: str
    batch: CandidateBatch
    stats: dict[str, Any]
    retrieved_context_path: str | None
    response_path: str



class AgeaGraphRagAdapter:
    """Execute local retrieval and parse only the DIVER response contract."""

    def __init__(self, *, graph_root: str, data_dir: str, run_dir: Path,
                 query_method: str = "local", disable_api_thinking: bool = False,
                 graphrag_query_retries: int = 2, enable_graph_filter: bool = False,
                 extraction_parser: str = "bnrr") -> None:
        if extraction_parser != "bnrr" or enable_graph_filter:
            raise ValueError("DIVER requires the bnrr parser without graph filtering")
        if query_method != "local" or graphrag_query_retries < 0:
            raise ValueError("DIVER requires local search and nonnegative retries")
        self.graph_root = str(Path(graph_root).resolve())
        self.data_dir = str(Path(data_dir).resolve())
        self.query_method = query_method
        self.disable_api_thinking = disable_api_thinking
        self.graphrag_query_retries = graphrag_query_retries
        self.extraction_parser = extraction_parser
        self.context_dir = run_dir / "turn_logs" / "retrieved_contexts"
        self.response_dir = run_dir / "turn_logs" / "llm_responses"
        for directory in (self.context_dir, self.response_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._async_runtime = _GraphRagAsyncRuntime()

    def close(self) -> None:
        self._async_runtime.close()

    def __enter__(self) -> "AgeaGraphRagAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def query(self, prompt: str, turn: int, *,
              response_override_path: str | Path | None = None,
              resume_saved_response: bool = False,
              retrieval_query: str | None = None) -> QueryResult:
        from ..bnrr_parser import extraction_finish_reasons, parse_response
        if self.extraction_parser != "bnrr":
            raise ValueError("Only the DIVER parser is included")
        if retrieval_query is not None and (not isinstance(retrieval_query, str) or not retrieval_query.strip()):
            raise ValueError("Explicit retrieval query must not be empty")
        response_source = "live"
        if resume_saved_response:
            if response_override_path is not None:
                raise ValueError("Resume and seed override are mutually exclusive")
            raw_response_path = str(self.response_dir / f"first_llm_response_query_{turn}.txt")
            context_path = str(self.context_dir / f"retrieved_context_query_{turn}.json")
            saved = Path(raw_response_path).read_text()
            marker = "Full GraphRAG Response (including retrieved context):\n"
            if not saved.startswith(f"Query: {prompt}\n") or marker not in saved:
                raise ValueError("Saved response/query mismatch on resume")
            context = json.loads(Path(context_path).read_text())
            if context.get("query") != prompt or context.get("turn") != turn:
                raise ValueError("Saved response/context mismatch on resume")
            if retrieval_query is not None:
                from ..separated_query import PROTOCOL
                if context.get("retrieval_query") != retrieval_query or context.get("retrieval_protocol") != PROTOCOL:
                    raise ValueError("Saved retrieval query/protocol mismatch on resume")
            elif context.get("retrieval_query") is not None:
                raise ValueError("Separated response requires its original retrieval query")
            if context.get("capture_status") == "shared_seed_replay":
                response_source, query_attempt = "shared_seed_replay", 0
            elif context.get("capture_status") == "captured":
                query_attempt = 1
            else:
                raise ValueError("Saved response is not a completed capture")
            response = saved.split(marker, 1)[1]
        elif response_override_path is not None:
            response_source, query_attempt = "shared_seed_replay", 0
            response, context_path, raw_response_path = _replay_saved_response(
                query=prompt, turn=turn, source_path=Path(response_override_path),
                context_dir=self.context_dir, response_dir=self.response_dir,
                allow_empty_response=True, retrieval_query=retrieval_query)
        else:
            max_attempts = self.graphrag_query_retries + 1
            for query_attempt in range(1, max_attempts + 1):
                try:
                    response, context_path, raw_response_path = _run_live_local_query(
                        query=prompt, turn=turn, graph_root=self.graph_root,
                        data_dir=self.data_dir, context_dir=self.context_dir,
                        response_dir=self.response_dir, query_method=self.query_method,
                        disable_api_thinking=self.disable_api_thinking,
                        runtime=self._async_runtime, allow_empty_response=True,
                        retrieval_query=retrieval_query)
                    break
                except Exception as exc:
                    if query_attempt >= max_attempts:
                        raise
                    delay = min(2 ** (query_attempt - 1), 8)
                    print(f"[retry] GraphRAG extraction turn={turn} "
                          f"attempt={query_attempt}/{max_attempts} failed: {exc}; "
                          f"retrying in {delay}s", file=sys.stderr)
                    time.sleep(delay)

        context = json.loads(Path(context_path).read_text())
        request_dir = Path(response_override_path).parent if response_override_path else self.response_dir.parent.parent
        finish_reasons = context.get("extraction_finish_reasons")
        if finish_reasons is None:
            finish_reasons = extraction_finish_reasons(request_dir / "requests.jsonl", 1 if response_override_path else turn)
            context["extraction_finish_reasons"] = finish_reasons
            Path(context_path).write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        nodes, edges, parse_stats = parse_response(response,
            extract_body=extract_actual_llm_response, finish_reasons=finish_reasons)
        batch = CandidateBatch.from_records(nodes, edges)
        return QueryResult(response=response, batch=batch, stats={
            "source": response_source, "graphrag_query_attempts": query_attempt,
            "response_characters": len(response), "raw_nodes": len(nodes),
            "raw_edges": len(edges), "kept_nodes_explicit": len(nodes),
            "kept_edges": len(edges), "candidate_nodes_with_endpoints": len(batch.nodes),
            "candidate_edges": len(batch.edges), "graph_filter_status": "disabled",
            **parse_stats,
        }, retrieved_context_path=context_path, response_path=str(raw_response_path))



def extract_actual_llm_response(full_response: str) -> str:

    if not full_response:
        return ""
    if not isinstance(full_response, str):
        return ""

    # Look for the separator line that indicates the start of the actual LLM response
    separator_patterns = [
        r'={80,}\s*\n\s*LLM Response:\s*\n',
        r'LLM Response:\s*\n',
        r'Response:\s*\n',
        r'={50,}\s*\n'
    ]

    for pattern in separator_patterns:
        match = re.search(pattern, full_response, re.IGNORECASE)
        if match:
            return full_response[match.end():].strip()

    # If no separator found, return the full response
    return full_response



def _replay_saved_response(
    *,
    query: str,
    turn: int,
    source_path: Path,
    context_dir: Path,
    response_dir: Path,
    retrieval_query: str | None = None,
    allow_empty_response: bool = False,
) -> tuple[str, str, str]:
    """Replay one audited seed response so policy ablations share initial state."""

    saved = source_path.read_text(encoding="utf-8")
    marker = "Full GraphRAG Response (including retrieved context):\n"
    response = saved.split(marker, 1)[1] if marker in saved else saved
    if not response.strip() and not allow_empty_response:
        raise ValueError(f"Shared seed response is empty: {source_path}")
    retrieval_metadata = {}
    if retrieval_query is not None:
        from ..separated_query import PROTOCOL
        sidecar = source_path.parent / "retrieval_query.txt"
        if not sidecar.is_file() or sidecar.read_text() != retrieval_query:
            raise ValueError("Shared seed lacks matching pure-retrieval provenance")
        retrieval_metadata = {"retrieval_query": retrieval_query,
            "generation_query": query, "retrieval_protocol": PROTOCOL}

    context_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    response_path = response_dir / f"first_llm_response_query_{turn}.txt"
    context_path = context_dir / f"retrieved_context_query_{turn}.json"
    response_path.write_text(
        _format_response_artifact(query, response), encoding="utf-8"
    )
    context_path.write_text(
        json.dumps(
            {
                "capture_status": "shared_seed_replay",
                "turn": turn,
                "query": query,
                "source_response_path": str(source_path.resolve()),
                **retrieval_metadata,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return response, str(context_path), str(response_path)



def _run_live_local_query(
    *,
    query: str,
    turn: int,
    graph_root: str,
    data_dir: str,
    context_dir: Path,
    response_dir: Path,
    query_method: str,
    disable_api_thinking: bool,
    runtime: _GraphRagAsyncRuntime,
    retrieval_query: str | None = None,
    allow_empty_response: bool = False,
) -> tuple[str, str, str]:
    """Run the same GraphRAG local-search entry point and retain its context.

    GraphRAG's Python API returns ``context_data`` alongside the response.
    The extraction adapter serializes this local-search result, which also
    lets the adapter reuse and deterministically close one async runtime instead
    of accepting the CLI helper's fresh event loop per query.
    """

    if query_method != "local":
        raise ValueError(
            "The extraction adapter currently captures retrieved context only "
            f"for query_method='local', got {query_method!r}."
        )

    context_path = context_dir / f"retrieved_context_query_{turn}.json"
    response_path = response_dir / f"first_llm_response_query_{turn}.txt"
    context_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    response, context_data = _run_graphrag_local_search(
        config_filepath=None,
        data_dir=Path(data_dir),
        root_dir=Path(graph_root),
        community_level=COMMUNITY_LEVEL,
        response_type=RESPONSE_TYPE,
        streaming=False,
        query=query,
        verbose=False,
        disable_api_thinking=disable_api_thinking,
        runtime=runtime,
        **({"retrieval_query": retrieval_query} if retrieval_query is not None else {}),
    )

    response_text = str(response)
    response_path.write_text(
        _format_response_artifact(query, response_text), encoding="utf-8"
    )
    if not response_text.strip() and not allow_empty_response:
        raise RuntimeError(
            f"GraphRAG query returned an empty response at turn {turn}. "
            f"Diagnostics: {response_path}"
        )

    context_payload = {
        "capture_status": "captured",
        "query_method": query_method,
        "turn": turn,
        "query": query,
        "tables": _json_safe_context(context_data),
    }
    if retrieval_query is not None:
        from ..separated_query import PROTOCOL
        context_payload.update(retrieval_query=retrieval_query,
            generation_query=query, retrieval_protocol=PROTOCOL)
    context_path.write_text(
        json.dumps(context_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return response_text, str(context_path), str(response_path)



def _run_graphrag_local_search(
    *,
    config_filepath: Path | None,
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
    disable_api_thinking: bool,
    runtime: _GraphRagAsyncRuntime,
    retrieval_query: str | None = None,
) -> tuple[Any, Any]:
    """Run GraphRAG's local-search API on the adapter-owned event loop."""

    dependencies = (
        _graphrag_local_search_dependencies
        or _load_graphrag_local_search_dependencies()
    )
    api, load_config, resolve_output_files = dependencies
    _configure_graphrag_thinking(disable_api_thinking)
    root = root_dir.resolve()
    cli_overrides = {}
    if data_dir:
        cli_overrides["output.base_dir"] = str(data_dir)
    config = load_config(root, config_filepath, cli_overrides)
    dataframe_dict = resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        optional_list=["covariates"],
    )

    if retrieval_query is not None:
        if dataframe_dict["multi-index"]:
            raise ValueError("Separate retrieval currently supports a single local index only")
        from ..separated_query import local_search_separated
        return runtime.run(lambda: local_search_separated(config=config,
            dataframes=dataframe_dict, community_level=community_level,
            response_type=response_type, generation_query=query,
            retrieval_query=retrieval_query, verbose=verbose))

    if dataframe_dict["multi-index"]:
        covariates = (
            dataframe_dict["covariates"]
            if len(dataframe_dict["covariates"]) == dataframe_dict["num_indexes"]
            else None
        )
        return runtime.run(
            lambda: api.multi_index_local_search(
                config=config,
                entities_list=dataframe_dict["entities"],
                communities_list=dataframe_dict["communities"],
                community_reports_list=dataframe_dict["community_reports"],
                text_units_list=dataframe_dict["text_units"],
                relationships_list=dataframe_dict["relationships"],
                covariates_list=covariates,
                index_names=dataframe_dict["index_names"],
                community_level=community_level,
                response_type=response_type,
                streaming=streaming,
                query=query,
                verbose=verbose,
            )
        )

    return runtime.run(
        lambda: api.local_search(
            config=config,
            entities=dataframe_dict["entities"],
            communities=dataframe_dict["communities"],
            community_reports=dataframe_dict["community_reports"],
            text_units=dataframe_dict["text_units"],
            relationships=dataframe_dict["relationships"],
            covariates=dataframe_dict["covariates"],
            community_level=community_level,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )



async def _acompletion_without_thinking(completion: Any, **kwargs: Any) -> Any:
    """Apply the shared provider's non-reasoning option to LiteLLM calls."""

    request = dict(kwargs)
    options = _RUNNER.agent_completion_options(str(request.get("model", "")))
    if options:
        request["extra_body"] = {
            **(request.get("extra_body") or {}), **options["extra_body"]}
    return await completion(**request)



def _configure_graphrag_thinking(disable_api_thinking: bool) -> None:
    """Scope GraphRAG provider thinking behavior to this experiment process."""

    global _graphrag_default_acompletion
    global _graphrag_default_fnllm_parameter_builder
    from graphrag.language_model.providers.litellm import chat_model
    from graphrag.language_model.providers.fnllm import utils as fnllm_utils

    if _graphrag_default_acompletion is None:
        _graphrag_default_acompletion = chat_model.acompletion
    if _graphrag_default_fnllm_parameter_builder is None:
        _graphrag_default_fnllm_parameter_builder = (
            fnllm_utils.get_openai_model_parameters_from_dict
        )
    if not disable_api_thinking:
        chat_model.acompletion = _graphrag_default_acompletion
        fnllm_utils.get_openai_model_parameters_from_dict = (
            _graphrag_default_fnllm_parameter_builder
        )
        return

    default_completion = _graphrag_default_acompletion

    async def completion_without_thinking(**kwargs: Any) -> Any:
        return await _acompletion_without_thinking(default_completion, **kwargs)

    chat_model.acompletion = completion_without_thinking

    default_parameter_builder = _graphrag_default_fnllm_parameter_builder

    def parameters_without_thinking(config: dict[str, Any]) -> dict[str, Any]:
        params = default_parameter_builder(config)
        options = _RUNNER.agent_completion_options(str(config.get("model", "")))
        if options:
            params["extra_body"] = {
                **(params.get("extra_body") or {}), **options["extra_body"]}
        return params

    fnllm_utils.get_openai_model_parameters_from_dict = parameters_without_thinking



def _load_graphrag_local_search_dependencies() -> Any:
    """Import GraphRAG while avoiding NLTK's current-directory guard.

    NLTK blocks importing ``regex`` when the project root is on Python's
    current-directory search path.  AGEA works around this for subprocesses by
    using the interpreter's ``bin`` directory as ``cwd``.  The adapter mirrors
    that workaround only for the lazy import and restores the original working
    directory immediately afterward.
    """

    global _graphrag_local_search_dependencies
    original_cwd = Path.cwd()
    safe_cwd = Path(sys.executable).resolve().parent
    try:
        os.chdir(safe_cwd)
        import graphrag.api as api
        from graphrag.cli.query import _resolve_output_files
        from graphrag.config.load_config import load_config
    finally:
        os.chdir(original_cwd)
    _graphrag_local_search_dependencies = (api, load_config, _resolve_output_files)
    return _graphrag_local_search_dependencies



def _format_response_artifact(query: str, response: str) -> str:
    """Keep response artifacts compatible with AGEA's existing file layout."""

    return (
        f"Query: {query}\n"
        f"{'=' * 80}\n"
        "Full GraphRAG Response (including retrieved context):\n"
        f"{response}"
    )



def _json_safe_context(context_data: Any) -> Any:
    """Convert GraphRAG context tables to strict, portable JSON values.

    GraphRAG returns a mapping of pandas DataFrames.  Converting each frame via
    ``to_json`` handles NumPy scalars, timestamps and missing values without
    leaking Python-specific representations into the experiment artifact.
    """

    if hasattr(context_data, "to_json"):
        return json.loads(
            context_data.to_json(orient="records", force_ascii=False)
        )
    if isinstance(context_data, dict):
        return {
            str(key): _json_safe_context(value)
            for key, value in context_data.items()
        }
    if isinstance(context_data, (list, tuple)):
        return [_json_safe_context(value) for value in context_data]
    if context_data is None or isinstance(context_data, (str, int, float, bool)):
        return context_data
    return str(context_data)
