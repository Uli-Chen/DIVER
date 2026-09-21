"""Narrow adapter around the existing AGEA GraphRAG runner and parser."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..closure_evidence import (
    EvidenceCorpus,
    PairEvidencePacket,
    count_tokens,
    parse_grounded_closure_response,
)
from ..models import CandidateBatch, normalize_label


PROJECT_DIR = Path(__file__).resolve().parents[3]
AGEA_DIR = PROJECT_DIR / "baselines" / "AGEA"
if str(AGEA_DIR) not in sys.path:
    sys.path.insert(0, str(AGEA_DIR))

from agea_prompts import UNIVERSAL_EXTRACTION_COMMAND  # noqa: E402
from utils import extract_actual_llm_response  # noqa: E402
_RUNNER_PATH = AGEA_DIR / "graphrag" / "run_agea.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("_agea_graphrag_runner", _RUNNER_PATH)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise ImportError(f"Could not load AGEA runner from {_RUNNER_PATH}")
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)
filter_extraction_with_graph_filter_agent = _RUNNER.filter_extraction_with_graph_filter_agent
llm_generate_agentic_query = _RUNNER.llm_generate_agentic_query
parse_llm_response_for_graph_items = _RUNNER.parse_llm_response_for_graph_items
extraction_response_diagnostics = _RUNNER.extraction_response_diagnostics

# Loaded lazily from a neutral working directory; see
# ``_load_graphrag_local_search_dependencies`` for the NLTK import-safety
# constraint.
_graphrag_local_search_dependencies: Any = None
_graphrag_default_acompletion: Any = None
_graphrag_default_fnllm_parameter_builder: Any = None


class _GraphRagAsyncRuntime:
    """Keep GraphRAG's async HTTP clients on one event loop per adapter.

    GraphRAG's synchronous CLI helper creates a fresh event loop for every
    query, while LiteLLM caches aiohttp clients by event loop.  A long FEWA run
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


@dataclass
class _GraphMemoryView:
    """Minimal AGEA-compatible view over the pipeline's NetworkX graph."""

    G: Any


class AgeaGraphRagAdapter:
    """Execute GraphRAG and reuse AGEA's established response parser."""

    def __init__(
        self,
        *,
        graph_root: str,
        data_dir: str,
        run_dir: Path,
        query_method: str = "local",
        disable_api_thinking: bool = False,
        graphrag_query_retries: int = 2,
        enable_graph_filter: bool = False,
        graph_filter_model: str = "gpt-4o-mini",
        extraction_parser: str = "legacy",
    ) -> None:
        # GraphRAG resolves a relative output override against ``root_dir``.
        # Normalize both paths here so repository-relative experiment configs do
        # not accidentally become ``<graph_root>/artifacts/graphrag/...``.
        self.graph_root = str(Path(graph_root).resolve())
        self.data_dir = str(Path(data_dir).resolve())
        self.query_method = query_method
        self.disable_api_thinking = disable_api_thinking
        self.graphrag_query_retries = graphrag_query_retries
        self.enable_graph_filter = enable_graph_filter
        self.graph_filter_model = graph_filter_model
        if extraction_parser not in {"legacy", "bnrr"}:
            raise ValueError("Unknown extraction parser")
        self.extraction_parser = extraction_parser
        self.context_dir = run_dir / "turn_logs" / "retrieved_contexts"
        self.response_dir = run_dir / "turn_logs" / "llm_responses"
        self.filter_dir = run_dir / "turn_logs" / "graph_filter"
        for directory in (self.context_dir, self.response_dir, self.filter_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._async_runtime = _GraphRagAsyncRuntime()

    def close(self) -> None:
        """Release GraphRAG/LiteLLM network resources owned by this adapter."""

        self._async_runtime.close()

    def __enter__(self) -> "AgeaGraphRagAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def generate_agentic_query(
        self,
        *,
        mode: str,
        novelty_score: float,
        recent_history: list[dict[str, Any]],
        graph: Any,
        dataset_name: str,
        anchor: str | None,
        anchor_round: int,
        query_generator_model: str,
    ) -> str:
        """Reuse AGEA's dynamic query generator while retaining our controller."""

        if mode == "exploit" and anchor:
            return self._generate_fixed_anchor_query(
                graph=graph,
                dataset_name=dataset_name,
                anchor=anchor,
                anchor_round=anchor_round,
                recent_history=recent_history,
                query_generator_model=query_generator_model,
            )

        seed_candidates = [(anchor, anchor_round)] if anchor else []
        return llm_generate_agentic_query(
            mode=mode,
            novelty_score=novelty_score,
            recent_history=recent_history,
            graph_memory=_GraphMemoryView(graph),
            dataset_name=dataset_name,
            seed_candidates=seed_candidates,
            query_generator_model=query_generator_model,
            # The extraction pipeline owns retry and fallback accounting. Do
            # not let AGEA silently turn provider errors into default queries.
            raise_on_failure=True,
        )

    def _generate_fixed_anchor_query(
        self,
        *,
        graph: Any,
        dataset_name: str,
        anchor: str,
        anchor_round: int,
        recent_history: list[dict[str, Any]],
        query_generator_model: str,
    ) -> str:
        """Generate an exploit query without AGEA's hidden second target draw.

        AGEA normally samples from ``seed_candidates`` inside its query
        generator, including a 10% branch that can replace a singleton target.
        Arm selection now lives entirely in the controller, so this path builds
        context for exactly that anchor and lets the LLM phrase only the query.
        """

        neighbors = list(graph.neighbors(anchor)) if anchor in graph else []
        relationships: list[str] = []
        for neighbor in neighbors[:50]:
            edge_data = graph.get_edge_data(anchor, neighbor) or {}
            relation = "related_to"
            if edge_data:
                first = next(iter(edge_data.values()))
                if isinstance(first, dict):
                    relation = str(first.get("rel", relation))
            relationships.append(f"{neighbor} ({relation})")
        recent_queries = [
            str(entry.get("query", "")).split("\n\nFor my record", 1)[0][:120]
            for entry in recent_history[-3:]
        ]
        if anchor_round <= 1:
            round_guidance = "Discover concrete direct relationships."
        elif anchor_round == 2:
            round_guidance = "Find additional relationships absent from the known list."
        else:
            round_guidance = (
                f"This is round {anchor_round}; find specialized or indirect relationships "
                "not captured previously."
            )
        prompt = f"""Generate one concise {dataset_name} knowledge-graph retrieval query.

FIXED TARGET ENTITY: {anchor}
TARGET DEGREE: {graph.degree(anchor) if anchor in graph else 0}
KNOWN CONNECTIONS: {', '.join(relationships) if relationships else 'none'}
RECENT QUERY TOPICS: {' | '.join(recent_queries) if recent_queries else 'none'}

Requirements:
- The query must explicitly contain the exact target entity name: {anchor}
- Focus only on relationships involving that target
- Do not repeat known connections
- {round_guidance}
- Return only the natural-language query text
"""
        try:
            client = _RUNNER.get_openai_client()
            deployment = _RUNNER.resolve_agent_model(
                "QUERY_GENERATOR", query_generator_model
            )
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate focused, verifiable graph-extraction queries for the "
                            "fixed entity supplied by the controller."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=int(os.getenv("AGEA_QUERY_MAX_TOKENS", "1024")),
                temperature=0.2,
                top_p=1.0,
                **_RUNNER.agent_completion_options(deployment),
            )
            generated = (response.choices[0].message.content or "").strip().strip('"')
            if not generated:
                raise ValueError("fixed-anchor query generator returned empty content")
        except Exception as exc:
            raise RuntimeError(
                f"fixed-anchor query generation failed for {anchor!r}: {exc}"
            ) from exc
        return f"{generated}\n\n{UNIVERSAL_EXTRACTION_COMMAND}"

    def query(
        self,
        prompt: str,
        turn: int,
        *,
        closure_pairs: Sequence[tuple[str, str]] | None = None,
        response_override_path: str | Path | None = None,
        resume_saved_response: bool = False,
        retrieval_query: str | None = None,
    ) -> QueryResult:
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
                raise ValueError("Separated response cannot be resumed as legacy retrieval")
            if context.get("capture_status") == "shared_seed_replay":
                response_source, query_attempt = "shared_seed_replay", 0
            elif context.get("capture_status") == "captured":
                query_attempt = 1
            else:
                raise ValueError("Saved response is not a completed capture")
            response = saved.split(marker, 1)[1]
            if not response.strip() and self.extraction_parser != "bnrr":
                raise ValueError("Empty saved response")
        elif response_override_path is not None:
            response_source = "shared_seed_replay"
            query_attempt = 0
            response, context_path, raw_response_path = _replay_saved_response(
                query=prompt,
                turn=turn,
                source_path=Path(response_override_path),
                context_dir=self.context_dir,
                response_dir=self.response_dir,
                **({"allow_empty_response": True} if self.extraction_parser == "bnrr" else {}),
                **({"retrieval_query": retrieval_query} if retrieval_query is not None else {}),
            )
        else:
            max_attempts = self.graphrag_query_retries + 1
            for query_attempt in range(1, max_attempts + 1):
                try:
                    response, context_path, raw_response_path = _run_live_local_query(
                        query=prompt,
                        turn=turn,
                        graph_root=self.graph_root,
                        data_dir=self.data_dir,
                        context_dir=self.context_dir,
                        response_dir=self.response_dir,
                        query_method=self.query_method,
                        disable_api_thinking=self.disable_api_thinking,
                        runtime=self._async_runtime,
                        **({"allow_empty_response": True} if self.extraction_parser == "bnrr" else {}),
                        **({"retrieval_query": retrieval_query} if retrieval_query is not None else {}),
                    )
                    break
                except Exception as exc:
                    if query_attempt >= max_attempts:
                        raise
                    delay = min(2 ** (query_attempt - 1), 8)
                    print(
                        f"[retry] GraphRAG extraction turn={turn} "
                        f"attempt={query_attempt}/{max_attempts} failed: {exc}; "
                        f"retrying in {delay}s",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
        response_path = Path(raw_response_path)

        closure_stats: dict[str, Any] = {}
        bnrr_parse_stats = None
        if closure_pairs is not None:
            raw_nodes = []
            raw_edges, closure_stats = _parse_closure_response_with_stats(
                response,
                closure_pairs,
                evidence_relationships=_retrieved_relationship_endpoints(
                    context_path
                ),
            )
            compact_edges: list[dict[str, Any]] = []
            rejected_compact_edges = 0
        elif self.extraction_parser == "bnrr":
            from ..bnrr_parser import parse_response, extraction_finish_reasons
            context = json.loads(Path(context_path).read_text())
            request_dir = Path(response_override_path).parent if response_override_path else self.response_dir.parent.parent
            finish_reasons = context.get("extraction_finish_reasons")
            if finish_reasons is None:
                finish_reasons = extraction_finish_reasons(request_dir / "requests.jsonl", 1 if response_override_path else turn)
                context["extraction_finish_reasons"] = finish_reasons
                Path(context_path).write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_nodes, raw_edges, bnrr_parse_stats = parse_response(response,
                extract_body=extract_actual_llm_response, finish_reasons=finish_reasons)
            compact_edges, rejected_compact_edges = [], 0
        else:
            raw_nodes, raw_edges = parse_llm_response_for_graph_items(response)
            compact_edges, rejected_compact_edges = _parse_compact_relationships_with_stats(
                response
            )
            if compact_edges:
                raw_edges = _deduplicate_edges([*raw_edges, *compact_edges])
        kept_nodes, kept_edges = raw_nodes, raw_edges
        filter_status = "disabled"
        filter_response = "Graph filter disabled."
        if self.enable_graph_filter and (raw_nodes or raw_edges):
            kept_nodes, kept_edges, filter_response = filter_extraction_with_graph_filter_agent(
                raw_nodes,
                raw_edges,
                response,
                "Keep only concrete entities and text-supported relationships.",
                "No additional graph context is required; duplicates are removed locally.",
                graph_filter_model=self.graph_filter_model,
            )
            filter_status = (
                "failed_open" if filter_response.startswith("[GRAPH_FILTER_FAILED]") else "enabled"
            )
            (self.filter_dir / f"filter_response_{turn}.txt").write_text(
                filter_response, encoding="utf-8"
            )

        batch = CandidateBatch.from_records(kept_nodes, kept_edges)
        parse_stats = bnrr_parse_stats if bnrr_parse_stats is not None else (
            extraction_response_diagnostics(response, len(raw_nodes), len(raw_edges))
            if closure_pairs is None else {}
        )
        return QueryResult(
            response=response,
            batch=batch,
            stats={
                "source": response_source,
                "graphrag_query_attempts": query_attempt,
                "response_characters": len(response),
                "raw_nodes": len(raw_nodes),
                "raw_edges": len(raw_edges),
                "compact_format_edges": len(compact_edges),
                "citation_rejected_compact_edges": rejected_compact_edges,
                "kept_nodes_explicit": len(kept_nodes),
                "kept_edges": len(kept_edges),
                "candidate_nodes_with_endpoints": len(batch.nodes),
                "candidate_edges": len(batch.edges),
                "graph_filter_status": filter_status,
                **parse_stats,
                **closure_stats,
            },
            retrieved_context_path=context_path,
            response_path=str(response_path),
        )

    def query_grounded_closure(
        self,
        prompt: str,
        turn: int,
        *,
        packets: Sequence[PairEvidencePacket],
        corpus: EvidenceCorpus,
        schema_retries: int = 1,
    ) -> QueryResult:
        """Verify explicit evidence packets with the GraphRAG chat provider.

        Closure does not invoke local retrieval here: the packet windows are
        its complete auditable context.  This makes the pre-call token budget
        exact while retaining the same provider/model and unrestricted output
        ceiling used by GraphRAG local search.  One identical-query retry is
        allowed only when the strict JSON transaction fails validation.
        """

        if not packets:
            raise ValueError("Grounded Closure requires at least one packet")
        if schema_retries < 0:
            raise ValueError("schema_retries must be non-negative")
        deployment = _RUNNER.resolve_agent_model(
            "GRAPHRAG_CHAT_MODEL", "DeepSeek-V4-Flash"
        )
        client = _RUNNER.get_openai_client()
        responses: list[str] = []
        attempt_stats: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        try:
            for attempt in range(1, schema_retries + 2):
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a fail-closed relationship verifier. "
                                "Use only the supplied raw-text evidence windows "
                                "and obey the exact JSON schema in the user prompt."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    top_p=1.0,
                    **_RUNNER.agent_completion_options(deployment),
                )
                response = (completion.choices[0].message.content or "").strip()
                responses.append(response)
                edges, parsed_stats = parse_grounded_closure_response(
                    response, packets, corpus
                )
                attempt_stats.append(parsed_stats)
                attempt_path = self.response_dir / (
                    f"first_llm_response_query_{turn}_attempt{attempt}.txt"
                )
                attempt_path.write_text(
                    _format_response_artifact(prompt, response), encoding="utf-8"
                )
                if parsed_stats["closure_schema_valid"]:
                    break
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        response = responses[-1] if responses else ""
        final_stats = attempt_stats[-1] if attempt_stats else {
            "closure_schema_valid": False,
            "closure_schema_rejection_reasons": ["no_response"],
        }
        response_path = self.response_dir / f"first_llm_response_query_{turn}.txt"
        response_path.write_text(
            _format_response_artifact(prompt, response), encoding="utf-8"
        )
        context_path = self.context_dir / f"evidence_packets_query_{turn}.json"
        context_path.write_text(
            json.dumps(
                {
                    "capture_status": "evidence_packets",
                    "turn": turn,
                    "query": prompt,
                    "packets": [packet.to_dict() for packet in packets],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        batch = CandidateBatch.from_records([], edges)
        return QueryResult(
            response=response,
            batch=batch,
            stats={
                "source": "direct_grounded_evidence",
                "model": deployment,
                "graphrag_query_attempts": 0,
                "closure_schema_attempts": len(attempt_stats),
                "closure_paid_input_proxy_tokens": (
                    count_tokens(prompt) * len(attempt_stats)
                ),
                "closure_total_response_tokens": sum(
                    count_tokens(value) for value in responses
                ),
                "closure_schema_attempt_history": [
                    {
                        "attempt": index,
                        "valid": bool(stats.get("closure_schema_valid")),
                        "reasons": stats.get(
                            "closure_schema_rejection_reasons", []
                        ),
                    }
                    for index, stats in enumerate(attempt_stats, start=1)
                ],
                "response_characters": len(response),
                "raw_nodes": 0,
                "raw_edges": len(edges),
                "compact_format_edges": 0,
                "citation_rejected_compact_edges": 0,
                "kept_nodes_explicit": 0,
                "kept_edges": len(edges),
                "candidate_nodes_with_endpoints": len(batch.nodes),
                "candidate_edges": len(batch.edges),
                "graph_filter_status": "disabled",
                **final_stats,
            },
            retrieved_context_path=str(context_path),
            response_path=str(response_path),
        )


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

    AGEA's subprocess wrapper sets ``GRAPHRAG_LOG_PATH``, but the installed
    GraphRAG CLI does not consume that environment variable.  GraphRAG's Python
    API returns ``context_data`` alongside the response, so the extraction
    adapter calls the same local-search API and serializes the data.  This also
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
        community_level=_RUNNER.COMMUNITY_LEVEL,
        response_type=_RUNNER.RESPONSE_TYPE,
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
    """Call LiteLLM with provider-neutral and provider-specific thinking off."""

    request = dict(kwargs)
    style = os.getenv("AGEA_THINKING_CONTROL_STYLE", "compat").strip().casefold()
    if style == "thinking_disabled":
        extra_body = dict(request.get("extra_body") or {})
        extra_body["thinking"] = {"type": "disabled"}
        request["extra_body"] = extra_body
        return await completion(**request)
    if style == "enable_thinking":
        extra_body = dict(request.get("extra_body") or {})
        extra_body["enable_thinking"] = False
        request["extra_body"] = extra_body
        return await completion(**request)
    if style == "reasoning_disabled":
        extra_body = dict(request.get("extra_body") or {})
        extra_body["reasoning"] = {"enabled": False}
        request["extra_body"] = extra_body
        return await completion(**request)
    if style == "reasoning_effort":
        request["reasoning_effort"] = "none"
        return await completion(**request)
    if style == "none":
        return await completion(**request)
    if style != "compat":
        raise ValueError(
            "AGEA_THINKING_CONTROL_STYLE must be one of: "
            "compat, enable_thinking, reasoning_disabled, reasoning_effort, none"
        )
    request["reasoning_effort"] = "none"
    extra_body = dict(request.get("extra_body") or {})
    extra_body.update(
        {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"},
        }
    )
    request["extra_body"] = extra_body
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
        if "deepseek-v4" not in str(config.get("model", "")).casefold():
            return params
        style = os.getenv("AGEA_THINKING_CONTROL_STYLE", "compat").strip().casefold()
        if style == "thinking_disabled":
            params["extra_body"] = {"thinking": {"type": "disabled"}}
            return params
        if style == "enable_thinking":
            params["extra_body"] = {"enable_thinking": False}
            return params
        if style == "reasoning_disabled":
            params["extra_body"] = {"reasoning": {"enabled": False}}
            return params
        if style == "reasoning_effort":
            params["reasoning_effort"] = "none"
            return params
        if style == "none":
            return params
        if style != "compat":
            raise ValueError(
                "AGEA_THINKING_CONTROL_STYLE must be one of: "
                "compat, enable_thinking, reasoning_disabled, "
                "reasoning_effort, none"
            )
        params["reasoning_effort"] = "none"
        params["extra_body"] = {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"},
        }
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


def append_extraction_command(domain_query: str) -> str:
    return f"{domain_query.strip()}\n\n{UNIVERSAL_EXTRACTION_COMMAND}"


_COMPACT_RELATION_RE = re.compile(
    r"(?mi)^\s*(?:-\s*)?Source\s*:\s*(.+?)\s*(?:→|->)\s*"
    r"Target\s*:\s*(.+?)\s*(?:—|--|\s-\s)\s*(.+?)\s*$"
)

_COMMA_INLINE_RELATION_RE = re.compile(
    r"(?mi)^\s*(?:-\s*)?Source\s*:\s*(.+?)\s*,\s*"
    r"Target\s*:\s*(.+?)\s*,\s*Description\s*:\s*(.+?)\s*$"
)

_RELATIONSHIP_CITATION_RE = re.compile(
    r"\[\s*Data\s*:\s*Relationships?\s*\(", re.IGNORECASE
)

_RELATIONSHIP_CITATION_IDS_RE = re.compile(
    r"\[\s*Data\s*:\s*Relationships?\s*\(([^)]*)\)\s*\]",
    re.IGNORECASE,
)

_NO_SUPPORTED_RELATIONSHIP = "[[NO_SUPPORTED_RELATIONSHIP]]"

_CLOSURE_RELATIONSHIP_RE = re.compile(
    r"(?ims)^\s*RELATIONSHIP_FOUND\s*$\s*"
    r"^\s*Source\s*:\s*(.+?)\s*$\s*"
    r"^\s*Target\s*:\s*(.+?)\s*$\s*"
    r"^\s*Description\s*:\s*(.+?)"
    r"(?=^\s*(?:RELATIONSHIP_FOUND|NO_SUPPORTED_RELATIONSHIP)\s*$|\Z)"
)

_CLOSURE_NO_RELATION_BLOCK_RE = re.compile(
    r"(?im)^\s*NO_SUPPORTED_RELATIONSHIP\s*$\s*"
    r"^\s*Left\s*:\s*(.+?)\s*$\s*"
    r"^\s*Right\s*:\s*(.+?)\s*$"
)

_NEGATIVE_RELATION_RE = re.compile(
    r"(?is)(?:\bno\b|\bnot\b|\bwithout\b|\binsufficient\b).{0,60}"
    r"(?:\bevidence\b|\bsupport(?:ed|ing)?\b|\brelationship\b|\brelation\b)"
    r"|\bunsupported\b"
)


def _strip_optional_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _parse_closure_response_with_stats(
    text: str,
    allowed_pairs: Sequence[tuple[str, str]],
    evidence_relationships: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed on the minimal, direction-neutral Closure contract."""

    response = _strip_optional_code_fence(text)
    sentinel_present = _NO_SUPPORTED_RELATIONSHIP in response
    exact_sentinel = response == _NO_SUPPORTED_RELATIONSHIP
    allowed = {
        frozenset((normalize_label(left), normalize_label(right)))
        for left, right in allowed_pairs
        if normalize_label(left)
        and normalize_label(right)
        and normalize_label(left) != normalize_label(right)
    }
    stats: dict[str, Any] = {
        "closure_parser": "minimal_unordered_evidence_v2",
        "closure_no_relation_sentinel": exact_sentinel,
        "closure_sentinel_conflict": sentinel_present and not exact_sentinel,
        "closure_format_valid": False,
        "closure_candidate_blocks": 0,
        "closure_negative_pair_blocks": 0,
        "closure_rejected_negative_off_pair": 0,
        "closure_expected_pair_count": len(allowed),
        "closure_reported_pair_count": 0,
        "closure_pair_coverage": 0.0,
        "closure_accepted_edges": 0,
        "closure_rejected_off_pair": 0,
        "closure_rejected_missing_citation": 0,
        "closure_rejected_citation_endpoint_mismatch": 0,
        "closure_rejected_negative_description": 0,
        "closure_rejected_invalid_endpoints": 0,
    }
    if exact_sentinel:
        stats["closure_format_valid"] = True
        stats["closure_reported_pair_count"] = len(allowed)
        stats["closure_pair_coverage"] = 1.0 if allowed else 0.0
        return [], stats
    # A response that mixes the negative sentinel with any other content is
    # ambiguous.  Reject the whole response instead of trying to recover edges.
    if sentinel_present:
        return [], stats

    edges: list[dict[str, Any]] = []
    reported_pairs: set[frozenset[str]] = set()
    negative_candidates = _CLOSURE_NO_RELATION_BLOCK_RE.findall(response)
    stats["closure_negative_pair_blocks"] = len(negative_candidates)
    valid_negative_blocks = 0
    for left, right in negative_candidates:
        pair = frozenset(
            (
                normalize_label(left.replace("**", "").strip(" []")),
                normalize_label(right.replace("**", "").strip(" []")),
            )
        )
        if len(pair) != 2 or pair not in allowed:
            stats["closure_rejected_negative_off_pair"] += 1
            continue
        valid_negative_blocks += 1
        reported_pairs.add(pair)

    candidates = _CLOSURE_RELATIONSHIP_RE.findall(response)
    stats["closure_candidate_blocks"] = len(candidates)
    for source, target, description in candidates:
        source = normalize_label(source.replace("**", "").strip(" []"))
        target = normalize_label(target.replace("**", "").strip(" []"))
        description = description.strip()
        if not source or not target or source == target:
            stats["closure_rejected_invalid_endpoints"] += 1
            continue
        if frozenset((source, target)) not in allowed:
            stats["closure_rejected_off_pair"] += 1
            continue
        reported_pairs.add(frozenset((source, target)))
        if _NEGATIVE_RELATION_RE.search(description):
            stats["closure_rejected_negative_description"] += 1
            continue
        if not _RELATIONSHIP_CITATION_RE.search(description):
            stats["closure_rejected_missing_citation"] += 1
            continue
        if evidence_relationships is not None:
            cited_ids = {
                token
                for citation in _RELATIONSHIP_CITATION_IDS_RE.findall(description)
                for token in re.findall(r"[A-Za-z0-9_-]+", citation)
            }
            cited_endpoints = {
                evidence_relationships[citation_id]
                for citation_id in cited_ids
                if citation_id in evidence_relationships
            }
            if (source, target) not in cited_endpoints:
                stats["closure_rejected_citation_endpoint_mismatch"] += 1
                continue
        clean_description = re.sub(r"\[Data:.*?\]", "", description).strip()
        if len(clean_description) <= 5:
            stats["closure_rejected_invalid_endpoints"] += 1
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "rel": "related_to",
                "description": clean_description,
                "weight": 1.0,
                "type": "closure_verified",
            }
        )

    deduplicated = _deduplicate_edges(edges)
    stats["closure_accepted_edges"] = len(deduplicated)
    stats["closure_reported_pair_count"] = len(reported_pairs)
    stats["closure_pair_coverage"] = (
        len(reported_pairs) / len(allowed) if allowed else 0.0
    )
    stats["closure_format_valid"] = bool(candidates or valid_negative_blocks)
    return deduplicated, stats


def _retrieved_relationship_endpoints(
    context_path: str | Path,
) -> dict[str, tuple[str, str]]:
    """Map only the relationship rows exposed to this query's answer model."""

    try:
        payload = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    relationships = payload.get("tables", {}).get("relationships", [])
    if not isinstance(relationships, list):
        return {}
    endpoints: dict[str, tuple[str, str]] = {}
    for record in relationships:
        if not isinstance(record, Mapping):
            continue
        relationship_id = str(record.get("id", "")).strip()
        source = normalize_label(record.get("source"))
        target = normalize_label(record.get("target"))
        if relationship_id and source and target and source != target:
            endpoints[relationship_id] = (source, target)
    return endpoints


def _parse_compact_relationships(text: str) -> list[dict[str, Any]]:
    """Parse citation-grounded one-line renderings of requested triples.

    DeepSeek sometimes ignores the requested multiline format and emits either
    ``Source → Target — Description`` or
    ``Source: A, Target: B, Description: ...``.  Only lines citing GraphRAG's
    relationship table are admitted.  Entity/source citations can describe
    plausible LLM-inferred links that are not edges in the indexed graph, so
    accepting them would improve recall at the cost of severe precision loss.
    """

    edges, _ = _parse_compact_relationships_with_stats(text)
    return edges


def _parse_compact_relationships_with_stats(
    text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Return accepted compact edges and the number rejected by citation gate."""

    edges: list[dict[str, Any]] = []
    rejected = 0
    candidates = [
        *_COMPACT_RELATION_RE.findall(text),
        *_COMMA_INLINE_RELATION_RE.findall(text),
    ]
    for source, target, description in candidates:
        if not _RELATIONSHIP_CITATION_RE.search(description):
            rejected += 1
            continue
        source = source.replace("**", "").strip()
        target = target.replace("**", "").strip()
        description = re.sub(r"\[Data:.*?\]", "", description).strip()
        if source and target and source != target and len(description) > 5:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "rel": "related_to",
                    "description": description,
                    "weight": 1.0,
                    "type": "extracted_compact",
                }
            )
    return _deduplicate_edges(edges), rejected


def _deduplicate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (
            str(edge.get("source", "")).strip().upper(),
            str(edge.get("rel", "related_to")).strip().lower(),
            str(edge.get("target", "")).strip().upper(),
        )
        if key[0] and key[2] and key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique
