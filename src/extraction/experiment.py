#!/usr/bin/env python3
"""BNRR experiment runtime with durable round commits and offline replay.

Runtime files are local to tmp; no provider credentials are written to manifests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from extraction.bnrr_config import DEFAULTS
from extraction import moderation
from extraction.bnrr_queries import generate_explore_query, generate_exploit_query, QueryGenerationError
from extraction.bnrr_prompt_profiles import call_query, load_profile, manifest_profile, profile_settings
from extraction.paths import graph_root as index_root
SOURCE_ENV = PROJECT / ".env"
METHODS = ("FULL",)
SEEDS = DEFAULTS.seeds
METRICS = ("node_precision", "node_recall", "edge_pair_precision", "edge_pair_recall", "node_f1", "edge_pair_f1")
SUMMARY_METRICS = (*METRICS, "node_par", "edge_pair_par")


def compute_novelty(cumulative_nodes, cumulative_edges, current_nodes, current_edges):
    """Preserve the diagnostic novelty statistic used in saved query history."""
    if not current_nodes and not current_edges:
        return 0.0
    node_intersection = len(cumulative_nodes & current_nodes)
    node_novelty = 1.0 - node_intersection / len(current_nodes) if current_nodes else 0.0
    seen_edges = set()
    for source, target in cumulative_edges:
        seen_edges.add((source.upper(), target.upper()))
        seen_edges.add((target.upper(), source.upper()))
    edge_intersection = sum((source.upper(), target.upper()) in seen_edges
                            for source, target in current_edges)
    edge_novelty = 1.0 - edge_intersection / len(current_edges) if current_edges else 0.0
    return (node_novelty * len(current_nodes) + edge_novelty * len(current_edges)) / (
        len(current_nodes) + len(current_edges))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append(path, data):
    with Path(path).open("a") as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def environment():
    from dotenv import load_dotenv
    load_dotenv(SOURCE_ENV, override=True)
    from urllib.parse import urlsplit
    required = ("PROVIDER_API_KEY", "PROVIDER_API_BASE", "PROVIDER_CHAT_MODEL",
                "PROVIDER_EMBEDDING_API_KEY", "PROVIDER_EMBEDDING_API_BASE", "PROVIDER_EMBEDDING_MODEL")
    for name in required:
        if not os.getenv(name, "").strip() or os.environ[name] == "replace-me":
            raise RuntimeError(f"Set {name} in .env")
    for name in ("PROVIDER_API_BASE", "PROVIDER_EMBEDDING_API_BASE"):
        endpoint = urlsplit(os.environ[name])
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise RuntimeError(f"Invalid API endpoint: {name}")
    os.environ.update({"PYTHONPATH": str(PROJECT / "src"),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})


# Capture completion reasons and JSON/SSE usage for all new runs.
from extraction.request_audit import RequestAudit


def render(library, graph, turn, mode, anchor, history, *, explore_query=None, exploit_query=None):
    from extraction.control.actions import ActionKind, QueryAction
    # The controller owns topology; extraction never receives scores, degrees,
    # an allegedly complete adjacency list, or recursively rendered history.
    if mode == "exploit" and not anchor:
        raise ValueError("Exploit requires a fixed anchor")
    if turn > 1 and mode == "explore" and not explore_query:
        raise ValueError("Explore requires a generated query before retrieval")
    if turn > 1 and mode == "exploit" and not exploit_query:
        raise ValueError("Exploit requires a generated query before retrieval")
    values = {"query": (exploit_query if mode == "exploit" else explore_query) or "", "anchor": anchor or ""}
    action = QueryAction(kind=ActionKind.INCIDENT if mode == "exploit" else ActionKind.GLOBAL, anchor=anchor)
    return library.render(action, values=values, seed=turn == 1).text


def parser_for_turn(manifest, method, turn):
    parser = manifest.get("parser_by_method", {}).get(method, "bnrr")
    if parser != "bnrr" or manifest.get("bnrr_parser_migration"):
        raise RuntimeError("Historical parsers require the original frozen runtime")
    return parser



def check_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if "rank_schedule" in manifest:
        from extraction.control.rank_bnrr import validate_spec
        validate_spec(manifest["rank_schedule"])
    from extraction.separated_query import PROTOCOL
    if (manifest.get("prompt_protocol") != "bnrr-pure-retrieval-null" or
            manifest.get("retrieval_by_method") != {m: PROTOCOL for m in manifest["methods"]} or
            manifest.get("seed_retrieval_protocol") != PROTOCOL or
            manifest.get("bnrr_parser_protocol") != "bnrr-null-record-local" or
            manifest.get("extraction_completion_policy") != "skip-length-and-unparseable-v1"):
        raise RuntimeError("Frozen BNRR extraction contract is inconsistent")
    for method in manifest["methods"]:
        parser_for_turn(manifest, method, 1)
    prompt_config = manifest_profile(manifest)
    if manifest.get("moderation_policy") not in (None, DEFAULTS.moderation_policy):
        raise RuntimeError("Unknown frozen moderation policy")
    if manifest.get("moderation_policy") and manifest.get("moderation_attempts") != DEFAULTS.moderation_attempts:
        raise RuntimeError("Unexpected moderation attempt budget")
    expected_anchor_protocol = ("anchor-single-quote-escape-v1" if prompt_config["memory_profile"] == "recent_exclusion_desc" else "literal-v1")
    if manifest.get("query_anchor_protocol", "literal-v1") != expected_anchor_protocol:
        raise RuntimeError("Frozen anchor normalization protocol is inconsistent")
    if "prompt_files" in manifest:
        library = load_profile(PROJECT, prompt_config["profile"])
        expected_files = {k: str(p.relative_to(PROJECT)) for k, p in library.paths.items()}
        if manifest["prompt_files"] != expected_files:
            raise RuntimeError("Frozen prompt file mapping is inconsistent")
    for name, digest in manifest["code_hashes"].items():
        if sha(PROJECT / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, digest in manifest["runtime_hashes"].items():
        if sha(root / name) != digest:
            raise RuntimeError(f"Frozen runtime input changed: {name}")
    return manifest


def retrieval_for_turn(manifest, method, turn, seed_library, generated_query=None):
    """Use writer output directly, never split or strip a rendered instruction."""
    from extraction.separated_query import PROTOCOL
    policy = (manifest.get("seed_retrieval_protocol") if turn == 1
        else manifest.get("retrieval_by_method", {}).get(method))
    if policy != PROTOCOL:
        raise ValueError("Unknown retrieval protocol")
    query = seed_library.templates["seed"] if turn == 1 else generated_query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Missing pure retrieval query; refuse combined-prompt fallback")
    return query


def make_controller(manifest, seed):
    from extraction.control.bnrr import BnrrController, PROTOCOL
    if manifest.get("coverage_protocol", PROTOCOL) != PROTOCOL:
        raise RuntimeError("Historical controllers require the original frozen runtime")
    controller_type, extra = BnrrController, {}
    if "rank_schedule" in manifest:
        from extraction.control.rank_bnrr import RankBnrrController, validate_spec
        validate_spec(manifest["rank_schedule"])
        controller_type, extra = RankBnrrController, {"direction": manifest["rank_schedule"]["direction"]}
    return controller_type(manifest["horizon"], seed=seed, **extra,
        q_hi=manifest.get("q_hi", .9), q_lo=manifest.get("q_lo", .1), rho=manifest.get("rho", .2),
        gate_policy=manifest.get("bnrr_gate_policy", "residual_mass"),
        neighbor_penalty=manifest.get("neighbor_penalty", .5))



def worker(root, method, seed, seed_only=False, resume=False):
    if resume and (method != "FULL" or seed_only):
        raise ValueError("Resume supports the FULL worker, not seed generation")
    if method != "FULL":
        raise ValueError("The DIVER runtime supports FULL only")
    environment()
    manifest = check_manifest(root)
    if manifest.get("embedding_model") and manifest["embedding_model"] != os.environ.get("PROVIDER_EMBEDDING_MODEL"):
        raise RuntimeError("Embedding environment changed since manifest was frozen")
    if resume:
        for key in ("PROVIDER_CHAT_MODEL",):
            if manifest.get("chat_model") and os.environ.get(key) != manifest["chat_model"]:
                raise RuntimeError(f"Chat model changed on resume: {key}")
    import networkx as nx
    import numpy as np
    from openai import OpenAI
    from extraction.backends.graphrag import AgeaGraphRagAdapter, _RUNNER
    from extraction.control.bnrr import keyed_rng
    from extraction.metrics.graph import merge_batch
    from extraction.models import CandidateBatch
    from evaluation.graph_recovery import TruthData, evaluate_recovery
    random.seed(seed)
    np.random.seed(seed)
    out = root / ("seed_generation" if seed_only else "runs") / (f"seed{seed}" if seed_only else f"{method}_seed{seed}")
    out.mkdir(parents=True, exist_ok=resume)
    from extraction.resume import RoundJournal, restore_accounting, saved_query
    journal = RoundJournal(out, sha(root / "manifest.json"), resume=resume,
        parent_manifest_sha=manifest.get("resume_from_manifest_sha256"),
        config_manifest_sha=manifest.get("resume_config_manifest_sha256"),
        commit_manifest_sha=manifest.get("resume_commit_manifest_sha256")) if method == "FULL" and not seed_only else None
    if resume:
        config = json.loads((out / "config.json").read_text())
        if (config["method"], config["seed"], config["horizon"]) != (method, seed, manifest["horizon"]):
            raise RuntimeError("Resume method/seed/horizon mismatch")
    requests = RequestAudit(out / "requests.jsonl")
    requests.journal_starts = bool(journal)
    if resume:
        restore_accounting(requests)
    requests.install()
    _RUNNER.get_openai_client = lambda *args, **kwargs: OpenAI(api_key=os.environ["PROVIDER_API_KEY"],
        base_url=os.environ["PROVIDER_API_BASE"], max_retries=2, timeout=120)
    graph = nx.MultiDiGraph()
    truth = TruthData.load(root / "graph_root" / "output")
    controller = make_controller(manifest, seed)
    moderation_enabled = manifest.get("moderation_policy") == moderation.POLICY
    moderation_attempts = manifest.get("moderation_attempts", DEFAULTS.moderation_attempts)
    prompt_config = manifest_profile(manifest)
    library = load_profile(PROJECT, prompt_config["profile"])
    seed_library = load_profile(PROJECT, manifest.get("initialization_prompt_profile", "agea_adapted_null"))
    history, records = [], []
    epsilon = .3
    if resume and journal.completed:
        restored = audit_run(root, method, seed, prefix=journal.completed, return_state=True)
        graph, controller = restored["graph"], restored["controller"]
        history, records, epsilon = restored["history"], restored["records"], restored["epsilon"]
    if resume:
        if journal.completed >= manifest["horizon"]:
            # A crash after the last commit can safely regenerate terminal artifacts.
            print("All rounds committed; regenerating terminal outputs only", flush=True)
        previous = {name: json.loads((out / name).read_text()) for name in
            ("status.json", "FAILURE.json", "parse_failure.json", "query_generation_failure.json") if (out / name).exists()}
        append(out / "resume_events.jsonl", {"time": time.time(), "pid": os.getpid(),
            "completed_rounds": journal.completed, "requests_carried": requests.count,
            "errors_carried": requests.errors, "previous_status": previous})
        journal.rebuild_csv()
        write_json(out / "status.json", {"status": "running", "round": journal.completed,
            "request_errors": requests.errors, "updated": time.time(), "pid": os.getpid(),
            "resumed": True})
    adapter = AgeaGraphRagAdapter(graph_root=str(root / "graph_root"), data_dir=str(root / "graph_root/output"),
        run_dir=out, disable_api_thinking=True, graphrag_query_retries=moderation_attempts - 1, enable_graph_filter=False,
        extraction_parser=parser_for_turn(manifest, method, 1))
    if not resume:
        write_json(out / "config.json", {"method": method, "seed": seed, "horizon": 1 if seed_only else manifest["horizon"],
        "manifest_sha256": sha(root / "manifest.json"), "thinking_disabled": True, "filtering": False,
        "bnrr_prompt_profile": prompt_config["profile"], "query_memory_profile": prompt_config["memory_profile"]})
    try:
        for turn in range(len(records) + 1, (1 if seed_only else manifest["horizon"]) + 1):
            started = time.time()
            adapter.extraction_parser = parser_for_turn(manifest, method, turn)
            tolerant = adapter.extraction_parser == "bnrr"
            query_skip_stats = None
            requests.turn = turn
            recent = statistics.mean(h["novelty"] for h in history[-5:]) if history else 0.0
            decision = moderation.choose_decision(controller, graph, turn, history, moderation_enabled)
            seeds_used = []
            mode, anchor = decision["mode"], decision["anchor"]
            explore_query = exploit_query = None
            if turn > 1:
                model = _RUNNER.resolve_agent_model("PROVIDER_CHAT_MODEL", "deepseek-v4-flash")
                writer = generate_explore_query if mode == "explore" else generate_exploit_query
                if moderation_enabled:
                    from extraction.bnrr_queries import exploration_messages, exploitation_messages
                    messages = (call_query(exploitation_messages, library, graph, history, anchor,
                        memory_profile=prompt_config["memory_profile"]) if mode == "exploit" else
                        call_query(exploration_messages, library, graph, history, memory_profile=prompt_config["memory_profile"]))
                    writer_payload = {"messages": messages, "model": model}
                try:
                    cached = journal.get("query_generation.jsonl", turn) if journal else None
                    failure = (moderation.load_failure(out, "query_generation", turn, writer_payload, moderation_attempts)
                        if moderation_enabled else None)
                    if failure is not None:
                        query_audit = moderation.rejected_query(writer_payload, mode, anchor)
                        if cached is not None and cached != {"turn": turn, **query_audit}:
                            raise RuntimeError("Saved moderation rejection/query mismatch")
                        generated_query = None
                        query_skip_stats = moderation.skip_stats(failure)
                    elif cached is not None:
                        if cached["model"] != model:
                            raise RuntimeError("Query model changed on resume")
                        generated_query, query_audit = saved_query(library, graph, history, cached,
                            decision, prompt_config["memory_profile"])
                    else:
                        operation = lambda: call_query(writer, library, graph, history,
                            client=_RUNNER.get_openai_client(), model=model,
                            completion_options=_RUNNER.agent_completion_options(model),
                            memory_profile=prompt_config["memory_profile"],
                            **({"anchor": anchor} if mode == "exploit" else {}))
                        value, failure = (moderation.call(operation, out, "query_generation", turn,
                            writer_payload, moderation_attempts) if moderation_enabled else (operation(), None))
                        if failure is not None:
                            generated_query = None
                            query_audit = moderation.rejected_query(writer_payload, mode, anchor)
                            query_skip_stats = moderation.skip_stats(failure)
                        else:
                            generated_query, query_audit = value
                except QueryGenerationError as error:
                    if not tolerant or error.audit.get("rejection_reason") == "unexpected_reasoning":
                        write_json(out / "query_generation_failure.json", {"turn": turn,
                            "error_type": type(error).__name__, "audit": error.audit})
                        raise
                    query_audit = error.audit
                    generated_query = None
                    query_skip_stats = {"parse_status": "skipped", "round_skipped": True,
                        "skip_reason": query_audit.get("rejection_reason", "invalid_query"), "skip_stage": "query_generation"}
                except Exception as error:
                    write_json(out / "query_generation_failure.json", {"turn": turn,
                        "error_type": type(error).__name__, "audit": getattr(error, "audit", None)})
                    raise
                if mode == "explore": explore_query = generated_query
                else: exploit_query = generated_query
                if journal: journal.append("query_generation.jsonl", {"turn": turn, **query_audit})
                else: append(out / "query_generation.jsonl", {"turn": turn, **query_audit})
            query = "" if query_skip_stats else render(seed_library if turn == 1 else library, graph, turn, mode, anchor, history,
                explore_query=explore_query, exploit_query=exploit_query)
            if journal: journal.append("decisions.jsonl", decision)
            else: append(out / "decisions.jsonl", decision)
            if turn == 1 and manifest.get("reused_seed_source"):
                query = (root / "seed_generation" / f"seed{seed}" / "query.txt").read_text()
            before_nodes, before_edges = set(graph.nodes), set(graph.edges())
            retrieval_query = None if query_skip_stats else retrieval_for_turn(manifest, method, turn, seed_library,
                (exploit_query if mode == "exploit" else explore_query))
            retrieval_options = {"retrieval_query": retrieval_query} if retrieval_query is not None else {}
            response_override = None if seed_only or turn != 1 else root / "seed_generation" / f"seed{seed}" / "response.txt"
            saved_response = out / "turn_logs/llm_responses" / f"first_llm_response_query_{turn}.txt"
            if query_skip_stats:
                result = SimpleNamespace(batch=CandidateBatch(), stats=query_skip_stats)
            else:
                extraction_payload = {"query": query, "retrieval_query": retrieval_query}
                failure = (moderation.load_failure(out, "extraction", turn, extraction_payload, moderation_attempts)
                    if moderation_enabled else None)
                if moderation_enabled and response_override is not None:
                    initial_failure = moderation.load_failure(response_override.parent, "extraction", turn,
                        extraction_payload, moderation_attempts)
                    if initial_failure:
                        failure = moderation.save_failure(out, "extraction", turn, extraction_payload, moderation_attempts)
                if failure is not None:
                    result = SimpleNamespace(batch=CandidateBatch(), stats=moderation.skip_stats(failure))
                elif resume and saved_response.exists():
                    result = adapter.query(query, turn, resume_saved_response=True, **retrieval_options)
                elif journal and journal.get("graph_delta.jsonl", turn) is not None:
                    raise RuntimeError("Uncommitted graph records without saved response; refusing requery")
                else:
                    operation = lambda: adapter.query(query, turn, response_override_path=response_override, **retrieval_options)
                    result, failure = (moderation.call(operation, out, "extraction", turn,
                        extraction_payload, moderation_attempts) if moderation_enabled else (operation(), None))
                    if failure is not None:
                        result = SimpleNamespace(batch=CandidateBatch(), stats=moderation.skip_stats(failure))
            parse_stats = getattr(result, "stats", {})
            if parse_stats.get("parse_status") in {"format_error", "unstructured_empty"}:
                if not tolerant:
                    write_json(out / "parse_failure.json", {"turn": turn, "anchor": anchor, "stats": parse_stats})
                    write_json(out / "status.json", {"status": "failed", "round": turn - 1,
                        "failed_round": turn, "reason": parse_stats["parse_status"], "pid": os.getpid()})
                    raise ValueError(f"Extraction format error at round {turn}; inspect saved response before retrying")
                parse_stats = {**parse_stats, "skip_reason": parse_stats["parse_status"],
                    "parse_status": "skipped", "round_skipped": True}
            skipped = bool(parse_stats.get("round_skipped"))
            if skipped:
                result.batch = CandidateBatch()
            merge_batch(graph, result.batch)
            if mode == "exploit" and not skipped:
                controller.complete(anchor)
            elif mode == "exploit" and skipped:
                controller.skip(anchor)
            novelty = 0.0 if turn == 1 else compute_novelty(before_nodes, before_edges,
                set(result.batch.nodes), {atom.pair for atom in result.batch.edges})
            h = {"turn": turn, "mode": mode, "type": mode, "query": query, "novelty": novelty,
                "nodes_added_to_graph": len(set(graph.nodes) - before_nodes), "edges_added_to_graph": len(set(graph.edges()) - before_edges),
                "newly_discovered_entity_names": sorted(set(graph.nodes) - before_nodes),
                "seeds_used": [x[0] for x in seeds_used], "seeds_with_rounds": seeds_used, "epsilon": epsilon,
                "query_intent": query.split("\n\n", 1)[0], "parse_stats": parse_stats,
                "incident_pairs": sum(anchor in atom.pair for atom in result.batch.edges) if anchor else None,
                "collateral_pairs": sum(anchor not in atom.pair for atom in result.batch.edges) if anchor else None,
                "new_incident_pairs": sum(anchor in atom.pair and atom.pair not in before_edges for atom in result.batch.edges) if anchor else None,
                "new_collateral_pairs": sum(anchor not in atom.pair and atom.pair not in before_edges for atom in result.batch.edges) if anchor else None}
            if retrieval_query is not None:
                h["retrieval_query"] = retrieval_query
            if (manifest.get("history_observation_protocol") == "response-visible-v1" or
                    prompt_config["memory_profile"] in ("recent_exclusion_desc",)):
                # Response-visible feedback only, never retrieval context or truth.
                h.update(anchor=anchor, returned_entity_names=sorted(result.batch.nodes))
            nodes, edges = result.batch.to_records()
            delta = {"turn": turn, "nodes": nodes, "edges": edges}
            if journal:
                journal.append("graph_delta.jsonl", delta)
                journal.append("history.jsonl", h)
            else:
                append(out / "graph_delta.jsonl", delta)
                append(out / "history.jsonl", h)
            history.append(h)
            metrics = evaluate_recovery(graph, truth)
            row = {"turn": turn, "mode": mode, **{k: metrics[k] for k in METRICS},
                "matched_nodes": metrics["matched_nodes"], "matched_edges": metrics["matched_directed_edge_pairs"],
                "provider_requests": requests.count, "request_errors": requests.errors, "input_tokens": requests.input_tokens,
                "output_tokens": requests.output_tokens, "unknown_usage_requests": requests.unknown_usage,
                "seconds": time.time() - started, "novelty": novelty}
            if tolerant:
                row.update(round_status=parse_stats["parse_status"], skip_reason=parse_stats.get("skip_reason"),
                    skipped_records=len(parse_stats.get("skipped_records", [])))
            if journal:
                old_row = journal.get("metrics.jsonl", turn)
                if old_row is not None:
                    if any(old_row[k] != row[k] for k in row if k != "seconds"):
                        raise RuntimeError("Uncommitted metrics mismatch")
                    row = old_row
                journal.append("metrics.jsonl", row)
                journal.commit(turn, write_json)
                journal.rebuild_csv()
            else:
                append(out / "metrics.jsonl", row)
                with (out / "metrics.csv").open("a") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if turn == 1: w.writeheader()
                    w.writerow(row)
            records.append(row)
            write_json(out / "status.json", {"status": "running", "round": turn, "request_errors": requests.errors,
                "updated": time.time(), "pid": os.getpid()})
            print(f"{method} seed={seed} round={turn} mode={mode} requests={requests.count} errors={requests.errors}", flush=True)
            if skipped or parse_stats.get("skipped_records"):
                print(f"[skip] round={turn} status={parse_stats['parse_status']} reason={parse_stats.get('skip_reason')} records={len(parse_stats.get('skipped_records', []))}", flush=True)
            if seed_only:
                if parse_stats.get("skip_reason") == "content_moderation":
                    # Empty seed placeholder only; the sidecar records that no
                    # provider response was obtained and is replayed separately.
                    (out / "response.txt").write_text("")
                else:
                    shutil.copyfile(result.response_path, out / "response.txt")
                if tolerant:
                    write_json(out / "extraction_completion.json", {"finish_reasons": parse_stats.get("finish_reasons", [])})
                if retrieval_query is not None:
                    (out / "retrieval_query.txt").write_text(retrieval_query)
            if turn > 1:
                epsilon = max(.05, epsilon * .98)
        nx.write_graphml(graph, out / "final_graph.graphml")
        write_json(out / "summary.json", {"method": method, "seed": seed, "completed_rounds": len(records),
            "J_E": statistics.mean(r["edge_pair_f1"] for r in records[1:]) if len(records) > 1 else None,
            "final": records[-1], "exploit_rounds": sum(h["mode"] == "exploit" for h in history),
            "skipped_rounds": sum(bool(h["parse_stats"].get("round_skipped")) for h in history),
            "partial_rounds": sum(h["parse_stats"].get("parse_status") == "partial" for h in history)})
        write_json(out / "status.json", {"status": "completed", "round": len(records), "request_errors": requests.errors, "pid": os.getpid()})
    finally:
        adapter.close()
        if journal: journal.close()


def prepare(root, *, dataset="novel", horizon=DEFAULTS.rounds, methods=("FULL",), gate_policy=DEFAULTS.gate_policy, prompt_profile=DEFAULTS.prompt_profile, soft_bnrr=True):
    environment()
    import yaml
    if dataset not in {"novel", "medical", "agriculture"} or horizon < 2 or not methods or any(m not in METHODS for m in methods):
        raise ValueError("Invalid experiment configuration")
    prompt_config = profile_settings(prompt_profile)
    from extraction.separated_query import PROTOCOL as retrieval_protocol
    library = load_profile(PROJECT, prompt_profile)
    source_graph = index_root(dataset)
    from extraction.control.bnrr import BnrrController
    BnrrController(horizon, gate_policy=gate_policy)
    if not soft_bnrr or gate_policy != "residual_mass" or tuple(methods) != ("FULL",):
        raise ValueError("Soft BNRR requires residual_mass and FULL-only runs")
    settings_source = source_graph / "settings.yaml"
    import lancedb
    table = lancedb.connect(str(source_graph / "output/lancedb")).open_table("default-entity-description")
    if table.schema.field("vector").type.list_size != 4096:
        raise RuntimeError(f"Original {dataset} index must contain 4096-dimensional vectors")
    root.mkdir(parents=True, exist_ok=False)
    graph_root = root / "graph_root"
    graph_root.mkdir()
    settings = yaml.safe_load(settings_source.read_text())
    for model in settings["models"].values():
        model["max_retries"] = 2
        model["request_timeout"] = 120
    settings["models"]["default_chat_model"]["model"] = "${PROVIDER_CHAT_MODEL}"
    # Freeze completion settings explicitly instead of inheriting SDK defaults.
    settings["models"]["default_chat_model"].update(max_tokens=16384, temperature=0, top_p=1,
        type="openai_chat", encoding_model="cl100k_base")
    settings["vector_store"]["default_vector_store"]["db_uri"] = str((source_graph / "output/lancedb").resolve())
    (graph_root / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    shutil.copytree(source_graph / "prompts", graph_root / "prompts")
    (graph_root / "output").symlink_to((source_graph / "output").resolve(), target_is_directory=True)
    sources = [p for base in ("src", "configs/prompts", "scripts") for p in (PROJECT / base).rglob("*")
        if p.is_file() and p.suffix in {".py", ".txt"}]
    runtime = [settings_source, graph_root / "settings.yaml", *sorted((graph_root / "prompts").glob("*")), *sorted((source_graph / "output").glob("*.parquet")),
        *sorted(p for p in (source_graph / "output/lancedb").rglob("*") if p.is_file())]
    write_json(root / "manifest.json", {"dataset": dataset, "horizon": horizon, "seeds": SEEDS, "methods": methods,
        **({"coverage_protocol": "bnrr-soft-self-onehop", "neighbor_penalty": .5} if soft_bnrr else {}),
        "reused_seed_source": None,
        "settings_source": str(settings_source),
        "max_workers": len(SEEDS), "created": time.time(), "q_hi": DEFAULTS.q_hi, "q_lo": DEFAULTS.q_lo, "rho": DEFAULTS.rho,
        "formal_defaults": DEFAULTS.to_dict(), "rank_schedule": DEFAULTS.rank_schedule,
        "moderation_policy": DEFAULTS.moderation_policy, "moderation_attempts": DEFAULTS.moderation_attempts,
        "gate": {"epsilon": .3, "decay": .98, "min": .05, "threshold": .15, "window": 5, "adaptive": True, "success_rate_detection": True},
        "chat_model": os.environ["PROVIDER_CHAT_MODEL"], "chat_api_base": os.environ["PROVIDER_API_BASE"], "thinking": False,
        "extra_llm_filter": False, "adapter_query_retries": 2, "sdk_retries": 2, "http_timeout": 120,
        "prompt_protocol": "bnrr-pure-retrieval-null",
        "retrieval_by_method": {m: retrieval_protocol for m in methods},
        "seed_retrieval_protocol": retrieval_protocol,
        "bnrr_prompt_profile": prompt_profile,
        "query_memory_profile": prompt_config["memory_profile"],
        "query_anchor_protocol": "anchor-single-quote-escape-v1" if prompt_config["memory_profile"] == "recent_exclusion_desc" else "literal-v1",
        "prompt_files": {k: str(p.relative_to(PROJECT)) for k, p in library.paths.items()},
        "initialization_prompt_profile": prompt_profile,
        "history_observation_protocol": "response-visible-v1",
        "bnrr_gate_policy": gate_policy,
        "prompt_directory": "configs/prompts/bnrr",  # Base; prompt_files records per-file overrides.
        "parser_protocol": "method-specific-see-parser_by_method",
        "parser_by_method": {method: "bnrr" for method in methods},
        "bnrr_parser_protocol": "bnrr-null-record-local",
        "extraction_completion_policy": "skip-length-and-unparseable-v1",
        "round_failure_policy": "record-local-skip; skipped rounds consume budget but do not complete anchors",
        "embedding_model": os.environ["PROVIDER_EMBEDDING_MODEL"],
        "embedding_api_base": os.environ["PROVIDER_EMBEDDING_API_BASE"],
        "embedding_dimensions": 4096,
        "query_writer": {"methods": ["FULL"], "action": "all_postseed",
            "explore_temperature": 0.3, "exploit_temperature": 0.2, "max_tokens": 1024, "recent_queries": 3,
            "recent_entity_rounds": 2 if prompt_config["memory_profile"] in ("recent_exclusion_desc",) else None,
            "max_entities": 10, "memory_profile": prompt_config["memory_profile"],
            "recent_exclusion_settings": ({"recent_questions": 3, "response_window": 10,
                "max_common_entities": 5, "min_observed_rounds": 2, "applies_to": "explore_topic_only"}
                if prompt_config["memory_profile"] in ("recent_exclusion_desc",) else None),
            "description_guidance": ({"explore_max_entities": 10, "explore_description_chars": 240,
                "anchor_description_chars": 480, "max_partial_connections": 20, "connection_description_chars": 160}
                if prompt_config["memory_profile"] == "recent_exclusion_desc" else None),
            "uses_observed_gain_counts": False,
            "uses_novelty_or_topology": False, "uses_truth_or_retrieval_tables": False},
        "evaluation": f"directed endpoint pairs; offline truth; J_E mean F1 over rounds 2..{horizon}",
        "code_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in sources},
        "runtime_hashes": {os.path.relpath(p, root): sha(p) for p in runtime}})
    print(json.dumps({"root": str(root), "prepared": True, "api_requests": 0}))


def audit_run(root, method, seed, *, prefix=None, return_state=False):
    import networkx as nx
    from extraction.control.bnrr import keyed_rng
    from extraction.metrics.graph import merge_batch
    from extraction.models import CandidateBatch
    from extraction.backends.graphrag import _RUNNER
    from evaluation.graph_recovery import TruthData, evaluate_recovery
    out = root / "runs" / f"{method}_seed{seed}"
    def read(name):
        path = out / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    decisions, deltas, rows, history = [read(n) for n in ("decisions.jsonl", "graph_delta.jsonl", "metrics.jsonl", "history.jsonl")]
    manifest = json.loads((root / "manifest.json").read_text())
    parser_name = manifest.get("parser_by_method", {}).get(method)
    reparse_bnrr = parser_name == "bnrr"
    if reparse_bnrr:
        from extraction.backends import graphrag as backend
    horizon = manifest["horizon"] if prefix is None else prefix
    if prefix is not None and (method != "FULL" or not 0 < prefix <= manifest["horizon"]):
        raise RuntimeError("Invalid resume prefix")
    if any(len(items) < horizon if prefix is not None else len(items) != horizon for items in (decisions, deltas, rows, history)):
        raise RuntimeError("Artifact sequence does not match frozen horizon")
    decisions, deltas, rows, history = [items[:horizon] for items in (decisions, deltas, rows, history)]
    graph = nx.MultiDiGraph()
    truth = TruthData.load(root / "graph_root/output")
    control = make_controller(manifest, seed)
    moderation_enabled = manifest.get("moderation_policy") == moderation.POLICY
    moderation_attempts = manifest.get("moderation_attempts", DEFAULTS.moderation_attempts)
    epsilon = .3
    generated = None
    prompt_config = manifest_profile(manifest)
    if manifest.get("prompt_protocol") == "bnrr-pure-retrieval-null":
        from extraction.bnrr_queries import exploration_messages, exploitation_messages
        library = load_profile(PROJECT, prompt_config["profile"])
        generation_rows = read("query_generation.jsonl")
        if prefix is not None: generation_rows = generation_rows[:max(0, horizon - 1)]
        if [item["turn"] for item in generation_rows] != list(range(2, horizon + 1)):
            raise RuntimeError("Query generation sequence mismatch")
        generated = {item["turn"]: item for item in generation_rows}
    for turn, (decision, delta, row) in enumerate(zip(decisions, deltas, rows), 1):
        if any(item["turn"] != turn for item in (decision, delta, row)):
            raise RuntimeError("Round sequence mismatch")
        if method == "FULL":
            if moderation.choose_decision(control, graph, turn, history[:turn-1], moderation_enabled) != decision:
                raise RuntimeError(f"BNRR pre-query admission contract mismatch at {turn}")
        recorded_stats = history[turn-1].get("parse_stats", {})
        skipped = bool(recorded_stats.get("round_skipped"))
        query_skipped = recorded_stats.get("skip_stage") == "query_generation"
        moderation_skipped = recorded_stats.get("skip_reason") == "content_moderation"
        if moderation_skipped and not moderation_enabled:
            raise RuntimeError("Moderation skip is not enabled in the frozen manifest")
        if skipped and parser_for_turn(manifest, method, turn) != "bnrr":
            raise RuntimeError(f"Skip not allowed by historical protocol at {turn}")
        if decision["mode"] == "exploit" and not skipped:
            control.complete(decision["anchor"])
        elif decision["mode"] == "exploit" and skipped:
            control.skip(decision["anchor"])
        if generated is not None and turn > 1:
            query = generated[turn]
            mode, anchor = decision["mode"], decision["anchor"]
            messages = (call_query(exploitation_messages, library, graph, history[:turn-1], anchor, memory_profile=prompt_config["memory_profile"]) if mode == "exploit"
                else call_query(exploration_messages, library, graph, history[:turn-1], memory_profile=prompt_config["memory_profile"]))
            if query["messages"] != messages or query["action"] != mode or query["anchor"] != anchor:
                raise RuntimeError(f"Online query-generation input mismatch at {turn}")
            if query_skipped:
                if moderation_skipped:
                    payload = {"messages": messages, "model": query["model"]}
                    expected = moderation.rejected_query(payload, mode, anchor)
                    if query != {"turn": turn, **expected}:
                        raise RuntimeError("Moderation query audit mismatch")
                else:
                    from extraction.resume import saved_query
                    try:
                        saved_query(library, graph, history[:turn-1], query, decision, prompt_config["memory_profile"])
                    except QueryGenerationError as error:
                        if error.audit != {k: v for k, v in query.items() if k != "turn"} or error.audit.get("rejection_reason") != recorded_stats.get("skip_reason"):
                            raise RuntimeError(f"Rejected query replay mismatch at {turn}")
                    else:
                        raise RuntimeError(f"Valid query was skipped at {turn}")
                if history[turn-1]["query"] or delta["nodes"] or delta["edges"]:
                    raise RuntimeError(f"Rejected query contributed graph data at {turn}")
            else:
                if mode == "exploit" and prompt_config["memory_profile"] == "recent_exclusion_desc":
                    from extraction.query_anchor import canonicalize_anchor
                    canonical, normalization = canonicalize_anchor(" ".join(query["raw_query"].strip().split()), anchor, graph.nodes)
                    if canonical != query["query"] or normalization != query.get("anchor_normalization"):
                        raise RuntimeError(f"Anchor normalization replay mismatch at {turn}")
                rendered = render(library, graph, turn, mode, anchor, history[:turn-1],
                    **{"exploit_query" if mode == "exploit" else "explore_query": query["query"]})
                if rendered != history[turn-1]["query"]:
                    raise RuntimeError(f"Victim query does not match query writer at {turn}")
        expected_retrieval = None if query_skipped else retrieval_for_turn(manifest, method, turn, load_profile(PROJECT, manifest.get("initialization_prompt_profile", "agea_adapted_null")),
            generated[turn]["query"] if generated is not None and turn > 1 else None)
        if moderation_skipped:
            stage = "query_generation" if query_skipped else "extraction"
            payload = ({"messages": generated[turn]["messages"], "model": generated[turn]["model"]}
                if query_skipped else {"query": history[turn-1]["query"], "retrieval_query": expected_retrieval})
            failure = moderation.load_failure(out, stage, turn, payload, moderation_attempts)
            if failure is None or recorded_stats != moderation.skip_stats(failure) or delta["nodes"] or delta["edges"]:
                raise RuntimeError(f"Moderation failure evidence mismatch at {turn}")
            if not query_skipped and history[turn-1].get("retrieval_query") != expected_retrieval:
                raise RuntimeError("Moderation retrieval query mismatch")
        if expected_retrieval is not None and not moderation_skipped:
            from extraction.separated_query import PROTOCOL
            context = json.loads((out / "turn_logs/retrieved_contexts" / f"retrieved_context_query_{turn}.json").read_text())
            if (history[turn-1].get("retrieval_query") != expected_retrieval or
                    context.get("retrieval_query") != expected_retrieval or
                    context.get("retrieval_protocol") != PROTOCOL or
                    context.get("generation_query") != history[turn-1]["query"] or
                    context.get("query") != history[turn-1]["query"]):
                raise RuntimeError(f"Pure retrieval/generation separation mismatch at {turn}")
        if history[turn-1].get("parse_stats", {}).get("parse_status") in {"format_error", "unstructured_empty"}:
            raise RuntimeError(f"Failed parser round consumed at {turn}")
        if parser_name == "bnrr" and not query_skipped and not moderation_skipped:
            if manifest.get("extra_llm_filter"):
                raise RuntimeError("BNRR parser replay requires no extra model filter")
            from extraction.bnrr_parser import parse_response, extraction_finish_reasons
            response = (out / "turn_logs/llm_responses" / f"first_llm_response_query_{turn}.txt").read_text()
            context = json.loads((out / "turn_logs/retrieved_contexts" / f"retrieved_context_query_{turn}.json").read_text())
            reasons = context.get("extraction_finish_reasons", [])
            request_dir = root / "seed_generation" / f"seed{seed}" if context.get("source_response_path") else out
            wire_reasons = extraction_finish_reasons(request_dir / "requests.jsonl", 1 if context.get("source_response_path") else turn)
            if reasons != wire_reasons:
                raise RuntimeError(f"Completion metadata mismatch at {turn}")
            parsed_nodes, parsed_edges, parsed_stats = parse_response(response,
                extract_body=backend.extract_actual_llm_response, finish_reasons=reasons)
            if CandidateBatch.from_records(parsed_nodes, parsed_edges).to_records() != (delta["nodes"], delta["edges"]):
                raise RuntimeError(f"Saved response parser replay mismatch at {turn}")
            if any(recorded_stats.get(k) != v for k, v in parsed_stats.items()):
                raise RuntimeError(f"Record-local parser diagnostics mismatch at {turn}")
        elif query_skipped:
            if generated is None or turn == 1:
                raise RuntimeError(f"Missing rejected query provenance at {turn}")
        before_nodes, before_edges = set(graph.nodes), set(graph.edges())
        if parser_name == "bnrr":
            expected = {"round_status": recorded_stats["parse_status"], "skip_reason": recorded_stats.get("skip_reason"),
                "skipped_records": len(recorded_stats.get("skipped_records", []))}
            if any(row.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f"Skipped round accounting mismatch at {turn}")
        batch = CandidateBatch.from_records(delta["nodes"], delta["edges"])
        merge_batch(graph, batch)
        if (manifest.get("history_observation_protocol") == "response-visible-v1" or
                prompt_config["memory_profile"] in ("recent_exclusion_desc",)):
            expected_history = {"anchor": decision["anchor"], "returned_entity_names": sorted(batch.nodes),
                "nodes_added_to_graph": len(set(graph.nodes) - before_nodes),
                "edges_added_to_graph": len(set(graph.edges()) - before_edges),
                "newly_discovered_entity_names": sorted(set(graph.nodes) - before_nodes),
                "query_intent": history[turn-1]["query"].split("\n\n", 1)[0],
                "mode": decision["mode"], "turn": turn}
            if any(history[turn-1].get(key) != value for key, value in expected_history.items()):
                raise RuntimeError(f"Observed query-memory replay mismatch at {turn}")
        metrics = evaluate_recovery(graph, truth)
        if any(not math.isclose(metrics[k], row[k], abs_tol=1e-12) for k in METRICS):
            raise RuntimeError(f"Metric replay mismatch at {turn}")
        if turn > 1:
            epsilon = max(.05, epsilon * .98)
    if prefix is not None:
        requests = read("requests.jsonl")
        if len(requests) < rows[-1]["provider_requests"]:
            raise RuntimeError("Committed request accounting missing")
        if any(r["kind"] == "chat" and (not r["thinking_disabled"] or r["reasoning_content_nonempty"]) for r in requests):
            raise RuntimeError("Thinking contract mismatch")
        if return_state:
            return {"graph": graph, "controller": control, "history": history, "records": rows, "epsilon": epsilon}
        return {"rounds": horizon, "prefix_replay": "pass"}
    final = evaluate_recovery(nx.read_graphml(out / "final_graph.graphml", force_multigraph=True), truth)
    if any(not math.isclose(final[k], rows[-1][k], abs_tol=1e-12) for k in METRICS):
        raise RuntimeError("Final graph mismatch")
    requests = read("requests.jsonl")
    if any(r["kind"] == "chat" and (not r["thinking_disabled"] or r["reasoning_content_nonempty"]) for r in requests):
        raise RuntimeError("Thinking contract mismatch")
    if len(requests) != rows[-1]["provider_requests"]:
        raise RuntimeError("Provider request count mismatch")
    summary = json.loads((out / "summary.json").read_text())
    if parser_name == "bnrr":
        if summary.get("skipped_rounds") != sum(bool(h["parse_stats"].get("round_skipped")) for h in history):
            raise RuntimeError("Skipped round summary mismatch")
    if not math.isclose(summary["J_E"], statistics.mean(r["edge_pair_f1"] for r in rows[1:]), abs_tol=1e-12):
        raise RuntimeError("J_E summary mismatch")
    if any(not math.isclose(summary["final"][k], rows[-1][k], abs_tol=1e-12) for k in METRICS):
        raise RuntimeError("Final summary mismatch")
    return {"method": method, "seed": seed, "rounds": horizon, "metric_replay": "pass", "policy_replay": "pass",
        "request_count": len(requests), "request_errors": rows[-1]["request_errors"],
        "parser_replay": "pass"}


def summarize(root):
    manifest = check_manifest(root)
    result = {}
    audits = []
    for method in manifest["methods"]:
        rows = []
        for seed in manifest["seeds"]:
            out = root / "runs" / f"{method}_seed{seed}"
            status = json.loads((out / "status.json").read_text())
            if status["status"] != "completed" or status["round"] != manifest["horizon"]:
                raise RuntimeError("Incomplete run, refusing terminal comparison")
            audits.append(audit_run(root, method, seed))
            summary = json.loads((out / "summary.json").read_text())
            final = summary["final"]
            # The paper defines PAR = precision * recall in [0, 1]. Compute
            # each seed's value before aggregation, preserving covariance.
            final["node_par"] = final["node_precision"] * final["node_recall"]
            final["edge_pair_par"] = final["edge_pair_precision"] * final["edge_pair_recall"]
            rows.append(summary)
        result[method] = {"seeds": rows, "mean": {k: statistics.mean(r["final"][k] for r in rows) for k in SUMMARY_METRICS},
            "std": {k: statistics.stdev(r["final"][k] for r in rows) if len(rows) > 1 else None for k in SUMMARY_METRICS},
            "J_E_mean": statistics.mean(r["J_E"] for r in rows),
            "J_E_std": statistics.stdev(r["J_E"] for r in rows) if len(rows) > 1 else None}
    write_json(root / "RESULTS.json", result)
    write_json(root / "AUDIT.json", {"status": "passed", "runs": audits})
    description = (f"All entries are {len(manifest['seeds'])}-seed arithmetic means ± sample SD."
        if len(manifest["seeds"]) > 1 else "Single-seed results; sample SD is undefined and stored as null.")
    text = f"# DIVER: {manifest['dataset']}, {manifest['horizon']} rounds\n\n{description} Metrics use the 0–1 scale; PAR = P × R.\n\n| Method | Node P | Node R | Edge P | Edge R | Node F1 | Edge F1 | Node PAR | Edge PAR | J_E |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for method, row in result.items():
        values = [(f"{row['mean'][k]:.4f} ± {row['std'][k]:.4f}" if row["std"][k] is not None else f"{row['mean'][k]:.4f}")
                  for k in SUMMARY_METRICS]
        trajectory = f"{row['J_E_mean']:.4f} ± {row['J_E_std']:.4f}" if row["J_E_std"] is not None else f"{row['J_E_mean']:.4f}"
        text += "| " + " | ".join([method, *values, trajectory]) + " |\n"
    text += "\nDIVER uses null-aware extraction and soft exposure. See manifest.json for the exact experiment configuration.\n"
    (root / "RESULTS.md").write_text(text)


def supervise(root):
    environment()
    manifest = check_manifest(root)
    (root / "logs").mkdir()
    write_json(root / "LAUNCH.json", {"pid": os.getpid(), "started": time.time(), "max_workers": manifest["max_workers"]})
    def queue(jobs):
        active, remaining, completed = [], list(jobs), []
        while remaining or active:
            while remaining and len(active) < manifest["max_workers"]:
                name, args = remaining.pop(0)
                handle = (root / "logs" / f"{name}.log").open("ab")
                p = subprocess.Popen([sys.executable, str(PROJECT / "scripts/run_bnrr.py"), *args, "--root", str(root)],
                    cwd=PROJECT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
                active.append((name, p, handle))
            for name, p, handle in list(active):
                code = p.poll()
                if code is not None:
                    handle.close()
                    active.remove((name, p, handle))
                    completed.append({"name": name, "exit_code": code})
                    if code:
                        # Stop dispatch only. Other already-running workers finish normally.
                        remaining.clear()
            write_json(root / "STATUS.json", {"pid": os.getpid(), "active": [{"name": n, "pid": p.pid} for n, p, _ in active],
                "queued": [n for n, _ in remaining], "finished": completed, "updated": time.time()})
            if active:
                time.sleep(5)
        if any(x["exit_code"] for x in completed):
            raise RuntimeError("A worker failed; further stage dispatch blocked")
    if not manifest.get("reused_seed_source") and not manifest.get("shared_initialization"):
        queue([(f"seed{seed}", ["seed", "--seed", str(seed)]) for seed in manifest["seeds"]])
    # Freeze the independently obtained initial response for each paired seed.
    manifest = json.loads((root / "manifest.json").read_text())
    for seed in manifest["seeds"]:
        name = f"seed_generation/seed{seed}/response.txt"
        manifest["runtime_hashes"][name] = sha(root / name)
        if manifest.get("seed_retrieval_protocol"):
            name = f"seed_generation/seed{seed}/retrieval_query.txt"
            manifest["runtime_hashes"][name] = sha(root / name)
        name = f"seed_generation/seed{seed}/extraction_completion.json"
        if (root / name).exists():
            manifest["runtime_hashes"][name] = sha(root / name)
        name = f"seed_generation/seed{seed}/moderation/extraction_1.json"
        if (root / name).exists():
            manifest["runtime_hashes"][name] = sha(root / name)
    write_json(root / "manifest.json", manifest)
    jobs = [(f"{method}_seed{seed}", ["worker", "--method", method, "--seed", str(seed)]) for seed in manifest["seeds"] for method in manifest["methods"]]
    queue(jobs)
    summarize(root)
    write_json(root / "STATUS.json", {"status": "completed", "pid": os.getpid(), "finished": time.time(), "runs": len(jobs)})
