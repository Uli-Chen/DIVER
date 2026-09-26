#!/usr/bin/env python3
"""Prepare, run, resume, and evaluate DIVER experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from extraction import accounting as metered
from extraction.bnrr_config import DEFAULTS

PROJECT = Path(__file__).resolve().parents[2]
ENTRY = PROJECT / "scripts/run_bnrr.py"
PROFILE = DEFAULTS.prompt_profile
SEEDS = DEFAULTS.seeds
DATASETS = ("novel", "agriculture", "medical")
base_check = metered.check_manifest

def validate_seeds(seeds):
    if not isinstance(seeds, (list, tuple)) or len(seeds) not in (1, 2, 3) or len(set(seeds)) != len(seeds) or any(s not in SEEDS for s in seeds):
        raise ValueError("Select one to three distinct seeds from 42, 43, 44")
    return list(seeds)


def validate_rounds(rounds):
    if type(rounds) is not int or rounds < 2:
        raise ValueError("Round budget must be an integer of at least 2, including initialization")
    return rounds


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    metered.pilot.write_json(path, value)


def check(root):
    root = Path(root)
    manifest = base_check(root)
    seeds = validate_seeds(manifest.get("seeds"))
    rounds = validate_rounds(manifest.get("horizon"))
    expected = {"horizon": rounds, "methods": ["FULL"], "max_workers": len(seeds),
        "parser_by_method": {"FULL": "bnrr"}, "bnrr_prompt_profile": PROFILE,
        "initialization_prompt_profile": PROFILE, "query_memory_profile": "recent_exclusion_desc",
        "coverage_protocol": "bnrr-soft-self-onehop", "neighbor_penalty": .5,
        "moderation_policy": DEFAULTS.moderation_policy, "moderation_attempts": DEFAULTS.moderation_attempts,
        "bnrr_gate_policy": DEFAULTS.gate_policy, "q_hi": DEFAULTS.q_hi, "q_lo": DEFAULTS.q_lo, "rho": DEFAULTS.rho, "reused_seed_source": None,
        "extraction_max_tokens": 16384, "thinking": False, "extra_llm_filter": False}
    if manifest.get("dataset") not in DATASETS or root.name != f"{manifest['dataset']}_{rounds}r":
        raise RuntimeError("Stage name must match the supported dataset and frozen round budget")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError("Formal protocol mismatch: " + key)
    from extraction.bnrr_prompt_profiles import load_profile
    from extraction.bnrr_parser import parse_response
    from extraction.control.bnrr import BnrrController
    import networkx as nx
    library = load_profile(PROJECT, PROFILE)
    if any(path.parent.name != "bnrr" for path in library.paths.values()):
        raise RuntimeError("Mixed prompt families")
    contract = library.templates["output_contract"]
    if "unquoted lowercase literal null" not in contract or "do not fill missing fields with placeholders" in contract.lower():
        raise RuntimeError("Missing/conflicting null contract")
    graph = nx.MultiDiGraph([("A", "B")])
    controller = metered.pilot.make_controller(manifest, 42)
    if not isinstance(controller, BnrrController):
        raise RuntimeError("Soft controller not selected")
    rendered = {
        "seed": metered.pilot.render(library, graph, 1, "explore", None, []),
        "explore": metered.pilot.render(library, graph, 2, "explore", None, [], explore_query="Which other subjects are discussed?"),
        "exploit": metered.pilot.render(library, graph, 2, "exploit", "A", [], exploit_query="Which relationships involve A?")}
    if any(text.split("\n\n", 1)[1] != contract for text in rendered.values()):
        raise RuntimeError("Initialization/explore/exploit output contracts differ")
    nodes, edges, stats = parse_response("ENTITY: A\nDescription: null\nSource: A\nTarget: B\nDescription: null\nSource: null\nTarget: B\nDescription: null")
    if len(nodes) != 1 or len(edges) != 1 or nodes[0]["description"] is not None or edges[0]["description"] is not None or stats["relationship_records_skipped"] != 1:
        raise RuntimeError("Null parser contract failed")
    import yaml
    settings = yaml.safe_load((root / "graph_root/settings.yaml").read_text())
    if settings["models"]["default_chat_model"]["max_tokens"] != 16384:
        raise RuntimeError("Extraction output limit differs from manifest")
    return manifest, rendered


def prepare(root, dataset, seeds=SEEDS, rounds=DEFAULTS.rounds):
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset")
    seeds = validate_seeds(seeds)
    rounds = validate_rounds(rounds)
    metered.prepare(root, dataset=dataset, horizon=rounds, methods=("FULL",),
        gate_policy="residual_mass", prompt_profile=PROFILE, soft_bnrr=True)
    manifest = read(root / "manifest.json")
    manifest.update(seeds=seeds, max_workers=len(seeds), extraction_max_tokens=16384, runner_entrypoint="scripts/" + ENTRY.name,
        null_contract="literal-null-v1", initialization_counts_as_round=1,
        prompt_revision="agea-adapted-null")
    write(root / "manifest.json", manifest)
    _, rendered = check(root)
    write(root / "PROMPT_PREFLIGHT.json", {"status": "passed", "rendered_generation_queries": rendered,
        "prompt_hashes": {name: manifest["code_hashes"][name] for name in manifest["prompt_files"].values()},
        "seed_and_postseed_contract_identical": True, "all_seven_prompt_files_from_selected_profile": True,
        "null_parser_contract": "passed", "soft_controller": "verified", "api_calls": 0})


def availability(cohort, reuse_response=False):
    """One shared synthetic format probe and one embedding probe, separately billed."""
    from openai import OpenAI
    from extraction.backends.graphrag import _RUNNER
    from extraction.bnrr_prompt_profiles import load_profile
    from extraction.bnrr_parser import parse_response
    from extraction.request_audit import RequestAudit, cost_summary
    from extraction.resume import restore_accounting
    metered.environment()
    out = cohort / "availability"
    out.mkdir(exist_ok=reuse_response)
    audit = RequestAudit(out / "requests.jsonl")
    if reuse_response:
        restore_accounting(audit)
    audit.install()
    model = os.environ["PROVIDER_CHAT_MODEL"]
    prompt = ("Evidence: ENTITY ALPHA has no supplied description. ENTITY BETA has no supplied description. "
        "A directed relationship ALPHA to BETA is supplied, with no relationship description.\n\n" +
        load_profile(PROJECT, PROFILE).templates["output_contract"])
    if reuse_response:
        from urllib.parse import urlsplit
        saved = [json.loads(line) for line in audit.path.read_text().splitlines()]
        if ((out / "prompt.txt").read_text() != prompt or not saved or any(r["kind"] != "chat" for r in saved)
                or saved[-1]["status"] != 200 or saved[-1]["model"] != model
                or saved[-1]["host"] != urlsplit(os.environ["PROVIDER_API_BASE"]).hostname
                or saved[-1]["finish_reasons"] != ["stop"]):
            raise RuntimeError("Saved probe response does not match this incomplete availability check")
        response = (out / "response.txt").read_text()
        finish_reason = "stop"
    else:
        with OpenAI(api_key=os.environ["PROVIDER_API_KEY"], base_url=os.environ["PROVIDER_API_BASE"], max_retries=2, timeout=60) as client:
            reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=1024, temperature=0, **_RUNNER.agent_completion_options(model))
        choice = reply.choices[0]
        response = choice.message.content or ""
        finish_reason = choice.finish_reason
        (out / "prompt.txt").write_text(prompt)
        (out / "response.txt").write_text(response)
    nodes, edges, stats = parse_response(response, finish_reasons=[finish_reason] if finish_reason else [])
    from extraction.bnrr_parser import FIELD
    description_values = [match[2].strip() for line in response.splitlines()
        if (match := FIELD.match(line.replace("**", ""))) and match[1].lower() == "description"]
    if ({n["label"] for n in nodes} != {"ALPHA", "BETA"} or len(edges) != 1 or
            (edges[0]["source"], edges[0]["target"]) != ("ALPHA", "BETA") or
            any(n["description"] is not None for n in nodes) or edges[0]["description"] is not None or
            len(description_values) < 3 or any(value != "null" for value in description_values) or finish_reason != "stop"):
        raise RuntimeError("Live model null-format probe failed; inspect saved response")
    with OpenAI(api_key=os.environ["PROVIDER_EMBEDDING_API_KEY"], base_url=os.environ["PROVIDER_EMBEDDING_API_BASE"], max_retries=2, timeout=60) as client:
        reply = client.embeddings.create(model=os.environ["PROVIDER_EMBEDDING_MODEL"], input="Availability check", encoding_format="float")
    dimension = len(reply.data[0].embedding)
    if dimension != 4096:
        raise RuntimeError("Embedding dimension mismatch")
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    if any(r["kind"] == "chat" and (not r["thinking_disabled"] or r["reasoning_content_nonempty"]) for r in rows):
        raise RuntimeError("Availability probe violated no-thinking contract")
    write(out / "RESULT.json", {"status": "passed", "chat_model": model,
        "chat_api_base": os.environ["PROVIDER_API_BASE"], "thinking_disabled": True, "embedding_dimensions": dimension,
        "parse_stats": stats, "costs": cost_summary(rows), "scope": "synthetic availability probes; excluded from experiment metrics"})
    print("Live availability passed: null format, chat endpoint, 4096-dimensional embedding", flush=True)


def progress(root):
    result = []
    for seed in read(root / "manifest.json")["seeds"]:
        run = root / "runs" / f"FULL_seed{seed}"
        source = run if run.exists() else root / "seed_generation" / f"seed{seed}"
        status_path = source / "status.json"
        status = read(status_path) if status_path.exists() else {}
        commit = run / "COMMIT.json"
        round_no = read(commit)["round"] if commit.exists() else status.get("round", 0) if run.exists() else 0
        result.append({"seed": seed, "round": round_no, "status": status.get("status", "initializing"),
            "request_errors": status.get("request_errors", 0)})
    return result


def run(root):
    """Run one prepared experiment in the foreground."""
    manifest, _ = check(root)
    if (root / "LAUNCH.json").exists() or (root / "runs").exists():
        raise RuntimeError("This stage has already started; resume an interrupted worker with worker --resume")
    log = root / "supervisor.log"
    print(f"DIVER | {manifest['dataset']} | seeds {manifest['seeds']} | {manifest['horizon']} rounds", flush=True)
    print(f"Logs/results: {root}", flush=True)
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen([sys.executable, "-u", str(ENTRY), "supervise", "--root", str(root)],
            cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            current = progress(root)
            write(root / "PROGRESS.json", {"updated": time.time(), "seeds": current})
            print(time.strftime("%H:%M:%S"), " | ".join(f"seed{x['seed']} {x['round']}/{manifest['horizon']} {x['status']} errors={x['request_errors']}" for x in current), flush=True)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
    code = process.returncode
    terminal = {"status": "completed" if code == 0 else "failed", "exit_code": code,
        "finished": time.time(), "seeds": progress(root),
        "results": str(root / "RESULTS.md"), "diagnostics": str(log)}
    write(root / "TERMINAL.json", terminal)
    print(json.dumps(terminal), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "run", "availability", "supervise", "seed", "worker", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS), help="Prepare: one to three seeds")
    parser.add_argument("--rounds", type=int, default=DEFAULTS.rounds, help="Prepare: scheduled rounds including initialization (default: 100)")
    parser.add_argument("--method", default="FULL", choices=("FULL",))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-probe-response", action="store_true", help="Availability only: validate the saved successful chat response and finish its embedding check")
    args = parser.parse_args()
    if args.action == "prepare" and args.dataset is None:
        parser.error("prepare requires --dataset")
    if args.action in ("seed", "worker") and args.seed is None:
        parser.error(args.action + " requires --seed")
    if args.resume and args.action != "worker":
        parser.error("--resume applies to worker only")
    root = args.root.resolve()
    try:
        dataset = args.dataset if args.action == "prepare" else read(root / "manifest.json")["dataset"] if (root / "manifest.json").exists() else args.dataset or "novel"
        if args.action not in ("check", "summarize"):
            metered.environment()
        metered.pilot.check_manifest = lambda path: check(path)[0]
        if args.action in ("seed", "worker") and args.seed not in read(root / "manifest.json")["seeds"]:
            raise ValueError("Seed is not included in this frozen stage")
        if args.action == "prepare": prepare(root, args.dataset, args.seeds, rounds=args.rounds)
        elif args.action == "availability": availability(root, reuse_response=args.reuse_probe_response)
        elif args.action == "check":
            manifest, _ = check(root)
            print(json.dumps({"status": "passed", "dataset": manifest["dataset"], "profile": PROFILE}))
        elif args.action == "run": return run(root)
        elif args.action == "supervise": metered.pilot.supervise(root)
        elif args.action == "summarize": metered.summarize(root)
        else: metered.pilot.worker(root, args.method, args.seed, seed_only=args.action == "seed", resume=args.resume)
    except Exception as error:
        # Store failure kind without exposing provider error payloads or secrets.
        out = root / (Path("seed_generation") / f"seed{args.seed}" if args.action == "seed" else Path("runs") / f"FULL_seed{args.seed}" if args.action == "worker" else Path("."))
        write(out / "FAILURE.json", {"action": args.action, "error_type": type(error).__name__, "time": time.time()})
        if args.action in ("seed", "worker"):
            previous = read(out / "status.json") if (out / "status.json").exists() else {"round": 0}
            write(out / "status.json", {**previous, "status": "failed", "updated": time.time(), "pid": os.getpid()})
        details = traceback.format_exc()
        for name, value in os.environ.items():
            if ("key" in name.lower() or "token" in name.lower()) and len(value) > 8:
                details = details.replace(value, "[REDACTED]")
        print(details, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
