"""Provider usage accounting for the official BNRR runtime."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from extraction.request_audit import RequestAudit, cost_summary

from extraction import experiment as pilot
original_environment, original_prepare = pilot.environment, pilot.prepare
original_check, original_summarize = pilot.check_manifest, pilot.summarize
PROTOCOL = "provider-usage-json-sse-v1"


def environment():
    original_environment()


def prepare(root, **kwargs):
    original_prepare(root, **kwargs)
    path = Path(root) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["accounting_protocol"] = PROTOCOL
    manifest["query_max_tokens"] = 1024
    manifest["stream_include_usage"] = True
    manifest["runner_entrypoint"] = "scripts/run_bnrr.py"
    pilot.write_json(path, manifest)


def check_manifest(root):
    manifest = original_check(root)
    if manifest.get("accounting_protocol") != PROTOCOL or manifest.get("query_max_tokens") != 1024:
        raise RuntimeError("Metered runner requires a new metered manifest; cannot resume historical cohorts")
    return manifest


def summarize(root):
    original_summarize(root)
    manifest = check_manifest(root)
    costs = {}
    for method in manifest["methods"]:
        for seed in manifest["seeds"]:
            run = Path(root) / "runs" / f"{method}_seed{seed}"
            rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
            seed_root = Path(manifest["reused_seed_source"]) if manifest.get("reused_seed_source") else Path(root)
            seed_path = seed_root / "seed_generation" / f"seed{seed}" / "requests.jsonl"
            seed_rows = [json.loads(line) for line in seed_path.read_text().splitlines()] if seed_path.exists() else []
            metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            costs[f"{method}_seed{seed}"] = {
                "postseed": cost_summary(rows), "initialization": cost_summary(seed_rows) if seed_rows else None,
                "initialization_charged_once_per_method": True,
                "including_initialization": cost_summary(seed_rows + rows) if seed_rows else None,
                "quality_vs_cost": [{"turn": row["turn"], **{k: row[k] for k in pilot.METRICS},
                    "cumulative_cost": cost_summary(seed_rows + [r for r in rows if r["turn"] <= row["turn"]])
                        if seed_rows else None} for row in metrics]}
    totals = {}
    for method in manifest["methods"]:
        values = [costs[f"{method}_seed{seed}"]["including_initialization"] for seed in manifest["seeds"]]
        complete = all(v is not None and v["chat_tokens_complete"] for v in values)
        tokens = [v["chat_total_tokens"] for v in values] if complete else []
        totals[method] = {"chat_tokens_complete_all_seeds": complete,
            "chat_total_tokens_mean": statistics.mean(tokens) if tokens else None,
            "chat_total_tokens_sample_std": statistics.stdev(tokens) if len(tokens) > 1 else None}
    pilot.write_json(Path(root) / "EFFICIENCY.json", {"runs": costs, "methods": totals,
        "note": "Actual provider usage; unknown tokens remain null. Embeddings are separate. No monetary-price assumptions."})


pilot.environment, pilot.prepare = environment, prepare
pilot.check_manifest, pilot.summarize = check_manifest, summarize
pilot.RequestAudit = RequestAudit
# The legacy supervisor uses its module __file__ to dispatch child workers.
pilot.__file__ = str(ROOT / "scripts/run_bnrr.py")
