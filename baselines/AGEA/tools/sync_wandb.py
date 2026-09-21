#!/usr/bin/env python3
"""Mirror a completed standalone AGEA run into the MemATK W&B project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from extraction.wandb_tracking import WandbExperimentTracker, WandbOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--project", default="mematk")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--job-type", default="formal")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--artifact-policy", choices=("none", "audit", "full"), default="audit"
    )
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def agea_turn_record(
    history: dict[str, Any], evaluation: dict[str, Any], turn: int
) -> dict[str, Any]:
    cumulative = history.get("cumulative_metrics", {})
    new_nodes = int(history.get("nodes_added_to_graph", 0) or 0)
    new_edges = int(history.get("edges_added_to_graph", 0) or 0)
    novelty = safe_float(history.get("novelty", 0.0))
    anchor = next(iter(history.get("seeds_used", [])), None)
    return {
        "turn": turn,
        "mode": str(history.get("mode", "unknown")),
        "decision_reason": str(
            history.get("decision_reason")
            or history.get("details", {}).get("reason", "standalone_agea")
        ),
        "anchor": anchor,
        "action": None,
        "mode_decision": history.get("mode_decision", {}),
        "arm_decision": {"eligible_arm_count": len(history.get("seeds_used", []))},
        "raw_batch_reward": novelty,
        "reward_attribution_warning": False,
        "query_generation": history.get("query_generation", {}),
        "anchor_diagnostics": {},
        "action_metrics": {
            "action_kind": "agea",
            "new_nodes": new_nodes,
            "new_directed_pairs": new_edges,
            "evaluation_only": {},
        },
        "gain": {
            "graph": bool(new_nodes or new_edges),
            "relation": bool(new_edges),
            "meaningful": bool(new_nodes or new_edges),
        },
        "parser": {},
        "batch": {
            "new_nodes": new_nodes,
            "new_edges": new_edges,
            "count_novelty": novelty,
            "total_nodes_in_graph": int(history.get("total_nodes_in_graph", 0) or 0),
            "total_edges_in_graph": int(history.get("total_edges_in_graph", 0) or 0),
            "native_node_precision": safe_float(cumulative.get("precision_nodes"))
            / 100.0,
            "native_node_recall": safe_float(cumulative.get("leakage_rate_nodes"))
            / 100.0,
            "native_edge_precision": safe_float(cumulative.get("precision_edges"))
            / 100.0,
            "native_edge_recall": safe_float(cumulative.get("leakage_rate_edges"))
            / 100.0,
        },
        "evaluation": evaluation,
    }


def query_diagnostics(history: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(history)
    zero_gain = sum(
        not bool(
            record.get("nodes_added_to_graph", 0)
            or record.get("edges_added_to_graph", 0)
        )
        for record in history
    )
    return {
        "total_queries": total,
        "zero_gain_turns": zero_gain,
        "zero_gain_rate": zero_gain / total if total else 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    evaluation_path = args.evaluation.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    history = load_json(run_dir / "query_history.json")
    evaluated = load_json(evaluation_path)
    turn_metrics = evaluated.get("turn_metrics", [])
    if len(history) != len(turn_metrics):
        raise ValueError(
            "AGEA query history and uniform evaluation have different turn counts: "
            f"{len(history)} != {len(turn_metrics)}"
        )
    if not history:
        raise ValueError("AGEA run has no completed turns")

    run_id = run_dir.name
    experiment_config = {
        **config,
        "run_id": run_id,
        "protocol_run_id": config.get("run_id"),
        "controller_name": "standalone_agea",
        "random_seed": config.get("random_seed", "upstream-default"),
        "wandb_project": args.project,
        "wandb_entity": args.entity,
        "wandb_group": args.group,
        "wandb_job_type": args.job_type,
        "wandb_tags": args.tag,
        "wandb_artifact_policy": args.artifact_policy,
    }
    summary = {
        "dataset": config.get("dataset", "unknown"),
        "run_id": run_id,
        "turns_completed": len(history),
        "controller_name": "standalone_agea",
        "tri_action_variant": None,
        "query_diagnostics": query_diagnostics(history),
        "trajectory": evaluated.get("trajectory", {}),
        "final": evaluated.get("final", turn_metrics[-1]),
    }
    write_json(run_dir / "wandb_config.json", experiment_config)
    write_json(run_dir / "wandb_summary.json", summary)

    tracker = WandbExperimentTracker(
        options=WandbOptions(
            enabled=True,
            project=args.project,
            entity=args.entity,
            group=args.group,
            job_type=args.job_type,
            tags=(
                "baseline",
                "standalone-agea",
                str(config.get("dataset", "unknown")),
                *args.tag,
            ),
            notes="Post-run mirror of the vendored standalone AGEA baseline",
            artifact_policy=args.artifact_policy,
        ),
        experiment_config=experiment_config,
        run_dir=run_dir,
    )
    exit_code = 1
    try:
        tracker.start()
        for turn, (native, uniform) in enumerate(
            zip(history, turn_metrics, strict=True), start=1
        ):
            tracker.log_turn(agea_turn_record(native, uniform, turn))
        tracker.complete(summary)
        exit_code = 0
    except BaseException as error:
        tracker.fail(error)
        raise
    finally:
        tracker.finish(exit_code)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
