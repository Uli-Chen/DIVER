"""CLI entry point for GraphRAG extraction experiments."""

from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .pipeline import MedicalExtractionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--run-id", help="Unique directory name under artifacts/runs/mematk")
    parser.add_argument("--turns", type=int, help="Number of GraphRAG query turns")
    parser.add_argument("--random-seed", type=int, help="Controller random seed")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable online Weights & Biases tracking",
    )
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--wandb-entity", help="W&B team or user entity")
    parser.add_argument("--wandb-group", help="Explicit W&B comparison group")
    parser.add_argument("--wandb-job-type", help="W&B job type")
    parser.add_argument(
        "--wandb-tag",
        action="append",
        dest="wandb_tags",
        help="Additional W&B tag; may be repeated",
    )
    parser.add_argument(
        "--wandb-artifact-policy",
        choices=("none", "audit", "full"),
        help="Files mirrored to W&B: none, compact audit set, or full run directory",
    )
    parser.add_argument(
        "--disable-api-thinking",
        action="store_true",
        help="Disable provider reasoning/thinking for GraphRAG answer generation",
    )
    parser.add_argument(
        "--graphrag-query-retries",
        type=int,
        help="Retries after a failed or empty GraphRAG extraction response",
    )
    parser.add_argument(
        "--enable-graph-filter",
        action="store_true",
        help="Enable AGEA's optional LLM graph filter (disabled by default)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.load(args.config)
    if args.run_id:
        config.run_id = args.run_id
    if args.turns is not None:
        config.turns = args.turns
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    if args.wandb is not None:
        config.wandb_enabled = args.wandb
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.wandb_group:
        config.wandb_group = args.wandb_group
    if args.wandb_job_type:
        config.wandb_job_type = args.wandb_job_type
    if args.wandb_tags:
        config.wandb_tags = args.wandb_tags
    if args.wandb_artifact_policy:
        config.wandb_artifact_policy = args.wandb_artifact_policy
    if args.disable_api_thinking:
        config.disable_api_thinking = True
    if args.graphrag_query_retries is not None:
        config.graphrag_query_retries = args.graphrag_query_retries
    if args.enable_graph_filter:
        config.enable_graph_filter = True

    summary = MedicalExtractionPipeline(config).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
