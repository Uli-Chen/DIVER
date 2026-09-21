"""Weights & Biases mapping for auditable extraction experiments.

The local run directory remains the authoritative experiment record.  This
module mirrors scalar telemetry, comparison tables, and selected immutable run
artifacts to W&B without exposing evaluator-only values to the controller.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ExperimentConfig


AUDIT_ARTIFACT_FILES = (
    "wandb_run.json",
    "config.json",
    "summary.json",
    "wandb_summary.json",
    "wandb_config.json",
    "EXPERIMENT_RESULTS.md",
    "turn_metrics.csv",
    "decision_state.jsonl",
    "evaluation_metrics.jsonl",
    "controller_state.json",
    "extracted_graph.graphml",
    "extraction_analysis.json",
    "uniform_evaluation.json",
)


@dataclass(frozen=True)
class WandbOptions:
    enabled: bool
    project: str
    entity: str | None
    group: str | None
    job_type: str
    tags: tuple[str, ...]
    notes: str | None
    artifact_policy: str

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> WandbOptions:
        return cls(
            enabled=bool(config.wandb_enabled),
            project=str(config.wandb_project),
            entity=config.wandb_entity,
            group=config.wandb_group,
            job_type=str(config.wandb_job_type),
            tags=tuple(str(tag) for tag in config.wandb_tags),
            notes=config.wandb_notes,
            artifact_policy=str(config.wandb_artifact_policy),
        )


class WandbExperimentTracker:
    """Mirror one MemATK run into one W&B Run."""

    def __init__(
        self,
        *,
        options: WandbOptions,
        experiment_config: Mapping[str, Any],
        run_dir: Path,
        wandb_module: Any | None = None,
    ) -> None:
        self.options = options
        self.experiment_config = dict(experiment_config)
        self.run_dir = run_dir.resolve()
        self._wandb = wandb_module
        self._run: Any | None = None
        self._turn_rows: list[dict[str, Any]] = []

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, run_dir: Path
    ) -> WandbExperimentTracker:
        return cls(
            options=WandbOptions.from_config(config),
            experiment_config=config.to_dict(),
            run_dir=run_dir,
        )

    @property
    def enabled(self) -> bool:
        return self.options.enabled

    def start(self) -> None:
        """Start the online W&B run before any paid experiment calls occur."""

        if not self.enabled:
            return
        if self._wandb is None:
            import wandb

            self._wandb = wandb

        config = dict(self.experiment_config)
        group = self.options.group or _default_group(config)
        tags = _default_tags(config, self.options.tags)
        self._run = self._wandb.init(
            project=self.options.project,
            entity=self.options.entity,
            name=str(config["run_id"]),
            group=group,
            job_type=self.options.job_type,
            tags=tags,
            notes=self.options.notes,
            config=config,
            dir=str(self.run_dir.parent),
            mode="online",
            save_code=True,
            resume="never",
        )
        self._run.define_metric("turn")
        self._run.define_metric("*", step_metric="turn")
        self._run.summary["status"] = "running"
        self._write_run_manifest(status="running", group=group, tags=tags)

    def log_turn(self, record: Mapping[str, Any]) -> None:
        """Log one post-persistence turn using a stable namespace mapping."""

        if self._run is None:
            return
        payload = _turn_payload(record)
        self._run.log(payload, step=int(record["turn"]))
        self._turn_rows.append(payload)

    def complete(self, summary: Mapping[str, Any]) -> None:
        """Publish summary fields, a turn table, and the configured artifact."""

        if self._run is None:
            return
        for key, value in _summary_scalars(summary).items():
            self._run.summary[f"summary/{key}"] = value
        self._run.summary["status"] = "completed"
        self._write_run_manifest(status="completed")

        if self._turn_rows:
            columns = _table_columns(self._turn_rows)
            table = self._wandb.Table(columns=columns)
            for row in self._turn_rows:
                table.add_data(*(row.get(column) for column in columns))
            self._run.log({"turn": len(self._turn_rows), "tables/turn_metrics": table})

        if self.options.artifact_policy != "none":
            artifact_name = f"mematk-run-{_wandb_name(str(summary['run_id']))}"
            artifact = self._wandb.Artifact(
                name=artifact_name,
                type="experiment-run",
                description="Auditable MemATK experiment outputs",
                metadata=_artifact_metadata(summary, self.experiment_config),
            )
            for path, artifact_path in self._artifact_files():
                artifact.add_file(
                    local_path=str(path),
                    name=artifact_path,
                    policy="immutable",
                )
            prompt_dir = Path(str(self.experiment_config.get("prompt_dir", "")))
            if (
                prompt_dir.is_dir()
                and self.experiment_config.get("controller_name")
                == "learning_free_tri_action"
            ):
                artifact.add_dir(local_path=str(prompt_dir), name="prompt_bundle")
            self._run.log_artifact(artifact, aliases=["latest"])

    def fail(self, error: BaseException) -> None:
        if self._run is None:
            return
        self._run.summary["status"] = "failed"
        self._run.summary["error/type"] = type(error).__name__
        self._run.summary["error/message"] = str(error)[:2000]
        self._write_run_manifest(
            status="failed",
            error={"type": type(error).__name__, "message": str(error)[:2000]},
        )

    def finish(self, exit_code: int) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        finally:
            self._run = None

    def _artifact_files(self) -> list[tuple[Path, str]]:
        if self.options.artifact_policy == "full":
            return [
                (path, str(path.relative_to(self.run_dir)))
                for path in sorted(self.run_dir.rglob("*"))
                if path.is_file()
                and "wandb" not in path.relative_to(self.run_dir).parts
            ]
        return [
            (self.run_dir / name, name)
            for name in AUDIT_ARTIFACT_FILES
            if (self.run_dir / name).is_file()
        ]

    def _write_run_manifest(
        self,
        *,
        status: str,
        group: str | None = None,
        tags: Sequence[str] | None = None,
        error: Mapping[str, str] | None = None,
    ) -> None:
        if self._run is None:
            return
        path = self.run_dir / "wandb_run.json"
        previous: dict[str, Any] = {}
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            **previous,
            "status": status,
            "entity": getattr(self._run, "entity", None),
            "project": getattr(self._run, "project", self.options.project),
            "run_id": getattr(self._run, "id", None),
            "run_name": getattr(self._run, "name", None),
            "url": getattr(self._run, "url", None),
            "group": group if group is not None else previous.get("group"),
            "job_type": self.options.job_type,
            "tags": list(tags) if tags is not None else previous.get("tags", []),
            "artifact_policy": self.options.artifact_policy,
        }
        if error is not None:
            payload["error"] = dict(error)
        elif status == "completed":
            payload.pop("error", None)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _turn_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map a nested turn record into chart-friendly W&B namespaces."""

    turn = int(record["turn"])
    mode = str(record.get("mode", "unknown"))
    action = record.get("action") or {}
    action_kind = str(
        action.get("kind")
        or record.get("action_metrics", {}).get("action_kind", "legacy")
    )
    payload: dict[str, Any] = {
        "turn": turn,
        "decision/mode": mode,
        "decision/reason": str(record.get("decision_reason", "")),
        "decision/anchor": record.get("anchor") or "",
        "decision/action_kind": action_kind,
        "decision/is_explore": int(mode == "explore"),
        "decision/is_exploit": int(mode == "exploit"),
        "decision/is_global": int(
            action_kind in {"global", "global_discovery"}
        ),
        "decision/is_incident": int(
            action_kind in {"incident", "incident_expansion"}
        ),
        "decision/is_closure": int(
            action_kind in {"closure", "ego_closure"}
        ),
        "reward/raw_batch": float(record.get("raw_batch_reward", 0.0)),
        "reward/attribution_warning": int(
            bool(record.get("reward_attribution_warning", False))
        ),
    }
    effective_reward = record.get("effective_fewa_reward")
    if isinstance(effective_reward, (int, float)) and not isinstance(
        effective_reward, bool
    ):
        payload["reward/effective_fewa"] = effective_reward

    _merge_numeric(payload, "decision", record.get("mode_decision", {}), max_depth=0)
    _merge_numeric(
        payload, "decision/local", record.get("arm_decision", {}), max_depth=0
    )
    _merge_numeric(
        payload,
        "decision/local/selection",
        record.get("arm_decision", {}).get("selection", {}),
        max_depth=0,
    )
    _merge_numeric(payload, "query", record.get("query_generation", {}), max_depth=0)
    _merge_numeric(payload, "anchor", record.get("anchor_diagnostics", {}), max_depth=0)
    _merge_numeric(
        payload,
        "action",
        {
            key: value
            for key, value in record.get("action_metrics", {}).items()
            if key != "evaluation_only"
        },
        max_depth=0,
    )
    _merge_numeric(payload, "gain", record.get("gain", {}), max_depth=0)
    _merge_numeric(payload, "parser", record.get("parser", {}), max_depth=1)
    _merge_numeric(payload, "cost", record.get("cost", {}), max_depth=0)
    _merge_numeric(payload, "batch", record.get("batch", {}), max_depth=0)
    _merge_numeric(payload, "evaluation", record.get("evaluation", {}), max_depth=0)
    _merge_numeric(
        payload,
        "evaluation/action",
        record.get("action_metrics", {}).get("evaluation_only", {}),
    )
    return payload


def _merge_numeric(
    destination: dict[str, Any],
    prefix: str,
    value: Mapping[str, Any],
    *,
    max_depth: int | None = None,
) -> None:
    for key, item in _flatten_numeric(value, max_depth=max_depth).items():
        destination[f"{prefix}/{key}"] = item


def _flatten_numeric(
    value: Mapping[str, Any],
    prefix: str = "",
    *,
    max_depth: int | None = None,
) -> dict[str, int | float]:
    flattened: dict[str, int | float] = {}
    for key, item in value.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, bool):
            flattened[path] = int(item)
        elif isinstance(item, (int, float)):
            flattened[path] = item
        elif isinstance(item, Mapping) and (max_depth is None or max_depth > 0):
            flattened.update(
                _flatten_numeric(
                    item,
                    path,
                    max_depth=None if max_depth is None else max_depth - 1,
                )
            )
    return flattened


def _flatten_scalars(
    value: Mapping[str, Any],
    prefix: str = "",
    *,
    max_depth: int | None = None,
) -> dict[str, str | int | float | bool | None]:
    flattened: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if item is None or isinstance(item, (str, int, float, bool)):
            flattened[path] = item
        elif isinstance(item, Mapping) and (max_depth is None or max_depth > 0):
            flattened.update(
                _flatten_scalars(
                    item,
                    path,
                    max_depth=None if max_depth is None else max_depth - 1,
                )
            )
    return flattened


def _summary_scalars(
    summary: Mapping[str, Any],
) -> dict[str, str | int | float | bool | None]:
    """Select stable aggregates without creating per-entity W&B keys."""

    flattened = {
        key: value
        for key, value in summary.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    section_depths = {
        "final": 0,
        "trajectory": 0,
        "query_diagnostics": 1,
        "mode_diagnostics": 2,
        "anchor_diagnostics": 1,
        "tri_action_diagnostics": 2,
        "formal_acceptance": 1,
        "rotting_diagnostics": 0,
        "arm_diagnostics": 0,
        "action_evaluation": 1,
    }
    for section, max_depth in section_depths.items():
        value = summary.get(section)
        if not isinstance(value, Mapping):
            continue
        flattened.update(
            {
                f"{section}/{key}": item
                for key, item in _flatten_scalars(value, max_depth=max_depth).items()
            }
        )
    return flattened


def _default_group(config: Mapping[str, Any]) -> str:
    controller = str(config.get("controller_name", "legacy_anchor"))
    return _wandb_name(
        f"{config.get('dataset', 'unknown')}-{controller}-{config.get('turns', 0)}turn"
    )


def _default_tags(
    config: Mapping[str, Any], configured_tags: Sequence[str]
) -> list[str]:
    controller = str(config.get("controller_name", "legacy_anchor"))
    variant = (
        str(config.get("tri_action_variant", "unknown"))
        if controller == "learning_free_tri_action"
        else str(config.get("anchor_sampling_policy", "unknown"))
    )
    tags = [
        str(config.get("dataset", "unknown")),
        controller,
        variant,
        f"seed-{config.get('random_seed', 'unknown')}",
        f"{config.get('turns', 0)}-turn",
        str(config.get("query_method", "unknown")),
        *configured_tags,
    ]
    return list(dict.fromkeys(tag for tag in tags if tag))


def _table_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows))
    priority = [
        "turn",
        "decision/mode",
        "decision/action_kind",
        "decision/reason",
        "decision/anchor",
    ]
    return [key for key in priority if key in keys] + sorted(keys - set(priority))


def _artifact_metadata(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    final = {
        key: value
        for key, value in summary.get("final", {}).items()
        if isinstance(value, (str, int, float, bool)) and _json_number(value)
    }
    return {
        "run_id": summary.get("run_id"),
        "dataset": summary.get("dataset"),
        "turns_completed": summary.get("turns_completed"),
        "controller_name": summary.get("controller_name"),
        "tri_action_variant": summary.get("tri_action_variant"),
        "random_seed": config.get("random_seed"),
        "final": final,
    }


def _json_number(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _wandb_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned[:128] or "run"
