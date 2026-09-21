from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extraction.wandb_tracking import WandbExperimentTracker, WandbOptions


class FakeTable:
    def __init__(self, *, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[tuple[Any, ...]] = []

    def add_data(self, *values: Any) -> None:
        self.rows.append(values)


class FakeArtifact:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.files: list[tuple[str, str, str]] = []
        self.directories: list[tuple[str, str]] = []

    def add_file(self, *, local_path: str, name: str, policy: str) -> None:
        self.files.append((local_path, name, policy))

    def add_dir(self, *, local_path: str, name: str) -> None:
        self.directories.append((local_path, name))


class FakeRun:
    entity = "test-entity"
    project = "mematk"
    id = "run-123"
    name = "medical_l2_seed42"
    url = "https://wandb.example/test-entity/mematk/runs/run-123"

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.defined_metrics: list[tuple[str, dict[str, Any]]] = []
        self.artifacts: list[tuple[FakeArtifact, list[str]]] = []
        self.exit_code: int | None = None

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.defined_metrics.append((name, kwargs))

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((payload, step))

    def log_artifact(self, artifact: FakeArtifact, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class FakeWandb:
    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs: dict[str, Any] = {}

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run


def tracker(tmp_path: Path, fake: FakeWandb) -> WandbExperimentTracker:
    return WandbExperimentTracker(
        options=WandbOptions(
            enabled=True,
            project="mematk",
            entity=None,
            group=None,
            job_type="smoke",
            tags=("m2",),
            notes=None,
            artifact_policy="audit",
        ),
        experiment_config={
            "dataset": "medical",
            "run_id": "medical_l2_seed42",
            "turns": 10,
            "query_method": "local",
            "random_seed": 42,
            "controller_name": "learning_free_tri_action",
            "tri_action_variant": "bnrr_closure",
            "anchor_sampling_policy": "uniform_fresh",
            "prompt_dir": str(tmp_path / "missing-prompts"),
        },
        run_dir=tmp_path,
        wandb_module=fake,
    )


def test_complete_online_mapping(tmp_path: Path) -> None:
    fake = FakeWandb()
    instance = tracker(tmp_path, fake)
    for name in (
        "config.json",
        "summary.json",
        "turn_metrics.csv",
        "decision_state.jsonl",
        "evaluation_metrics.jsonl",
        "extracted_graph.graphml",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    instance.start()
    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["group"] == "medical-learning_free_tri_action-10turn"
    assert "bnrr_closure" in fake.init_kwargs["tags"]

    instance.log_turn(
        {
            "turn": 1,
            "mode": "explore",
            "decision_reason": "seed",
            "anchor": None,
            "action": {"kind": "global"},
            "mode_decision": {"epsilon": 0.3},
            "arm_decision": {
                "eligible_arm_count": 2,
                "selection": {
                    "bnrr_informative": True,
                    "bnrr_policy_tv": 0.75,
                    "paired_bnrr_uniform_disagreement": True,
                    "paired_uniform_arm": "SENSITIVE ENTITY",
                },
            },
            "raw_batch_reward": 0.25,
            "reward_attribution_warning": False,
            "query_generation": {"attempts": 1},
            "anchor_diagnostics": {"anchor_required": False},
            "action_metrics": {
                "new_directed_pairs": 3,
                "validated_pair_list": [["A", "B"]],
                "evaluation_only": {
                    "matched_new_directed_pairs": 2,
                    "residual_yield": {"uu": 1, "ku": 1, "kk": 0},
                },
            },
            "gain": {"meaningful": True},
            "parser": {"candidate_nodes": 4},
            "batch": {
                "new_nodes": 4,
                "htsn_combined": 0.5,
                "new_node_weights": {"SENSITIVE ENTITY": 0.7},
            },
            "evaluation": {"node_f1": 0.4, "edge_pair_f1": 0.2},
        }
    )
    turn_payload, step = fake.run.metrics[-1]
    assert step == 1
    assert turn_payload["decision/is_global"] == 1
    assert turn_payload["batch/new_nodes"] == 4
    assert turn_payload["evaluation/action/matched_new_directed_pairs"] == 2
    assert turn_payload["evaluation/action/residual_yield/uu"] == 1
    assert turn_payload["decision/local/selection/bnrr_informative"] == 1
    assert turn_payload["decision/local/selection/bnrr_policy_tv"] == 0.75
    assert (
        turn_payload[
            "decision/local/selection/paired_bnrr_uniform_disagreement"
        ]
        == 1
    )
    assert not any("paired_uniform_arm" in key for key in turn_payload)
    assert not any("validated_pair_list" in key for key in turn_payload)
    assert not any("SENSITIVE ENTITY" in key for key in turn_payload)

    summary = {
        "dataset": "medical",
        "run_id": "medical_l2_seed42",
        "turns_completed": 1,
        "controller_name": "learning_free_tri_action",
        "tri_action_variant": "bnrr_closure",
        "final": {"node_f1": 0.4, "edge_pair_f1": 0.2},
        "arm_diagnostics": {
            "epochs": 1,
            "active_priors": {"SENSITIVE ENTITY": 0.7},
        },
    }
    instance.complete(summary)
    assert fake.run.summary["summary/final/node_f1"] == 0.4
    assert fake.run.summary["summary/arm_diagnostics/epochs"] == 1
    assert not any("SENSITIVE ENTITY" in key for key in fake.run.summary)
    assert fake.run.summary["status"] == "completed"
    assert any("tables/turn_metrics" in payload for payload, _ in fake.run.metrics)
    artifact, aliases = fake.run.artifacts[0]
    assert aliases == ["latest"]
    assert {name for _, name, _ in artifact.files} >= {
        "wandb_run.json",
        "decision_state.jsonl",
        "evaluation_metrics.jsonl",
        "extracted_graph.graphml",
    }
    manifest = json.loads((tmp_path / "wandb_run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["url"] == FakeRun.url

    instance.finish(0)
    assert fake.run.exit_code == 0


def test_disabled_tracker_is_noop(tmp_path: Path) -> None:
    fake = FakeWandb()
    options = WandbOptions(
        enabled=False,
        project="mematk",
        entity=None,
        group=None,
        job_type="test",
        tags=(),
        notes=None,
        artifact_policy="none",
    )
    instance = WandbExperimentTracker(
        options=options,
        experiment_config={},
        run_dir=tmp_path,
        wandb_module=fake,
    )
    instance.start()
    instance.log_turn({"turn": 1})
    instance.complete({})
    instance.finish(0)
    assert fake.init_kwargs == {}
    assert not (tmp_path / "wandb_run.json").exists()


def test_triaction_enum_values_map_to_wandb_one_hot(tmp_path: Path) -> None:
    fake = FakeWandb()
    instance = tracker(tmp_path, fake)
    instance.start()

    expected = {
        "global_discovery": (1, 0, 0),
        "incident_expansion": (0, 1, 0),
        "ego_closure": (0, 0, 1),
    }
    for turn, (action_kind, one_hot) in enumerate(expected.items(), start=1):
        instance.log_turn(
            {
                "turn": turn,
                "mode": "explore" if turn == 1 else "exploit",
                "action": {"kind": action_kind},
            }
        )
        payload, _ = fake.run.metrics[-1]
        assert (
            payload["decision/is_global"],
            payload["decision/is_incident"],
            payload["decision/is_closure"],
        ) == one_hot
