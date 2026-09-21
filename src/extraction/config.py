"""Configuration loading for reproducible extraction experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


EXTRACTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXTRACTION_DIR.parents[1]


@dataclass
class ExperimentConfig:
    dataset: str = "medical"
    graph_root: str = str(PROJECT_DIR / "artifacts" / "graphrag" / "medical")
    data_dir: str = str(PROJECT_DIR / "artifacts" / "graphrag" / "medical" / "output")
    output_root: str = str(PROJECT_DIR / "artifacts" / "runs" / "mematk")
    run_id: str = "medical_uniform_fresh_50turn"
    wandb_enabled: bool = False
    wandb_project: str = "mematk"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "extraction"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_artifact_policy: str = "audit"
    turns: int = 50
    query_method: str = "local"
    disable_api_thinking: bool = False
    graphrag_query_retries: int = 2
    shared_seed_response_path: str | None = None
    enable_graph_filter: bool = False
    graph_filter_model: str = "gpt-4o-mini"
    initial_epsilon: float = 0.30
    epsilon_decay: float = 0.98
    min_epsilon: float = 0.05
    htsn_threshold: float = 0.15
    htsn_window: int = 5
    adaptive_htsn_threshold: bool = True
    mode_feedback_signal: str = "htsn_combined"
    explore_success_window: int = 20
    explore_success_min_samples: int = 5
    explore_success_threshold: float = 0.20
    max_consecutive_failed_explore: int = 2
    mode_sampling_policy: str = "stochastic_epsilon"
    deterministic_explore_surplus_cap: float = 0.0
    random_seed: int = 42
    controller_name: str = "legacy_anchor"
    tri_action_variant: str = "bnrr_closure"
    prompt_dir: str = str(PROJECT_DIR / "baselines" / "_shared" / "legacy_prompts" / "triaction")
    deterministic_action_prompts: bool = False
    closure_bundle_size: int = 1
    closure_bnrr_percentile: float = 0.50
    prevent_consecutive_closure: bool = True
    closure_pair_policy: str = "uniform"
    closure_min_shared_text_units: int = 0
    max_closure_local_share: float = 0.50
    fewa_delta: float = 0.05
    fewa_max_arms: int = 20
    anchor_sampling_policy: str = "uniform_fresh"
    bnrr_schedule_profile: str = "linear"
    bnrr_schedule_start_fraction: float = 0.20
    bnrr_schedule_end_fraction: float = 0.70
    global_bnrr_initial_pool_fraction: float = 0.10
    global_bnrr_relax_exploit_fraction: float = 0.50
    reward_normalizer: int = 512
    query_generator_model: str = "gpt-4o-mini"
    query_generation_retries: int = 3
    query_similarity_threshold: float = 0.85
    require_exploit_anchor_in_query: bool = True
    acceptance_min_query_unique_rate: float = 0.90
    acceptance_max_zero_gain_rate: float = 0.20
    acceptance_max_consecutive_zero_gain: int = 4
    acceptance_min_anchor_query_adherence: float = 1.0
    acceptance_max_consecutive_meaningful_zero_gain: int = 4
    acceptance_require_both_post_seed_modes: bool = True
    seed_query: str = (
        "Identify major diseases, treatments, diagnostic tests, drugs, symptoms, "
        "risk factors, and care organizations in the medical corpus, and explain "
        "their concrete relationships."
    )
    exploration_queries: list[str] = field(
        default_factory=lambda: [
            (
                "Identify diseases and their treatments, complications, and care "
                "teams in the medical corpus."
            ),
            (
                "Identify diagnostic tests, biomarkers, symptoms, and the conditions "
                "they diagnose in the medical corpus."
            ),
            (
                "Identify drugs and therapies, including indications, side effects, "
                "contraindications, and alternatives."
            ),
            (
                "Identify anatomy, genes, risk factors, and prevention or screening "
                "relationships in the medical corpus."
            ),
            (
                "Identify clinical organizations, professional roles, patient "
                "resources, and their care relationships."
            ),
        ]
    )
    @classmethod
    def load(cls, path: str | Path | None = None) -> "ExperimentConfig":
        config = cls()
        if path is None:
            return config
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, Mapping):
            raise ValueError("Experiment configuration must be a YAML mapping.")
        unknown = set(values) - set(asdict(config))
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for key, value in values.items():
            setattr(config, key, value)
        return config

    def validate(self) -> None:
        if self.dataset not in {"medical", "novel", "agriculture"}:
            raise ValueError(
                "This runner currently supports the medical, novel, and agriculture "
                "GraphRAG datasets."
            )
        if self.turns <= 0:
            raise ValueError("turns must be positive")
        if self.wandb_enabled and not self.wandb_project.strip():
            raise ValueError("wandb_project must be non-empty when W&B is enabled")
        if not self.wandb_job_type.strip():
            raise ValueError("wandb_job_type must be non-empty")
        if self.wandb_artifact_policy not in {"none", "audit", "full"}:
            raise ValueError(
                "wandb_artifact_policy must be one of ['audit', 'full', 'none']"
            )
        if not isinstance(self.wandb_tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in self.wandb_tags
        ):
            raise ValueError("wandb_tags must be a list of non-empty strings")
        if not 0.0 <= self.htsn_threshold <= 1.0:
            raise ValueError("htsn_threshold must be in [0, 1]")
        if self.mode_feedback_signal not in {"htsn_combined", "count_novelty"}:
            raise ValueError(
                "mode_feedback_signal must be one of "
                "['count_novelty', 'htsn_combined']"
            )
        if not 0.0 <= self.initial_epsilon <= 1.0:
            raise ValueError("initial_epsilon must be in [0, 1]")
        if not 0.0 <= self.min_epsilon <= 1.0:
            raise ValueError("min_epsilon must be in [0, 1]")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if self.reward_normalizer <= 0:
            raise ValueError("reward_normalizer must be positive")
        for name in (
            "explore_success_threshold",
            "query_similarity_threshold",
            "acceptance_min_query_unique_rate",
            "acceptance_max_zero_gain_rate",
            "acceptance_min_anchor_query_adherence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "explore_success_window",
            "explore_success_min_samples",
            "max_consecutive_failed_explore",
            "graphrag_query_retries",
            "query_generation_retries",
            "acceptance_max_consecutive_zero_gain",
            "acceptance_max_consecutive_meaningful_zero_gain",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.fewa_max_arms <= 0:
            raise ValueError("fewa_max_arms must be positive")
        if self.bnrr_schedule_profile not in {
            "hard_switch",
            "linear",
            "staged",
        }:
            raise ValueError(
                "bnrr_schedule_profile must be one of "
                "['hard_switch', 'linear', 'staged']"
            )
        if not (
            0.0
            <= self.bnrr_schedule_start_fraction
            < self.bnrr_schedule_end_fraction
            <= 1.0
        ):
            raise ValueError(
                "BNRR schedule fractions must satisfy "
                "0 <= start < end <= 1"
            )
        if not 0.0 < self.global_bnrr_initial_pool_fraction <= 1.0:
            raise ValueError(
                "global_bnrr_initial_pool_fraction must be in (0, 1]"
            )
        if not 0.0 < self.global_bnrr_relax_exploit_fraction <= 1.0:
            raise ValueError(
                "global_bnrr_relax_exploit_fraction must be in (0, 1]"
            )
        valid_mode_sampling_policies = {
            "deterministic_epsilon_deficit",
            "stochastic_epsilon",
        }
        if self.mode_sampling_policy not in valid_mode_sampling_policies:
            raise ValueError(
                "mode_sampling_policy must be one of "
                f"{sorted(valid_mode_sampling_policies)}"
            )
        if self.deterministic_explore_surplus_cap < 0.0:
            raise ValueError("deterministic_explore_surplus_cap must be non-negative")
        valid_controllers = {"legacy_anchor", "learning_free_tri_action"}
        if self.controller_name not in valid_controllers:
            raise ValueError(
                f"controller_name must be one of {sorted(valid_controllers)}"
            )
        valid_tri_action_variants = {
            "incident_only",
            "uniform_closure",
            "bnrr_closure",
            "raw_bnrr_greedy",
        }
        if self.tri_action_variant not in valid_tri_action_variants:
            raise ValueError(
                "tri_action_variant must be one of "
                f"{sorted(valid_tri_action_variants)}"
            )
        if self.closure_bundle_size <= 0:
            raise ValueError("closure_bundle_size must be positive")
        if not 0.0 <= self.closure_bnrr_percentile <= 1.0:
            raise ValueError("closure_bnrr_percentile must be in [0, 1]")
        if self.closure_pair_policy not in {"uniform", "text_unit_overlap"}:
            raise ValueError(
                "closure_pair_policy must be one of "
                "['text_unit_overlap', 'uniform']"
            )
        if self.closure_min_shared_text_units < 0:
            raise ValueError("closure_min_shared_text_units must be non-negative")
        if not 0.0 <= self.max_closure_local_share <= 1.0:
            raise ValueError("max_closure_local_share must be in [0, 1]")
        prompt_path = Path(self.prompt_dir)
        if not prompt_path.is_absolute():
            prompt_path = PROJECT_DIR / prompt_path
        self.prompt_dir = str(prompt_path.resolve())
        if self.controller_name == "learning_free_tri_action":
            if not self.deterministic_action_prompts:
                raise ValueError(
                    "learning_free_tri_action requires deterministic_action_prompts=true"
                )
            if not Path(self.prompt_dir).is_dir():
                raise FileNotFoundError(self.prompt_dir)
        valid_policies = {
            "degree_conditioned_bnrr_high_fresh",
            "degree_conditioned_bnrr_high_then_degree_frontier_fresh",
            "degree_conditioned_bnrr_low_fresh",
            "degree_conditioned_bnrr_scheduled_fresh",
            "degree_conditioned_shuffled_bnrr_high_fresh",
            "degree_conditioned_shuffled_bnrr_low_fresh",
            "degree_conditioned_shuffled_bnrr_scheduled_fresh",
            "degree_conditioned_uniform_fresh",
            "frontier_alternating_fresh",
            "global_bnrr_normalized_annealed_fresh",
            "global_bnrr_raw_annealed_fresh",
            "global_shuffled_bnrr_raw_annealed_fresh",
            "minimum_degree_alternating_fresh",
            "agea_hub",
            "open_ego_alternating_fresh",
            "structural_opportunity_fresh",
            "supported_open_ego_alternating_fresh",
            "ts_pl_fewa",
            "uniform_fresh",
        }
        if self.anchor_sampling_policy not in valid_policies:
            raise ValueError(
                "anchor_sampling_policy must be one of "
                f"{sorted(valid_policies)}"
            )
        for required in (self.graph_root, self.data_dir):
            if not Path(required).exists():
                raise FileNotFoundError(required)
        if self.shared_seed_response_path and not Path(
            self.shared_seed_response_path
        ).exists():
            raise FileNotFoundError(self.shared_seed_response_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
