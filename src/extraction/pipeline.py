"""End-to-end topology-sensitive extraction and graph-truth evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import pandas as pd

from .backends.graphrag import AgeaGraphRagAdapter, append_extraction_command
from .config import ExperimentConfig
from .control.controllers import (
    AdaptiveModeController,
    AgeaHubController,
    DegreeConditionedBnrrFreshController,
    EpochFewaController,
    FreshAnchorController,
    FrontierAlternatingFreshController,
    GlobalBnrrAnnealingFreshController,
    MinimumDegreeAlternatingFreshController,
    OpenEgoAlternatingFreshController,
    rotting_diagnostics,
)
from .control.actions import ActionKind, EntityPair, QueryAction
from .control.triaction import LearningFreeTriActionController, LocalCandidateSnapshot
from evaluation.graph_recovery import TruthData, aggregate_trajectory, evaluate_recovery
from .metrics.graph import (
    compute_batch_scores,
    compute_node_scores,
    merge_batch,
    simple_projection,
)
from .models import CandidateBatch, normalize_label
from .prompts import PromptLibrary
from .wandb_tracking import WandbExperimentTracker


def _load_entity_text_units(data_dir: str | Path) -> dict[str, frozenset[str]]:
    """Load input-side memberships without exposing relationship labels.

    The controller only looks up endpoints that already exist in its online
    graph.  It never enumerates this mapping to discover new entity titles.
    """

    frame = pd.read_parquet(
        Path(data_dir) / "entities.parquet",
        columns=["title", "text_unit_ids"],
    )
    memberships: dict[str, set[str]] = {}
    for record in frame.to_dict(orient="records"):
        title = normalize_label(record.get("title"))
        if not title:
            continue
        raw_units = record.get("text_unit_ids")
        if hasattr(raw_units, "tolist"):
            raw_units = raw_units.tolist()
        if isinstance(raw_units, str):
            values = [raw_units]
        elif isinstance(raw_units, (list, tuple, set)):
            values = raw_units
        else:
            values = []
        memberships.setdefault(title, set()).update(
            str(value) for value in values if str(value).strip()
        )
    return {title: frozenset(values) for title, values in memberships.items()}


class MedicalExtractionPipeline:
    """Run adaptive extraction while preserving auditable turn-level artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config
        self.run_dir = Path(config.output_root).resolve() / config.run_id
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {self.run_dir}. Choose a new --run-id."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.graph = nx.MultiDiGraph()
        self.truth = TruthData.load(config.data_dir)
        self.mode_controller = AdaptiveModeController(
            initial_epsilon=config.initial_epsilon,
            epsilon_decay=config.epsilon_decay,
            min_epsilon=config.min_epsilon,
            htsn_threshold=config.htsn_threshold,
            htsn_window=config.htsn_window,
            adaptive_threshold=config.adaptive_htsn_threshold,
            explore_success_window=config.explore_success_window,
            explore_success_min_samples=config.explore_success_min_samples,
            explore_success_threshold=config.explore_success_threshold,
            max_consecutive_failed_explore=config.max_consecutive_failed_explore,
            sampling_policy=config.mode_sampling_policy,
            deterministic_explore_surplus_cap=(
                config.deterministic_explore_surplus_cap
            ),
            seed=config.random_seed,
        )
        if config.anchor_sampling_policy == "uniform_fresh":
            self.arm_controller = FreshAnchorController(seed=config.random_seed)
        elif config.anchor_sampling_policy == "agea_hub":
            self.arm_controller = AgeaHubController(seed=config.random_seed)
        elif config.anchor_sampling_policy == "frontier_alternating_fresh":
            self.arm_controller = FrontierAlternatingFreshController(
                seed=config.random_seed
            )
        elif config.anchor_sampling_policy == "minimum_degree_alternating_fresh":
            self.arm_controller = MinimumDegreeAlternatingFreshController(
                seed=config.random_seed
            )
        elif config.anchor_sampling_policy in {
            "degree_conditioned_bnrr_high_fresh",
            "degree_conditioned_bnrr_high_then_degree_frontier_fresh",
            "degree_conditioned_bnrr_low_fresh",
            "degree_conditioned_bnrr_scheduled_fresh",
            "degree_conditioned_shuffled_bnrr_high_fresh",
            "degree_conditioned_shuffled_bnrr_scheduled_fresh",
            "degree_conditioned_uniform_fresh",
            "degree_conditioned_shuffled_bnrr_low_fresh",
        }:
            variants = {
                "degree_conditioned_bnrr_high_fresh": "bnrr_high",
                "degree_conditioned_bnrr_high_then_degree_frontier_fresh": (
                    "bnrr_high_then_dfa"
                ),
                "degree_conditioned_bnrr_low_fresh": "bnrr_low",
                "degree_conditioned_bnrr_scheduled_fresh": (
                    "bnrr_high_scheduled"
                ),
                "degree_conditioned_uniform_fresh": "uniform",
                "degree_conditioned_shuffled_bnrr_high_fresh": (
                    "shuffled_bnrr_high"
                ),
                "degree_conditioned_shuffled_bnrr_low_fresh": (
                    "shuffled_bnrr_low"
                ),
                "degree_conditioned_shuffled_bnrr_scheduled_fresh": (
                    "shuffled_bnrr_high_scheduled"
                ),
            }
            self.arm_controller = DegreeConditionedBnrrFreshController(
                policy_name=config.anchor_sampling_policy,
                variant=variants[config.anchor_sampling_policy],
                schedule_profile=config.bnrr_schedule_profile,
                schedule_total_turns=config.turns,
                schedule_start_fraction=config.bnrr_schedule_start_fraction,
                schedule_end_fraction=config.bnrr_schedule_end_fraction,
                seed=config.random_seed,
            )
        elif config.anchor_sampling_policy in {
            "global_bnrr_raw_annealed_fresh",
            "global_bnrr_normalized_annealed_fresh",
            "global_shuffled_bnrr_raw_annealed_fresh",
        }:
            variants = {
                "global_bnrr_raw_annealed_fresh": "raw",
                "global_bnrr_normalized_annealed_fresh": "normalized",
                "global_shuffled_bnrr_raw_annealed_fresh": "shuffled_raw",
            }
            self.arm_controller = GlobalBnrrAnnealingFreshController(
                policy_name=config.anchor_sampling_policy,
                variant=variants[config.anchor_sampling_policy],
                schedule_total_turns=config.turns,
                initial_pool_fraction=(
                    config.global_bnrr_initial_pool_fraction
                ),
                relax_exploit_fraction=(
                    config.global_bnrr_relax_exploit_fraction
                ),
                seed=config.random_seed,
            )
        elif config.anchor_sampling_policy == "open_ego_alternating_fresh":
            self.arm_controller = OpenEgoAlternatingFreshController(
                seed=config.random_seed,
                minimum_topology_degree=1,
                policy_name=config.anchor_sampling_policy,
            )
        elif (
            config.anchor_sampling_policy
            == "supported_open_ego_alternating_fresh"
        ):
            self.arm_controller = OpenEgoAlternatingFreshController(
                seed=config.random_seed,
                minimum_topology_degree=2,
                policy_name=config.anchor_sampling_policy,
            )
        elif (
            config.anchor_sampling_policy
            == "structural_opportunity_fresh"
        ):
            self.arm_controller = OpenEgoAlternatingFreshController(
                seed=config.random_seed,
                minimum_topology_degree=1,
                policy_name=config.anchor_sampling_policy,
                topology_score_name="structural_opportunity",
                topology_slot_period=1,
            )
        elif config.anchor_sampling_policy == "ts_pl_fewa":
            self.arm_controller = EpochFewaController(
                delta=config.fewa_delta,
                max_arms=config.fewa_max_arms,
                seed=config.random_seed,
            )
        else:  # guarded by ExperimentConfig.validate
            raise ValueError(config.anchor_sampling_policy)
        self.tri_controller: LearningFreeTriActionController | None = None
        self.prompt_library: PromptLibrary | None = None
        if config.controller_name == "learning_free_tri_action":
            self.tri_controller = LearningFreeTriActionController(
                variant=config.tri_action_variant,
                seed=config.random_seed,
                closure_bundle_size=config.closure_bundle_size,
                closure_bnrr_percentile=config.closure_bnrr_percentile,
                prevent_consecutive_closure=config.prevent_consecutive_closure,
                closure_pair_policy=config.closure_pair_policy,
                closure_min_shared_text_units=(
                    config.closure_min_shared_text_units
                ),
                max_closure_local_share=config.max_closure_local_share,
                entity_text_units=(
                    _load_entity_text_units(config.data_dir)
                    if config.closure_pair_policy == "text_unit_overlap"
                    else {}
                ),
            )
            self.prompt_library = PromptLibrary(config.prompt_dir)
        self.adapter = AgeaGraphRagAdapter(
            graph_root=config.graph_root,
            data_dir=config.data_dir,
            run_dir=self.run_dir,
            query_method=config.query_method,
            disable_api_thinking=config.disable_api_thinking,
            graphrag_query_retries=config.graphrag_query_retries,
            enable_graph_filter=config.enable_graph_filter,
            graph_filter_model=config.graph_filter_model,
        )
        self.turn_records: list[dict[str, Any]] = []
        self.query_history: list[dict[str, Any]] = []
        self.exploit_pulls = 0
        self.wandb = WandbExperimentTracker.from_config(config, self.run_dir)

    def _available_arms(
        self,
    ) -> tuple[
        list[str],
        dict[str, float],
        dict[str, dict[str, float | int]],
    ]:
        scores = compute_node_scores(self.graph)
        projected = simple_projection(self.graph)
        pulled = set(getattr(self.arm_controller, "rewards", {}))
        eligible_nodes = [
            node for node, score in scores.items() if score.degree > 0
        ]
        priors = {
            node: score.sensitivity
            for node, score in scores.items()
            if score.degree > 0
        }
        topology: dict[str, dict[str, float | int]] = {}
        for node, score in scores.items():
            if score.sensitivity <= 0.0 or score.degree <= 0:
                continue
            pulled_neighbor_count = sum(
                1 for neighbor in projected.neighbors(node) if neighbor in pulled
            )
            pulled_neighbor_fraction = pulled_neighbor_count / score.degree
            openness = score.bnrr / score.degree
            topology[node] = {
                "degree": score.degree,
                "neighbor_edges": score.neighbor_edges,
                "bnrr": score.bnrr,
                "openness": openness,
                "pulled_neighbor_count": pulled_neighbor_count,
                "pulled_neighbor_fraction": pulled_neighbor_fraction,
                "structural_opportunity": openness
                * (1.0 - pulled_neighbor_fraction),
            }
        return eligible_nodes, priors, topology

    def _seed_query_text(self) -> str:
        """Build the fixed first-turn query used by the formal protocol."""

        return append_extraction_command(self.config.seed_query)

    def _query_text(
        self,
        mode: str,
        turn: int,
        anchor: str | None,
        action: QueryAction | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self.tri_controller is not None:
            if action is None:
                raise ValueError("Tri-action query rendering requires an explicit action")
            return self._tri_action_query_text(action, turn)

        anchor_required = bool(
            self.config.require_exploit_anchor_in_query
            and mode == "exploit"
            and anchor
        )
        if turn == 1:
            query = self._seed_query_text()
            return query, _query_generation_metadata(
                query=query,
                source="seed",
                attempts=1,
                max_similarity=0.0,
                similarity_rejections=0,
                anchor_required=anchor_required,
                anchor_present=None,
                anchor_rejections=0,
                fallback_reason="none",
            )

        recent_novelty = _mean(
            record["batch"]["count_novelty"]
            for record in self.turn_records[-max(1, self.config.htsn_window) :]
        )
        anchor_round = 1 + sum(
            entry.get("anchor") == anchor
            for entry in self.query_history
            if anchor is not None
        )
        similarity_rejections = 0
        anchor_rejections = 0
        generation_error_attempts = 0
        last_generation_error: str | None = None
        max_similarity = 0.0
        max_attempts = max(1, self.config.query_generation_retries + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                query = self.adapter.generate_agentic_query(
                    mode=mode,
                    novelty_score=recent_novelty,
                    recent_history=self.query_history,
                    graph=self.graph,
                    dataset_name=self.config.dataset,
                    anchor=anchor,
                    anchor_round=anchor_round,
                    query_generator_model=self.config.query_generator_model,
                )
            except Exception as exc:
                generation_error_attempts += 1
                last_generation_error = f"{type(exc).__name__}: {exc}"
                continue
            anchor_present = (
                _query_contains_anchor(query, str(anchor))
                if anchor_required and anchor is not None
                else None
            )
            similarity = max(
                (
                    _query_similarity(query, previous["query"])
                    for previous in self.query_history
                ),
                default=0.0,
            )
            max_similarity = max(max_similarity, similarity)
            similarity_ok = similarity < self.config.query_similarity_threshold
            anchor_ok = not anchor_required or bool(anchor_present)
            if similarity_ok and anchor_ok:
                return query, _query_generation_metadata(
                    query=query,
                    source=(
                        "fixed_anchor_dynamic"
                        if mode == "exploit" and anchor
                        else "agea_dynamic"
                    ),
                    attempts=attempt,
                    max_similarity=max_similarity,
                    similarity_rejections=similarity_rejections,
                    anchor_required=anchor_required,
                    anchor_present=anchor_present,
                    anchor_rejections=anchor_rejections,
                    fallback_reason="none",
                    generation_error_attempts=generation_error_attempts,
                    last_generation_error=last_generation_error,
                )
            similarity_rejections += int(not similarity_ok)
            anchor_rejections += int(not anchor_ok)

        # A repeated default response from a failing query generator must not
        # recreate the old five-template loop.  This fallback stays deterministic
        # and auditable while making the requested target/round explicit.
        if mode == "exploit" and anchor:
            domain_query = (
                f"Perform focused {self.config.dataset} relationship expansion round {anchor_round} "
                f"for {anchor}. Retrieve only additional named entities and directly "
                f"supported relationships absent from earlier turns; diversification "
                f"fallback for turn {turn}."
            )
        else:
            topic = self.config.exploration_queries[(turn - 2) % len(self.config.exploration_queries)]
            domain_query = (
                f"{topic} Select a previously unqueried subdomain and return different "
                f"named entities and directly supported relationships; diversification "
                f"fallback for turn {turn}."
            )
        query = append_extraction_command(domain_query)
        anchor_present = (
            _query_contains_anchor(query, str(anchor))
            if anchor_required and anchor is not None
            else None
        )
        if anchor_required and not anchor_present:
            raise ValueError(
                f"Diversified fallback for turn {turn} omitted anchor {anchor!r}"
            )
        if generation_error_attempts == max_attempts:
            fallback_reason = "provider_exhausted"
        elif generation_error_attempts:
            fallback_reason = "generation_and_validation_exhausted"
        elif anchor_rejections and similarity_rejections:
            fallback_reason = "mixed_exhausted"
        elif anchor_rejections:
            fallback_reason = "anchor_exhausted"
        else:
            fallback_reason = "similarity_exhausted"
        return query, _query_generation_metadata(
            query=query,
            source="diversified_fallback",
            attempts=max_attempts,
            max_similarity=max_similarity,
            similarity_rejections=similarity_rejections,
            anchor_required=anchor_required,
            anchor_present=anchor_present,
            anchor_rejections=anchor_rejections,
            fallback_reason=fallback_reason,
            generation_error_attempts=generation_error_attempts,
            last_generation_error=last_generation_error,
        )

    def _tri_action_query_text(
        self, action: QueryAction, turn: int
    ) -> tuple[str, dict[str, Any]]:
        """Render the selected action exclusively from the configured prompt bundle."""

        if self.prompt_library is None:
            raise RuntimeError("Tri-action prompt library is not initialized")
        anchor = normalize_label(action.anchor) if action.anchor else ""
        outgoing: list[str] = []
        incoming: list[str] = []
        neighbor_context: list[str] = []
        if anchor and anchor in self.graph:
            for source, target, attrs in self.graph.out_edges(anchor, data=True):
                relation = str(attrs.get("rel") or attrs.get("relation") or "related_to")
                outgoing.append(f"- {source} -> {target} [{relation}]")
            for source, target, attrs in self.graph.in_edges(anchor, data=True):
                relation = str(attrs.get("rel") or attrs.get("relation") or "related_to")
                incoming.append(f"- {source} -> {target} [{relation}]")
            neighbor_context = sorted(set(outgoing + incoming))

        recent_queries = [
            _domain_query(str(entry.get("query", "")))[:240]
            for entry in self.query_history[-3:]
        ]
        if action.kind == ActionKind.CLOSURE and not action.pairs:
            raise ValueError("Closure prompt requires at least one entity pair")
        numbered_pairs = "\n".join(
            f"{index}. {pair.left} | {pair.right}"
            for index, pair in enumerate(action.pairs, start=1)
        )
        pair_questions = "\n".join(
            f'{index}. "{pair.left}" and "{pair.right}"'
            for index, pair in enumerate(action.pairs, start=1)
        )
        entity_a = action.pairs[0].left if action.pairs else ""
        entity_b = action.pairs[0].right if action.pairs else ""
        values = {
            "dataset": self.config.dataset,
            "turn": turn,
            "seed_query": self.config.seed_query,
            "topic": action.topic or "",
            "recent_queries": "\n".join(f"- {item}" for item in recent_queries)
            or "- none",
            "anchor": anchor,
            "degree": self.graph.degree(anchor) if anchor in self.graph else 0,
            "known_outgoing": "\n".join(outgoing[:100]) or "- none",
            "known_incoming": "\n".join(incoming[:100]) or "- none",
            "numbered_pairs": numbered_pairs or "- none",
            "pair_questions": pair_questions or "- none",
            "entity_a": entity_a,
            "entity_b": entity_b,
            "neighbor_context": "\n".join(neighbor_context[:100]) or "- none",
            "evidence_packets": str(
                action.metadata.get("evidence_packets_json", "- none")
            ),
        }
        rendered = self.prompt_library.render(
            action,
            values=values,
            seed=bool(action.metadata.get("seed")),
        )
        anchor_required = bool(action.anchor and action.kind != ActionKind.CLOSURE)
        closure_pair_present = (
            all(
                _query_contains_anchor(rendered.text, endpoint)
                for pair in action.pairs
                for endpoint in (pair.left, pair.right)
            )
            if action.kind == ActionKind.CLOSURE
            else None
        )
        metadata = _query_generation_metadata(
            query=rendered.text,
            source="file_backed_action_prompt",
            attempts=1,
            max_similarity=max(
                (
                    _query_similarity(rendered.text, previous["query"])
                    for previous in self.query_history
                ),
                default=0.0,
            ),
            similarity_rejections=0,
            anchor_required=anchor_required,
            anchor_present=(
                _query_contains_anchor(rendered.text, anchor)
                if anchor_required
                else None
            ),
            anchor_rejections=0,
            fallback_reason="none",
        )
        metadata.update(
            {
                "action_kind": action.kind.value,
                "closure_pair_present": closure_pair_present,
                "template_name": rendered.template_name,
                "template_path": rendered.template_path,
                "output_contract_path": rendered.output_contract_path,
            }
        )
        return rendered.text, metadata

    def run(self) -> dict[str, Any]:
        """Run the experiment and always release adapter network resources."""

        exit_code = 1
        tracker = getattr(self, "wandb", None)
        try:
            if tracker is not None:
                tracker.start()
            summary = self._run()
            exit_code = 0
            return summary
        except BaseException as error:
            if tracker is not None:
                tracker.fail(error)
            raise
        finally:
            try:
                self.adapter.close()
            finally:
                if tracker is not None:
                    tracker.finish(exit_code)

    def _run(self) -> dict[str, Any]:
        _write_json(self.run_dir / "config.json", self.config.to_dict())
        htsn_history: list[float] = []
        mode_history: list[str] = []
        meaningful_gain_history: list[bool] = []

        for turn in range(1, self.config.turns + 1):
            tri_snapshot: LocalCandidateSnapshot | None = None
            if self.tri_controller is not None:
                tri_snapshot = self.tri_controller.build_snapshot(self.graph)
                arms = list(tri_snapshot.local_entities)
                priors = {
                    node: tri_snapshot.node_scores[node].bnrr for node in arms
                }
                topology: dict[str, dict[str, float | int]] = {}
            else:
                arms, priors, topology = self._available_arms()
            decision = self.mode_controller.choose(
                turn,
                arms,
                htsn_history,
                mode_history=mode_history,
                meaningful_gain_history=meaningful_gain_history,
            )
            mode = decision.mode
            controller_policy = (
                "learning_free_tri_action"
                if self.tri_controller is not None
                else str(self.arm_controller.state_dict().get("policy", "unknown"))
            )
            arm_decision: dict[str, Any] = {
                "policy": controller_policy,
                "selected_arm": None,
                "reason": "mode_explore",
                "eligible_arm_count": len(arms),
                "active_arms": [],
            }
            selection: dict[str, Any] = {}
            anchor: str | None = None
            action: QueryAction | None = None
            if self.tri_controller is not None:
                if mode == "exploit":
                    if tri_snapshot is None:
                        raise RuntimeError("Missing tri-action candidate snapshot")
                    action = self.tri_controller.select_local(
                        tri_snapshot, turn=turn
                    )
                    if action is None:
                        mode = "explore"
                        decision = replace(
                            decision,
                            mode=mode,
                            reason="tri_action_returned_no_local_action",
                            explore_suppressed=False,
                        )
                    else:
                        self.exploit_pulls += 1
                if mode == "explore":
                    action = self.tri_controller.select_global(
                        self.config.exploration_queries,
                        turn=turn,
                        seed_topic=(self.config.seed_query if turn == 1 else None),
                    )
                if action is None:
                    raise RuntimeError("Tri-action controller failed to select an action")
                anchor = action.anchor
                arm_decision = {
                    "policy": controller_policy,
                    "variant": self.config.tri_action_variant,
                    "selected_arm": anchor,
                    "reason": action.metadata.get("reason", "unknown"),
                    "eligible_arm_count": len(arms),
                    "active_arms": [anchor] if anchor else [],
                    "action": action.to_dict(),
                    "incident_pool_size": (
                        len(tri_snapshot.incident_eligible) if tri_snapshot else 0
                    ),
                    "closure_executable_pool_size": (
                        len(tri_snapshot.closure_executable) if tri_snapshot else 0
                    ),
                    "closure_eligible_pool_size": (
                        len(tri_snapshot.closure_eligible) if tri_snapshot else 0
                    ),
                }
            else:
                if mode == "exploit":
                    if isinstance(
                        self.arm_controller,
                        (
                            DegreeConditionedBnrrFreshController,
                            GlobalBnrrAnnealingFreshController,
                        ),
                    ):
                        anchor = self.arm_controller.select(
                            arms,
                            self.exploit_pulls + 1,
                            priors,
                            topology=topology,
                            global_turn=turn,
                        )
                    elif isinstance(
                        self.arm_controller,
                        (
                            MinimumDegreeAlternatingFreshController,
                            OpenEgoAlternatingFreshController,
                            AgeaHubController,
                        ),
                    ):
                        anchor = self.arm_controller.select(
                            arms,
                            self.exploit_pulls + 1,
                            priors,
                            topology=topology,
                        )
                    else:
                        anchor = self.arm_controller.select(
                            arms, self.exploit_pulls + 1, priors
                        )
                    selection = (
                        self.arm_controller.selection_history[-1]
                        if self.arm_controller.selection_history
                        else {}
                    )
                    arm_decision = {
                        "policy": controller_policy,
                        "selected_arm": anchor,
                        "reason": selection.get("reason", "unknown"),
                        "eligible_arm_count": len(arms),
                        "active_arms": list(self.arm_controller.active_arms),
                        "epoch_id": self.arm_controller.epoch_id,
                        "epoch_refreshed": bool(selection.get("epoch_refreshed")),
                        "selection": selection,
                        "admission": (
                            self.arm_controller.admission_history[-1]
                            if selection.get("epoch_refreshed")
                            and self.arm_controller.admission_history
                            else None
                        ),
                    }
                if mode == "exploit" and anchor is None:
                    mode = "explore"
                    decision = replace(
                        decision,
                        mode=mode,
                        reason="arm_controller_returned_no_anchor",
                        explore_suppressed=False,
                    )
                    arm_decision["reason"] = "no_anchor_fallback_to_explore"
                elif mode == "exploit":
                    self.exploit_pulls += 1

            query, query_generation = self._query_text(
                mode, turn, anchor, action=action
            )
            result = self.adapter.query(
                query,
                turn,
                closure_pairs=(
                    tuple((pair.left, pair.right) for pair in action.pairs)
                    if action is not None and action.kind == ActionKind.CLOSURE
                    else None
                ),
                response_override_path=(
                    self.config.shared_seed_response_path if turn == 1 else None
                ),
            )
            anchor_diagnostics = _response_anchor_diagnostics(
                anchor=anchor,
                response=result.response,
                batch=result.batch,
            )
            action_metrics = _action_batch_metrics(
                graph=self.graph,
                batch=result.batch,
                action=action,
                truth=self.truth,
            )
            batch_score = compute_batch_scores(self.graph, result.batch)
            candidate_atoms = batch_score.candidate_nodes + batch_score.candidate_edges
            if candidate_atoms > self.config.reward_normalizer:
                raise ValueError(
                    f"Turn {turn} produced {candidate_atoms} candidate atoms, exceeding "
                    f"the preregistered reward_normalizer M={self.config.reward_normalizer}."
                )
            # The proposal defines the FEWA reward as Z=Y_HA/M for a fixed,
            # preregistered maximum number of candidate atoms per query.
            raw_batch_reward = batch_score.y_ha / self.config.reward_normalizer
            merge_batch(self.graph, result.batch)
            evaluation = evaluate_recovery(self.graph, self.truth)

            if self.tri_controller is None:
                effective_fewa_reward = _effective_fewa_reward(
                    anchor=anchor,
                    raw_batch_reward=raw_batch_reward,
                    anchor_diagnostics=anchor_diagnostics,
                )
                self.arm_controller.observe(anchor, effective_fewa_reward or 0.0)
            else:
                effective_fewa_reward = None
            # A batch has a meaningful historical score only when the pre-turn
            # graph contains at least one non-isolated arm.  In particular, do
            # not feed the mandated cold-start zero back into the controller.
            if self.tri_controller is not None:
                feedback_value = action_metrics["directed_pair_count_novelty"]
                feedback_name = "directed_pair_count_novelty"
                htsn_history.append(feedback_value)
            elif arms:
                feedback_name = self.config.mode_feedback_signal
                feedback_value = (
                    batch_score.count_novelty
                    if feedback_name == "count_novelty"
                    else batch_score.htsn_combined
                )
                htsn_history.append(feedback_value)
            else:
                feedback_name = self.config.mode_feedback_signal
            gain = {
                "graph": bool(batch_score.new_nodes or batch_score.new_edges),
                "relation": (
                    action_metrics["new_directed_pairs"] > 0
                    if self.tri_controller is not None
                    else batch_score.new_edges > 0
                ),
                "sensitive": batch_score.y_ha > 0.0,
            }
            gain["meaningful"] = gain["relation"] or gain["sensitive"]
            mode_history.append(mode)
            meaningful_gain_history.append(gain["meaningful"])
            mode_decision_payload = decision.to_dict()
            mode_decision_payload["signal_name"] = feedback_name

            record = {
                "turn": turn,
                "mode": mode,
                "decision_reason": decision.reason,
                "mode_decision": mode_decision_payload,
                "arm_decision": arm_decision,
                "anchor": anchor,
                "action": action.to_dict() if action else None,
                # ``reward`` remains the per-query raw sensitive yield for CSV
                # compatibility.  FEWA is updated only with the separately
                # recorded, anchor-attributable effective reward.
                "reward": raw_batch_reward,
                "raw_batch_reward": raw_batch_reward,
                "effective_fewa_reward": effective_fewa_reward,
                "reward_attribution_warning": bool(
                    (
                        action_metrics["adherence"] < 1.0
                        if action is not None
                        else anchor is not None
                        and not anchor_diagnostics["anchor_response_adherent"]
                    )
                ),
                "query_generation": query_generation,
                "anchor_diagnostics": anchor_diagnostics,
                "action_metrics": action_metrics,
                "gain": gain,
                "parser": result.stats,
                "batch": batch_score.to_dict(),
                "evaluation": evaluation,
            }
            self.turn_records.append(record)
            self.query_history.append(
                {
                    "turn": turn,
                    "mode": mode,
                    "decision_reason": decision.reason,
                    "mode_decision": mode_decision_payload,
                    "arm_decision": arm_decision,
                    "anchor": anchor,
                    "action": action.to_dict() if action else None,
                    "query": query,
                    "query_generation": query_generation,
                    "anchor_diagnostics": anchor_diagnostics,
                    "action_metrics": {
                        key: value
                        for key, value in action_metrics.items()
                        if key != "evaluation_only"
                    },
                    "gain": gain,
                    "raw_batch_reward": raw_batch_reward,
                    "effective_fewa_reward": effective_fewa_reward,
                    "response_path": result.response_path,
                    "retrieved_context_path": result.retrieved_context_path,
                    "htsn_combined": batch_score.htsn_combined,
                    "count_novelty": batch_score.count_novelty,
                    "retrospective_gain": batch_score.retrospective_gain,
                    "nodes_added_to_graph": batch_score.new_nodes,
                    "edges_added_to_graph": batch_score.new_edges,
                    "newly_discovered_entity_names": sorted(
                        batch_score.new_node_weights
                    ),
                    "seeds_used": [anchor] if anchor else [],
                    "seeds_with_rounds": [],
                }
            )
            self._save_progress()
            self.wandb.log_turn(record)
            print(
                f"[turn {turn}/{self.config.turns}] mode={mode} "
                f"action={action.kind.value if action else 'legacy'} "
                f"anchor={anchor or '-'} "
                f"candidates={batch_score.candidate_nodes + batch_score.candidate_edges} "
                f"new={batch_score.new_nodes + batch_score.new_edges} "
                f"HTSN={batch_score.htsn_combined:.4f} "
                f"node_TSC={evaluation['node_tsc_rank']:.4f}"
            )

        summary = self._summary()
        _write_json(self.run_dir / "summary.json", summary)
        (self.run_dir / "EXPERIMENT_RESULTS.md").write_text(
            _render_results(summary, self.turn_records), encoding="utf-8"
        )
        self.wandb.complete(summary)
        return summary

    def _save_progress(self) -> None:
        _write_json(self.run_dir / "turn_metrics.json", self.turn_records)
        _write_json(self.run_dir / "query_history.json", self.query_history)
        controller_state = (
            self.tri_controller.state_dict()
            if self.tri_controller is not None
            else self.arm_controller.state_dict()
        )
        controller_payload = {
            "mode_decisions": [
                record.get("mode_decision", {}) for record in self.turn_records
            ],
            "controller": controller_state,
        }
        if self.tri_controller is not None:
            controller_payload["tri_action_controller"] = controller_state
        else:
            # Preserve the legacy key for existing result consumers.
            controller_payload["arm_controller"] = controller_state
        _write_json(
            self.run_dir / "controller_state.json", controller_payload
        )
        decision_records: list[dict[str, Any]] = []
        evaluation_records: list[dict[str, Any]] = []
        for record in self.turn_records:
            action_metrics = dict(record.get("action_metrics", {}))
            evaluation_only = action_metrics.pop("evaluation_only", {})
            decision_records.append(
                {
                    "turn": record["turn"],
                    "mode": record["mode"],
                    "decision_reason": record["decision_reason"],
                    "mode_decision": record.get("mode_decision", {}),
                    "local_decision": record.get("arm_decision", {}),
                    "action": record.get("action"),
                    "query_generation": record.get("query_generation", {}),
                    "parser": record.get("parser", {}),
                    "action_metrics": action_metrics,
                    "batch": record.get("batch", {}),
                    "gain": record.get("gain", {}),
                }
            )
            evaluation_records.append(
                {
                    "turn": record["turn"],
                    "evaluation": record.get("evaluation", {}),
                    "action_evaluation": evaluation_only,
                }
            )
        _write_jsonl(self.run_dir / "decision_state.jsonl", decision_records)
        _write_jsonl(
            self.run_dir / "evaluation_metrics.jsonl", evaluation_records
        )
        _write_json(
            self.run_dir / "extracted_graph.json", nx.node_link_data(self.graph, edges="edges")
        )
        nx.write_graphml(self.graph, self.run_dir / "extracted_graph.graphml")
        _write_flat_csv(self.run_dir / "turn_metrics.csv", self.turn_records)

    def _summary(self) -> dict[str, Any]:
        evaluations = [record["evaluation"] for record in self.turn_records]
        batches = [record["batch"] for record in self.turn_records]
        final_evaluation = evaluations[-1] if evaluations else {}
        query_diagnostics = _query_diagnostics(self.query_history, self.turn_records)
        mode_diagnostics = _mode_diagnostics(self.turn_records)
        tri_action_diagnostics: dict[str, Any] = {}
        if self.tri_controller is not None:
            tri_action_diagnostics = _tri_action_diagnostics(self.turn_records)
            anchor_diagnostics = tri_action_diagnostics
            arm_diagnostics = self.tri_controller.state_dict()
            rotting = {}
        else:
            anchor_diagnostics = _aggregate_anchor_diagnostics(self.turn_records)
            arm_diagnostics = _arm_diagnostics(self.arm_controller.state_dict())
            rotting = rotting_diagnostics(self.arm_controller.rewards)
        formal_acceptance = {
            "query_unique_rate": {
                "value": query_diagnostics["query_unique_rate"],
                "threshold": self.config.acceptance_min_query_unique_rate,
                "passed": query_diagnostics["query_unique_rate"]
                >= self.config.acceptance_min_query_unique_rate,
            },
            "zero_gain_rate": {
                "value": query_diagnostics["zero_gain_rate"],
                "threshold": self.config.acceptance_max_zero_gain_rate,
                "passed": query_diagnostics["zero_gain_rate"]
                <= self.config.acceptance_max_zero_gain_rate,
            },
            "max_consecutive_zero_gain": {
                "value": query_diagnostics["max_consecutive_zero_gain"],
                "threshold": self.config.acceptance_max_consecutive_zero_gain,
                "passed": query_diagnostics["max_consecutive_zero_gain"]
                <= self.config.acceptance_max_consecutive_zero_gain,
            },
            "max_consecutive_meaningful_zero_gain": {
                "value": query_diagnostics[
                    "max_consecutive_meaningful_zero_gain"
                ],
                "threshold": self.config.acceptance_max_consecutive_meaningful_zero_gain,
                "passed": query_diagnostics[
                    "max_consecutive_meaningful_zero_gain"
                ]
                <= self.config.acceptance_max_consecutive_meaningful_zero_gain,
            },
            "anchor_query_adherence_rate": {
                "value": anchor_diagnostics["query_anchor_adherence_rate"],
                "threshold": self.config.acceptance_min_anchor_query_adherence,
                "passed": anchor_diagnostics["query_anchor_adherence_rate"]
                >= self.config.acceptance_min_anchor_query_adherence,
            },
            "both_post_seed_modes_observed": {
                "value": int(mode_diagnostics["both_post_seed_modes_observed"]),
                "threshold": int(self.config.acceptance_require_both_post_seed_modes),
                "passed": (
                    mode_diagnostics["both_post_seed_modes_observed"]
                    if self.config.acceptance_require_both_post_seed_modes
                    else True
                ),
            },
            "nonempty_main_response_rate": {
                "value": query_diagnostics["nonempty_main_response_rate"],
                "threshold": 1.0,
                "passed": query_diagnostics["nonempty_main_response_rate"] >= 1.0,
            },
        }
        formal_acceptance["all_passed"] = all(
            item["passed"]
            for item in formal_acceptance.values()
            if isinstance(item, dict)
        )
        return {
            "dataset": self.config.dataset,
            "run_id": self.config.run_id,
            "turns_completed": len(self.turn_records),
            "truth_nodes": len(self.truth.nodes),
            "truth_directed_edge_pairs": len(self.truth.edges),
            "final_graph_nodes": self.graph.number_of_nodes(),
            "final_graph_relation_triples": self.graph.number_of_edges(),
            "mean_htsn": _mean(item["htsn_combined"] for item in batches),
            "mean_count_novelty": _mean(item["count_novelty"] for item in batches),
            "mean_retrospective_gain": _mean(
                item["retrospective_gain"] for item in batches
            ),
            "controller_name": self.config.controller_name,
            "tri_action_variant": (
                self.config.tri_action_variant
                if self.tri_controller is not None
                else None
            ),
            "rotting_diagnostics": rotting,
            "arm_diagnostics": arm_diagnostics,
            "tri_action_diagnostics": tri_action_diagnostics,
            "query_diagnostics": query_diagnostics,
            "mode_diagnostics": mode_diagnostics,
            "anchor_diagnostics": anchor_diagnostics,
            "formal_acceptance": formal_acceptance,
            "trajectory": aggregate_trajectory(evaluations),
            "final": final_evaluation,
        }


def _mean(values: Any) -> float:
    materialized = list(values)
    return sum(float(value) for value in materialized) / len(materialized) if materialized else 0.0


def _action_batch_metrics(
    *,
    graph: nx.MultiDiGraph,
    batch: CandidateBatch,
    action: QueryAction | None,
    truth: TruthData,
) -> dict[str, Any]:
    """Score directed-pair novelty/adherence before merge; truth stays diagnostic."""

    historical_nodes = {normalize_label(node) for node in graph.nodes}
    historical_pairs = {
        (normalize_label(source), normalize_label(target))
        for source, target in graph.edges()
    }
    candidate_nodes = set(batch.nodes)
    candidate_pairs = {edge.pair for edge in batch.edges}
    new_nodes = candidate_nodes - historical_nodes
    new_pairs = candidate_pairs - historical_pairs

    if action is None or action.kind == ActionKind.GLOBAL:
        credited_pairs = set(candidate_pairs)
    elif action.kind == ActionKind.INCIDENT:
        anchor = normalize_label(action.anchor)
        credited_pairs = {
            pair for pair in candidate_pairs if anchor in pair
        }
    else:
        allowed = set(action.pairs)
        credited_pairs = {
            pair
            for pair in candidate_pairs
            if EntityPair.from_values(*pair) in allowed
        }

    credited_new_pairs = credited_pairs & new_pairs
    off_target_pairs = candidate_pairs - credited_pairs
    candidate_atoms = len(candidate_nodes) + len(candidate_pairs)
    novelty = (
        (len(new_nodes) + len(new_pairs)) / candidate_atoms
        if candidate_atoms
        else 0.0
    )
    directed_edge_novelty = (
        len(new_pairs) / len(candidate_pairs) if candidate_pairs else 0.0
    )
    adherence = (
        len(credited_pairs) / len(candidate_pairs) if candidate_pairs else 1.0
    )

    pre_known_truth_nodes = {
        truth.canonical(node) for node in historical_nodes
    } & truth.nodes
    canonical_new_pairs = {
        (truth.canonical(source), truth.canonical(target))
        for source, target in new_pairs
    }
    canonical_credited_new_pairs = {
        (truth.canonical(source), truth.canonical(target))
        for source, target in credited_new_pairs
    }
    matched_new_pairs = canonical_new_pairs & truth.edges
    matched_credited_pairs = canonical_credited_new_pairs & truth.edges
    residual_yield = {"uu": 0, "ku": 0, "kk": 0}
    credited_residual_yield = {"uu": 0, "ku": 0, "kk": 0}

    def residual_class(pair: tuple[str, str]) -> str:
        known = int(pair[0] in pre_known_truth_nodes) + int(
            pair[1] in pre_known_truth_nodes
        )
        return "kk" if known == 2 else "ku" if known == 1 else "uu"

    for pair in matched_new_pairs:
        residual_yield[residual_class(pair)] += 1
    for pair in matched_credited_pairs:
        credited_residual_yield[residual_class(pair)] += 1

    return {
        "action_kind": action.kind.value if action else "legacy",
        "validated_nodes": len(candidate_nodes),
        "validated_directed_pairs": len(candidate_pairs),
        "new_nodes": len(new_nodes),
        "new_directed_pairs": len(new_pairs),
        "credited_directed_pairs": len(credited_pairs),
        "credited_new_directed_pairs": len(credited_new_pairs),
        "off_target_directed_pairs": len(off_target_pairs),
        "adherence": adherence,
        "zero_pair_return": not candidate_pairs,
        "node_novelty": len(new_nodes) / len(candidate_nodes) if candidate_nodes else 0.0,
        "directed_edge_novelty": directed_edge_novelty,
        "directed_pair_count_novelty": novelty,
        "validated_pair_list": [list(pair) for pair in sorted(candidate_pairs)],
        "credited_pair_list": [list(pair) for pair in sorted(credited_pairs)],
        "evaluation_only": {
            "matched_new_directed_pairs": len(matched_new_pairs),
            "matched_credited_new_directed_pairs": len(matched_credited_pairs),
            "residual_yield": residual_yield,
            "credited_residual_yield": credited_residual_yield,
        },
    }


def _domain_query(query: str) -> str:
    domain = query.split("\n\nFor my record", 1)[0]
    domain = domain.split("\n\nEVIDENCE AND OUTPUT CONTRACT", 1)[0]
    return domain.strip()


def _query_tokens(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _domain_query(query).lower()))


def _query_similarity(first: str, second: str) -> float:
    left, right = _query_tokens(first), _query_tokens(second)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left | right else 0.0


def _normalized_phrase_present(text: str, phrase: str) -> bool:
    """Match a normalized entity phrase without accepting token substrings."""

    normalized_text = normalize_label(text)
    normalized_phrase = normalize_label(phrase)
    if not normalized_phrase:
        return False
    pattern = rf"(?<![A-Z0-9]){re.escape(normalized_phrase)}(?![A-Z0-9])"
    return re.search(pattern, normalized_text) is not None


def _query_contains_anchor(query: str, anchor: str) -> bool:
    """Check only the domain query, excluding the fixed extraction command."""

    return _normalized_phrase_present(_domain_query(query), anchor)


def _query_signature(query: str) -> str:
    return " ".join(sorted(_query_tokens(query)))


def _query_generation_metadata(
    *,
    query: str,
    source: str,
    attempts: int,
    max_similarity: float,
    similarity_rejections: int,
    anchor_required: bool,
    anchor_present: bool | None,
    anchor_rejections: int,
    fallback_reason: str,
    generation_error_attempts: int = 0,
    last_generation_error: str | None = None,
) -> dict[str, Any]:
    """Return a stable schema for generated and fallback queries."""

    return {
        "source": source,
        "attempts": attempts,
        # Keep the original key for old CSV consumers.
        "rejected_similar_queries": similarity_rejections,
        "similarity_rejected_queries": similarity_rejections,
        "max_similarity": max_similarity,
        "anchor_required": anchor_required,
        "anchor_present": anchor_present,
        "anchor_rejected_queries": anchor_rejections,
        "fallback_reason": fallback_reason,
        "generation_error_attempts": generation_error_attempts,
        "last_generation_error": last_generation_error,
        "domain_query_signature": _query_signature(query),
    }


def _response_anchor_diagnostics(
    *,
    anchor: str | None,
    response: str,
    batch: CandidateBatch,
) -> dict[str, Any]:
    """Audit whether an exploit response actually returns anchor relations."""

    if anchor is None:
        return {
            "anchor_required": False,
            "anchor_in_response_text": None,
            "anchor_in_candidate_nodes": None,
            "anchor_incident_candidate_edges": 0,
            "anchor_response_adherent": None,
        }

    normalized_anchor = normalize_label(anchor)
    incident_edges = sum(
        edge.source == normalized_anchor or edge.target == normalized_anchor
        for edge in batch.edges
    )
    return {
        "anchor_required": True,
        "anchor_in_response_text": _normalized_phrase_present(response, anchor),
        "anchor_in_candidate_nodes": normalized_anchor in batch.nodes,
        "anchor_incident_candidate_edges": incident_edges,
        # A node mention without a relationship is not a successful exploit pull.
        "anchor_response_adherent": incident_edges > 0,
    }


def _effective_fewa_reward(
    *,
    anchor: str | None,
    raw_batch_reward: float,
    anchor_diagnostics: Mapping[str, Any],
) -> float | None:
    """Conservatively credit reward only to an anchor-adherent exploit pull."""

    if anchor is None:
        return None
    return (
        float(raw_batch_reward)
        if anchor_diagnostics.get("anchor_response_adherent") is True
        else 0.0
    )


def _query_diagnostics(
    queries: list[Mapping[str, Any]], turns: list[Mapping[str, Any]]
) -> dict[str, Any]:
    signatures = [_query_signature(str(entry["query"])) for entry in queries]
    unique_queries = len(set(signatures))
    graph_gains: list[bool] = []
    relation_gains: list[bool] = []
    sensitive_gains: list[bool] = []
    meaningful_gains: list[bool] = []
    for record in turns:
        batch = record.get("batch", {})
        gain = record.get("gain", {})
        graph_gain = bool(
            gain.get(
                "graph",
                bool(batch.get("new_nodes", 0) or batch.get("new_edges", 0)),
            )
        )
        relation_gain = bool(
            gain.get("relation", bool(batch.get("new_edges", 0)))
        )
        sensitive_gain = bool(
            gain.get("sensitive", float(batch.get("y_ha", 0.0)) > 0.0)
        )
        meaningful_gain = bool(
            gain.get("meaningful", relation_gain or sensitive_gain)
        )
        graph_gains.append(graph_gain)
        relation_gains.append(relation_gain)
        sensitive_gains.append(sensitive_gain)
        meaningful_gains.append(meaningful_gain)

    zero_graph_gain = [not value for value in graph_gains]
    zero_relation_gain = [not value for value in relation_gains]
    zero_sensitive_gain = [not value for value in sensitive_gains]
    zero_meaningful_gain = [not value for value in meaningful_gains]
    anchor_counts: dict[str, int] = {}
    for entry in queries:
        anchor = entry.get("anchor")
        if anchor:
            anchor_counts[str(anchor)] = anchor_counts.get(str(anchor), 0) + 1
    repeated = {anchor: count for anchor, count in anchor_counts.items() if count >= 2}
    total = len(queries)
    response_nonempty = [
        int(record.get("parser", {}).get("response_characters", 0)) > 0
        for record in turns
    ]
    fallback_turns = sum(
        entry.get("query_generation", {}).get("source")
        == "diversified_fallback"
        for entry in queries
    )
    return {
        "total_queries": total,
        "unique_queries": unique_queries,
        "query_unique_rate": unique_queries / total if total else 0.0,
        # Historical field names retain graph-gain semantics for result continuity.
        "zero_gain_turns": sum(zero_graph_gain),
        "zero_gain_rate": _rate(zero_graph_gain),
        "max_consecutive_zero_gain": _longest_true_streak(zero_graph_gain),
        "zero_graph_gain_turns": sum(zero_graph_gain),
        "zero_graph_gain_rate": _rate(zero_graph_gain),
        "zero_relation_gain_turns": sum(zero_relation_gain),
        "zero_relation_gain_rate": _rate(zero_relation_gain),
        "zero_sensitive_gain_turns": sum(zero_sensitive_gain),
        "zero_sensitive_gain_rate": _rate(zero_sensitive_gain),
        "zero_meaningful_gain_turns": sum(zero_meaningful_gain),
        "zero_meaningful_gain_rate": _rate(zero_meaningful_gain),
        "max_consecutive_meaningful_zero_gain": _longest_true_streak(
            zero_meaningful_gain
        ),
        "repeated_arms": len(repeated),
        "repeated_arm_counts": repeated,
        "nonempty_main_responses": sum(response_nonempty),
        "nonempty_main_response_rate": _rate(response_nonempty),
        "query_generator_fallback_turns": fallback_turns,
        "query_generator_fallback_rate": fallback_turns / total if total else 0.0,
    }


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _longest_true_streak(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _arm_diagnostics(state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize admission separately from actual arm selections."""

    policy = str(state.get("policy", state.get("admission_policy", "unknown")))
    admissions = list(state.get("admission_history", []))
    selections = list(state.get("selection_history", []))
    if not selections and state.get("decisions"):
        selections = [
            {
                "selected_arm": record.get("selected_arm"),
                "epoch_id": None,
                "exploit_pull": record.get("details", {}).get("exploit_pull"),
            }
            for record in state.get("decisions", [])
        ]

    admitted = {
        str(arm)
        for admission in admissions
        for arm in admission.get("active_arms", [])
    }
    selected = {
        str(record["selected_arm"])
        for record in selections
        if record.get("selected_arm")
    }
    delays: list[int] = []
    unqueried_slots = 0
    per_epoch: list[dict[str, Any]] = []
    for admission in admissions:
        epoch_id = admission.get("epoch_id")
        start = int(admission.get("exploit_pull_start", 0))
        epoch_selections = [
            record
            for record in selections
            if record.get("epoch_id") == epoch_id and record.get("selected_arm")
        ]
        epoch_delays: dict[str, int | None] = {}
        for arm in admission.get("active_arms", []):
            matching = [
                int(record.get("exploit_pull", start)) - start
                for record in epoch_selections
                if record.get("selected_arm") == arm
            ]
            delay = min(matching) if matching else None
            epoch_delays[str(arm)] = delay
            if delay is None:
                unqueried_slots += 1
            else:
                delays.append(delay)
        per_epoch.append(
            {
                "epoch_id": epoch_id,
                "active_arms": list(admission.get("active_arms", [])),
                "admission_to_first_query_delay": epoch_delays,
            }
        )

    turnovers: list[float] = []
    for previous, current in zip(admissions, admissions[1:]):
        left = set(previous.get("active_arms", []))
        right = set(current.get("active_arms", []))
        union = left | right
        turnovers.append(1.0 - len(left & right) / len(union) if union else 0.0)

    bnrr_selections = [
        record for record in selections if "bnrr_policy_tv" in record
    ]
    informative_selections = [
        record
        for record in bnrr_selections
        if bool(record.get("bnrr_informative", False))
    ]
    scheduled_selections = [
        record
        for record in bnrr_selections
        if record.get("schedule_profile") is not None
    ]
    selected_degree_histogram = Counter(
        str(record["selected_degree"])
        for record in bnrr_selections
        if record.get("selected_degree") is not None
    )
    degree_marginal_violations = sum(
        record.get("degree_marginal_preserved") is False
        for record in bnrr_selections
    )

    diagnostics = {
        "policy": policy,
        "epochs": len(admissions),
        "unique_admitted_arms": len(admitted),
        "unique_selected_arms": len(selected),
        "selected_arms": sorted(selected),
        "mean_admission_to_first_query_delay": _mean(delays),
        "unqueried_admitted_slots": unqueried_slots,
        "mean_epoch_arm_set_turnover": _mean(turnovers),
        "per_epoch": per_epoch,
    }
    if bnrr_selections:
        diagnostics.update(
            {
                "bnrr_selection_slots": len(bnrr_selections),
                "bnrr_informative_slots": len(informative_selections),
                "bnrr_informative_slot_rate": (
                    len(informative_selections) / len(bnrr_selections)
                ),
                "mean_bnrr_policy_tv": _mean(
                    float(record.get("bnrr_policy_tv", 0.0))
                    for record in bnrr_selections
                ),
                "mean_bnrr_low_policy_tv": _mean(
                    float(record.get("bnrr_low_policy_tv", 0.0))
                    for record in bnrr_selections
                ),
                "mean_bnrr_high_policy_tv": _mean(
                    float(record.get("bnrr_high_policy_tv", 0.0))
                    for record in bnrr_selections
                ),
                "mean_selected_direction_policy_tv": _mean(
                    float(record.get("selected_direction_policy_tv", 0.0))
                    for record in bnrr_selections
                ),
                "d1_stratum_rate": _rate(
                    [bool(record.get("d1_stratum", False)) for record in bnrr_selections]
                ),
                "singleton_stratum_rate": _rate(
                    [
                        bool(record.get("singleton_stratum", False))
                        for record in bnrr_selections
                    ]
                ),
                "paired_uniform_disagreement_rate": _rate(
                    [
                        bool(record.get("paired_bnrr_uniform_disagreement", False))
                        for record in bnrr_selections
                    ]
                ),
                "paired_shuffled_disagreement_rate": _rate(
                    [
                        bool(record.get("paired_bnrr_shuffled_disagreement", False))
                        for record in bnrr_selections
                    ]
                ),
                "paired_high_uniform_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_bnrr_high_uniform_disagreement", False
                            )
                        )
                        for record in bnrr_selections
                    ]
                ),
                "paired_high_shuffled_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_bnrr_high_shuffled_disagreement", False
                            )
                        )
                        for record in bnrr_selections
                    ]
                ),
                "paired_low_high_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_bnrr_low_high_disagreement", False
                            )
                        )
                        for record in bnrr_selections
                    ]
                ),
                "conditional_uniform_disagreement_rate": _rate(
                    [
                        bool(record.get("paired_bnrr_uniform_disagreement", False))
                        for record in informative_selections
                    ]
                ),
                "conditional_shuffled_disagreement_rate": _rate(
                    [
                        bool(record.get("paired_bnrr_shuffled_disagreement", False))
                        for record in informative_selections
                    ]
                ),
                "conditional_high_uniform_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_bnrr_high_uniform_disagreement", False
                            )
                        )
                        for record in informative_selections
                    ]
                ),
                "conditional_high_shuffled_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_bnrr_high_shuffled_disagreement", False
                            )
                        )
                        for record in informative_selections
                    ]
                ),
                "fresh_slot_rate": _rate(
                    [
                        record.get("base_pool_source") == "fresh"
                        for record in bnrr_selections
                    ]
                ),
                "least_pulled_slot_rate": _rate(
                    [
                        record.get("base_pool_source") == "least_pulled"
                        for record in bnrr_selections
                    ]
                ),
                "degree_marginal_violation_count": degree_marginal_violations,
                "selected_degree_histogram": dict(
                    sorted(selected_degree_histogram.items(), key=lambda item: int(item[0]))
                ),
                "schedule_selection_slots": len(scheduled_selections),
                "schedule_active_slot_rate": _rate(
                    [
                        bool(record.get("schedule_active", False))
                        for record in scheduled_selections
                    ]
                ),
                "mean_schedule_requested_tail_fraction": _mean(
                    float(record.get("schedule_requested_tail_fraction", 0.0))
                    for record in scheduled_selections
                ),
                "mean_schedule_realized_tail_share": _mean(
                    float(record.get("schedule_tail_pool_share", 0.0))
                    for record in scheduled_selections
                    if record.get("schedule_tail_pool_share") is not None
                ),
                "schedule_capacity_mismatch_count": sum(
                    not bool(record.get("schedule_capacity_matched", False))
                    for record in scheduled_selections
                ),
                "paired_scheduled_uniform_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_scheduled_uniform_disagreement", False
                            )
                        )
                        for record in scheduled_selections
                    ]
                ),
                "paired_scheduled_raw_shuffled_disagreement_rate": _rate(
                    [
                        bool(
                            record.get(
                                "paired_scheduled_raw_shuffled_disagreement",
                                False,
                            )
                        )
                        for record in scheduled_selections
                    ]
                ),
                "schedule_phase_counts": dict(
                    sorted(
                        Counter(
                            str(record.get("schedule_phase", "unknown"))
                            for record in scheduled_selections
                        ).items()
                    )
                ),
            }
        )
    return diagnostics


def _mode_diagnostics(turns: list[Mapping[str, Any]]) -> dict[str, Any]:
    seed_count = sum(
        record.get("decision_reason") == "seed_turn" for record in turns
    )
    post_seed = [
        record
        for record in turns
        if record.get("decision_reason") != "seed_turn"
    ]
    post_seed_counts = Counter(str(record.get("mode", "unknown")) for record in post_seed)
    reason_counts = Counter(
        str(record.get("decision_reason", "unknown")) for record in turns
    )
    by_mode: dict[str, dict[str, float | int]] = {}
    for mode in ("explore", "exploit"):
        records = [record for record in post_seed if record.get("mode") == mode]
        effective_rewards = [
            float(record["effective_fewa_reward"])
            for record in records
            if record.get("effective_fewa_reward") is not None
        ]
        by_mode[mode] = {
            "turns": len(records),
            "mean_htsn": _mean(
                record.get("batch", {}).get("htsn_combined", 0.0)
                for record in records
            ),
            "mean_raw_batch_reward": _mean(
                record.get("raw_batch_reward", record.get("reward", 0.0))
                for record in records
            ),
            "mean_effective_fewa_reward": _mean(effective_rewards),
            "mean_new_nodes": _mean(
                record.get("batch", {}).get("new_nodes", 0) for record in records
            ),
            "mean_new_edges": _mean(
                record.get("batch", {}).get("new_edges", 0) for record in records
            ),
        }

    post_seed_total = len(post_seed)
    return {
        "seed_turns": seed_count,
        "post_seed_turns": post_seed_total,
        "post_seed_explore_turns": post_seed_counts.get("explore", 0),
        "post_seed_exploit_turns": post_seed_counts.get("exploit", 0),
        "post_seed_explore_ratio": (
            post_seed_counts.get("explore", 0) / post_seed_total
            if post_seed_total
            else 0.0
        ),
        "post_seed_exploit_ratio": (
            post_seed_counts.get("exploit", 0) / post_seed_total
            if post_seed_total
            else 0.0
        ),
        "both_post_seed_modes_observed": (
            post_seed_counts.get("explore", 0) > 0
            and post_seed_counts.get("exploit", 0) > 0
        ),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "by_mode": by_mode,
    }


def _aggregate_anchor_diagnostics(
    turns: list[Mapping[str, Any]],
) -> dict[str, Any]:
    exploit_turns = [
        record
        for record in turns
        if record.get("mode") == "exploit" and record.get("anchor")
    ]
    query_adherent = [
        bool(record.get("query_generation", {}).get("anchor_present"))
        for record in exploit_turns
    ]
    response_text_hits = [
        bool(record.get("anchor_diagnostics", {}).get("anchor_in_response_text"))
        for record in exploit_turns
    ]
    response_node_hits = [
        bool(record.get("anchor_diagnostics", {}).get("anchor_in_candidate_nodes"))
        for record in exploit_turns
    ]
    response_adherent = [
        bool(record.get("anchor_diagnostics", {}).get("anchor_response_adherent"))
        for record in exploit_turns
    ]
    raw_rewards = [
        float(record.get("raw_batch_reward", record.get("reward", 0.0)))
        for record in exploit_turns
    ]
    effective_rewards = [
        float(record.get("effective_fewa_reward") or 0.0)
        for record in exploit_turns
    ]
    return {
        "eligible_exploit_turns": len(exploit_turns),
        "query_anchor_adherent_turns": sum(query_adherent),
        "query_anchor_adherence_rate": (
            _rate(query_adherent) if exploit_turns else 1.0
        ),
        "response_anchor_text_hit_turns": sum(response_text_hits),
        "response_anchor_text_hit_rate": _rate(response_text_hits),
        "response_anchor_node_hit_turns": sum(response_node_hits),
        "response_anchor_node_hit_rate": _rate(response_node_hits),
        "response_anchor_adherent_turns": sum(response_adherent),
        "response_anchor_adherence_rate": _rate(response_adherent),
        "reward_attribution_warning_turns": sum(not value for value in response_adherent),
        "mean_raw_exploit_reward": _mean(raw_rewards),
        "mean_effective_fewa_reward": _mean(effective_rewards),
        "anchor_rejected_query_generations": sum(
            int(record.get("query_generation", {}).get("anchor_rejected_queries", 0))
            for record in exploit_turns
        ),
    }


def _tri_action_diagnostics(turns: list[Mapping[str, Any]]) -> dict[str, Any]:
    actions = [
        str((record.get("action") or {}).get("kind", "unknown"))
        for record in turns
    ]
    action_counts = Counter(actions)
    local_turns = [
        record
        for record in turns
        if (record.get("action") or {}).get("kind")
        in {ActionKind.INCIDENT.value, ActionKind.CLOSURE.value}
    ]
    closure_turns = [
        record
        for record in local_turns
        if (record.get("action") or {}).get("kind") == ActionKind.CLOSURE.value
    ]
    query_adherent = [
        bool(
            record.get("query_generation", {}).get(
                "closure_pair_present"
                if (record.get("action") or {}).get("kind")
                == ActionKind.CLOSURE.value
                else "anchor_present"
            )
        )
        for record in local_turns
    ]
    response_adherence = [
        float(record.get("action_metrics", {}).get("adherence", 0.0))
        for record in local_turns
    ]
    local_kinds = [
        str((record.get("action") or {}).get("kind", ""))
        for record in local_turns
    ]
    consecutive_closure = sum(
        left == ActionKind.CLOSURE.value and right == ActionKind.CLOSURE.value
        for left, right in zip(local_kinds, local_kinds[1:])
    )
    singleton_closures = sum(
        bool((record.get("action") or {}).get("metadata", {}).get("singleton_bin"))
        for record in closure_turns
    )
    all_tie_closures = sum(
        bool((record.get("action") or {}).get("metadata", {}).get("all_tie_bin"))
        for record in closure_turns
    )
    selected_pairs = [
        tuple(pair)
        for record in closure_turns
        for pair in (record.get("action") or {}).get("pairs", [])
    ]
    residual_yield = Counter()
    credited_residual_yield = Counter()
    by_action: dict[str, dict[str, Any]] = {}
    for record in turns:
        evaluation_only = record.get("action_metrics", {}).get("evaluation_only", {})
        residual_yield.update(evaluation_only.get("residual_yield", {}))
        credited_residual_yield.update(
            evaluation_only.get("credited_residual_yield", {})
        )
    for kind in sorted(set(actions)):
        selected = [
            record
            for record in turns
            if (record.get("action") or {}).get("kind") == kind
        ]
        truth_yield = Counter()
        credited_truth_yield = Counter()
        for record in selected:
            evaluation_only = record.get("action_metrics", {}).get(
                "evaluation_only", {}
            )
            truth_yield.update(evaluation_only.get("residual_yield", {}))
            credited_truth_yield.update(
                evaluation_only.get("credited_residual_yield", {})
            )
        by_action[kind] = {
            "turns": len(selected),
            "zero_pair_returns": sum(
                bool(record.get("action_metrics", {}).get("zero_pair_return"))
                for record in selected
            ),
            "mean_adherence": _mean(
                record.get("action_metrics", {}).get("adherence", 0.0)
                for record in selected
            ),
            "validated_directed_pairs": sum(
                int(
                    record.get("action_metrics", {}).get(
                        "validated_directed_pairs", 0
                    )
                )
                for record in selected
            ),
            "new_directed_pairs": sum(
                int(record.get("action_metrics", {}).get("new_directed_pairs", 0))
                for record in selected
            ),
            "credited_new_directed_pairs": sum(
                int(
                    record.get("action_metrics", {}).get(
                        "credited_new_directed_pairs", 0
                    )
                )
                for record in selected
            ),
            "truth_residual_yield": dict(truth_yield),
            "credited_truth_residual_yield": dict(credited_truth_yield),
        }
    closure_count = len(closure_turns)
    return {
        # Compatibility keys consumed by the existing acceptance summary.
        "eligible_exploit_turns": len(local_turns),
        "query_anchor_adherent_turns": sum(query_adherent),
        "query_anchor_adherence_rate": _rate(query_adherent),
        "response_anchor_text_hit_turns": 0,
        "response_anchor_text_hit_rate": 0.0,
        "response_anchor_node_hit_turns": 0,
        "response_anchor_node_hit_rate": 0.0,
        "response_anchor_adherent_turns": sum(
            value >= 1.0 for value in response_adherence
        ),
        "response_anchor_adherence_rate": (
            _mean(response_adherence) if local_turns else 1.0
        ),
        "reward_attribution_warning_turns": sum(
            value < 1.0 for value in response_adherence
        ),
        "mean_raw_exploit_reward": _mean(
            record.get("raw_batch_reward", 0.0) for record in local_turns
        ),
        "mean_effective_fewa_reward": 0.0,
        "anchor_rejected_query_generations": 0,
        # Tri-action mechanism diagnostics.
        "action_counts": dict(action_counts),
        "post_seed_action_counts": dict(Counter(actions[1:])),
        "consecutive_closure_count": consecutive_closure,
        "unique_incident_anchors": len(
            {
                record.get("anchor")
                for record in local_turns
                if (record.get("action") or {}).get("kind")
                == ActionKind.INCIDENT.value
            }
        ),
        "unique_closure_anchors": len(
            {record.get("anchor") for record in closure_turns}
        ),
        "unique_closure_pairs": len(set(selected_pairs)),
        "repeated_closure_pairs": len(selected_pairs) - len(set(selected_pairs)),
        "singleton_bin_closure_ratio": (
            singleton_closures / closure_count if closure_count else 0.0
        ),
        "all_tie_bin_closure_ratio": (
            all_tie_closures / closure_count if closure_count else 0.0
        ),
        "mean_local_action_adherence": (
            _mean(response_adherence) if local_turns else 1.0
        ),
        "nonempty_local_pair_response_rate": _rate(
            [
                not bool(record.get("action_metrics", {}).get("zero_pair_return"))
                for record in local_turns
            ]
        )
        if local_turns
        else 1.0,
        "by_action": by_action,
        "residual_yield_by_action_agnostic_truth": dict(residual_yield),
        "credited_residual_yield": dict(credited_residual_yield),
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _flatten(prefix: str, value: Mapping[str, Any], output: dict[str, Any]) -> None:
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, Mapping):
            # Per-node/edge details stay in JSON; CSV contains scalar curves only.
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = item


def _write_flat_csv(path: Path, records: list[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, Mapping):
                _flatten(key, value, row)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_results(summary: Mapping[str, Any], turns: list[Mapping[str, Any]]) -> str:
    final = summary.get("final", {})
    mode_diagnostics = summary.get("mode_diagnostics", {})
    anchor_diagnostics = summary.get("anchor_diagnostics", {})
    arm_diagnostics = summary.get("arm_diagnostics", {})
    lines = [
        "# Medical extraction results",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Turns: {summary['turns_completed']}",
        f"- Recovered graph: {summary['final_graph_nodes']} nodes / "
        f"{summary['final_graph_relation_triples']} relation triples",
        f"- Node precision: {float(final.get('node_precision', 0.0)):.4f}",
        f"- Node recall: {float(final.get('node_recall', 0.0)):.4f}",
        "- Directed edge-pair precision: "
        f"{float(final.get('edge_pair_precision', 0.0)):.4f}",
        f"- Directed edge-pair recall: {float(final.get('edge_pair_recall', 0.0)):.4f}",
        f"- Node TSC (rank): {float(final.get('node_tsc_rank', 0.0)):.4f}",
        f"- Edge TSC (rank): {float(final.get('edge_tsc_rank', 0.0)):.4f}",
        "- AUTSC, node rank: "
        f"{float(summary.get('trajectory', {}).get('au_node_tsc_rank', 0.0)):.4f}",
        "- AUTSC, edge rank: "
        f"{float(summary.get('trajectory', {}).get('au_edge_tsc_rank', 0.0)):.4f}",
        f"- Mean HTSN: {float(summary['mean_htsn']):.4f}",
        f"- Mean count novelty: {float(summary['mean_count_novelty']):.4f}",
        "- Post-seed explore/exploit: "
        f"{int(mode_diagnostics.get('post_seed_explore_turns', 0))}/"
        f"{int(mode_diagnostics.get('post_seed_exploit_turns', 0))}",
        "- Exploit query anchor adherence: "
        f"{float(anchor_diagnostics.get('query_anchor_adherence_rate', 0.0)):.4f}",
        "- Exploit response anchor adherence: "
        f"{float(anchor_diagnostics.get('response_anchor_adherence_rate', 0.0)):.4f}",
        "",
        "| Turn | Mode | Reason | Anchor | HTSN | Raw reward | Effective FEWA | "
        "Anchor response | Node TSC | Edge TSC |",
        "|---:|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for record in turns:
        batch = record["batch"]
        evaluation = record["evaluation"]
        effective_reward = record.get("effective_fewa_reward")
        effective_text = (
            f"{float(effective_reward):.4f}" if effective_reward is not None else "-"
        )
        response_adherent = record.get("anchor_diagnostics", {}).get(
            "anchor_response_adherent"
        )
        response_text = (
            "yes" if response_adherent is True else "no" if response_adherent is False else "-"
        )
        lines.append(
            f"| {record['turn']} | {record['mode']} | {record['decision_reason']} | "
            f"{record['anchor'] or '-'} | {batch['htsn_combined']:.4f} | "
            f"{float(record.get('raw_batch_reward', record.get('reward', 0.0))):.4f} | "
            f"{effective_text} | {response_text} | "
            f"{evaluation['node_tsc_rank']:.4f} | {evaluation['edge_tsc_rank']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Arm-policy diagnostics",
            "",
            f"- Policy: `{arm_diagnostics.get('policy', 'unknown')}`",
            f"- Epochs: {int(arm_diagnostics.get('epochs', 0))}",
            "- Unique admitted / selected arms: "
            f"{int(arm_diagnostics.get('unique_admitted_arms', 0))} / "
            f"{int(arm_diagnostics.get('unique_selected_arms', 0))}",
            "- Selected arms: "
            f"{', '.join(arm_diagnostics.get('selected_arms', [])) or '-'}",
            "- Mean admission-to-first-query delay: "
            f"{float(arm_diagnostics.get('mean_admission_to_first_query_delay', 0.0)):.4f}",
            "- Unqueried admitted slots: "
            f"{int(arm_diagnostics.get('unqueried_admitted_slots', 0))}",
            "- Mean epoch arm-set turnover: "
            f"{float(arm_diagnostics.get('mean_epoch_arm_set_turnover', 0.0)):.4f}",
        ]
    )
    per_epoch = list(arm_diagnostics.get("per_epoch", []))
    if per_epoch:
        lines.extend(
            [
                "",
                "| Epoch | Active arms | Admission-to-first-query delay |",
                "|---:|---|---|",
            ]
        )
        for epoch in per_epoch:
            active = ", ".join(str(arm) for arm in epoch.get("active_arms", []))
            delays = ", ".join(
                f"{arm}: {'unqueried' if delay is None else delay}"
                for arm, delay in epoch.get(
                    "admission_to_first_query_delay", {}
                ).items()
            )
            lines.append(
                f"| {epoch.get('epoch_id', '-')} | {active or '-'} | {delays or '-'} |"
            )
    lines.extend(
        [
            "",
            "## Protocol acceptance",
            "",
            "| Criterion | Value | Threshold | Passed |",
            "|---|---:|---:|---|",
        ]
    )
    acceptance = summary.get("formal_acceptance", {})
    for name in (
        "query_unique_rate",
        "zero_gain_rate",
        "max_consecutive_zero_gain",
        "max_consecutive_meaningful_zero_gain",
        "anchor_query_adherence_rate",
        "both_post_seed_modes_observed",
        "nonempty_main_response_rate",
    ):
        item = acceptance.get(name, {})
        lines.append(
            f"| {name} | {float(item.get('value', 0.0)):.4f} | "
            f"{float(item.get('threshold', 0.0)):.4f} | "
            f"{'yes' if item.get('passed', False) else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"All formal acceptance criteria passed: "
            f"{'yes' if acceptance.get('all_passed', False) else 'no'}.",
            "",
            "HTSN is scored against the graph snapshot before each batch. RTSN is computed after "
            "the merge, and their difference is reported as retrospective gain. GraphRAG truth "
            "does not expose a structured relation type, so truth edge metrics use "
            "directed endpoint pairs; extraction novelty still uses relation triples.",
            "",
        ]
    )
    return "\n".join(lines)
