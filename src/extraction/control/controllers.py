"""Sequential query controllers used by the extraction pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .admission import TopologyPlackettLuceAdmission, is_valid_arm


@dataclass
class EpochFewaController:
    """A seeded, epoch-frozen FEWA-style rotting-bandit controller.

    Each entity is an arm and its observed reward is historical sensitive mass per
    candidate atom. TS-PL samples a set of arms that remains fixed for one exploit
    pull per admitted arm. The filtering pass uses recent windows of 1, 2, 4, ...
    pulls, which emphasizes decaying marginal yield.
    """

    delta: float = 0.05
    max_arms: int = 20
    seed: int = 42
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)

    def _refresh_arms(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float],
    ) -> bool:
        if self.active_arms and turn <= self.epoch_end_turn:
            return False

        self.epoch_id += 1
        available_arms = tuple(sorted(set(str(arm) for arm in available)))
        admission = TopologyPlackettLuceAdmission(seed=self.seed).admit(
            available_arms,
            priors,
            {arm: len(values) for arm, values in self.rewards.items()},
            max_arms=self.max_arms,
            epoch_id=self.epoch_id,
        )
        self.active_arms = admission.active_arms
        admission_record: dict[str, object] = admission.to_dict()
        # One exploit-pull opportunity per admitted arm defines the epoch.
        realized_epoch_length = max(1, len(self.active_arms))

        self.epoch_end_turn = turn + realized_epoch_length - 1
        self.active_priors = {
            arm: float(priors.get(arm, 0.0)) for arm in self.active_arms
        }
        for arm in self.active_arms:
            self.rewards.setdefault(arm, [])
        admission_record["exploit_pull_start"] = turn
        admission_record["exploit_pull_end"] = self.epoch_end_turn
        self.admission_history.append(admission_record)
        return True

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
    ) -> str | None:
        priors = priors or {}
        refreshed = self._refresh_arms(available, turn, priors)
        if not self.active_arms:
            self.selection_history.append(
                {
                    "exploit_pull": turn,
                    "epoch_id": self.epoch_id,
                    "epoch_refreshed": refreshed,
                    "selected_arm": None,
                    "reason": "no_active_arm",
                    "active_arms": [],
                }
            )
            return None

        # Topology is frozen with the admitted set. It is used only for
        # cold-start ordering and deterministic ties; FEWA's filtering signal
        # remains the attributable reward history.
        epoch_priors = self.active_priors

        unpulled = [arm for arm in self.active_arms if not self.rewards[arm]]
        if unpulled:
            selected = min(
                unpulled, key=lambda arm: (-epoch_priors.get(arm, 0.0), arm)
            )
            self.selection_history.append(
                {
                    "exploit_pull": turn,
                    "epoch_id": self.epoch_id,
                    "epoch_refreshed": refreshed,
                    "selected_arm": selected,
                    "reason": "globally_unpulled_topology_prior",
                    "active_arms": list(self.active_arms),
                    "unpulled_arms": list(unpulled),
                    "filter_windows": [],
                }
            )
            return selected

        active = list(self.active_arms)
        minimum_pulls = min(len(self.rewards[arm]) for arm in active)
        total_pulls = max(1, sum(len(values) for values in self.rewards.values()))
        window = 1
        filter_windows: list[dict[str, object]] = []
        while window <= minimum_pulls and len(active) > 1:
            means = {
                arm: sum(self.rewards[arm][-window:]) / window for arm in active
            }
            best_mean = max(means.values())
            confidence = math.sqrt(
                2.0
                * math.log(
                    max(2.0, 2.0 * total_pulls * len(active) / max(self.delta, 1e-12))
                )
                / window
            )
            retained = [
                arm for arm in active if means[arm] >= best_mean - 2.0 * confidence
            ]
            filter_windows.append(
                {
                    "window": window,
                    "means": means,
                    "best_mean": best_mean,
                    "confidence": confidence,
                    "input_arms": list(active),
                    "retained_arms": list(retained),
                }
            )
            active = retained
            window *= 2

        # FEWA balances pulls among statistically plausible arms. Recent reward
        # and lexical order make ties reproducible.
        selected = min(
            active,
            key=lambda arm: (
                len(self.rewards[arm]),
                -self.rewards[arm][-1],
                -epoch_priors.get(arm, 0.0),
                arm,
            ),
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": refreshed,
                "selected_arm": selected,
                "reason": "fewa_filtered_least_pulled",
                "active_arms": list(self.active_arms),
                "retained_arms": list(active),
                "filter_windows": filter_windows,
            }
        )
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is None:
            return
        self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": "topology_pl_fewa",
            "epoch_length_semantics": "realized_active_set_size",
            "current_realized_epoch_length": len(self.active_arms),
            "delta": self.delta,
            "max_arms": self.max_arms,
            "admission_policy": "topology_plackett_luce",
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_priors": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class FreshAnchorController:
    """Uniformly sweep the positive-topology global frontier without repeats.

    BNRR sensitivity is used only as a binary eligibility gate.  Its magnitude,
    observed rewards, and epoch state do not affect selection while a globally
    fresh eligible anchor exists.  Once that frontier is exhausted, selection
    falls back to the globally least-pulled eligible anchors.
    """

    seed: int = 42
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
    ) -> str | None:
        priors = priors or {}
        eligible = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm) and float(priors.get(arm, 0.0)) > 0.0
        ]
        self.epoch_id += 1
        fresh = [arm for arm in eligible if not self.rewards.get(arm)]
        pool = fresh
        if fresh:
            reason = "uniform_fresh"
        elif eligible:
            minimum_pulls = min(len(self.rewards.get(arm, [])) for arm in eligible)
            pool = [
                arm
                for arm in eligible
                if len(self.rewards.get(arm, [])) == minimum_pulls
            ]
            reason = "uniform_least_pulled"
        else:
            reason = "no_active_arm"

        selected = self._rng.choice(pool) if pool else None
        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(priors.get(selected, 0.0))} if selected else {}
        )
        self.epoch_end_turn = turn
        self.admission_history.append(
            {
                "policy": "uniform_global_fresh",
                "epoch_id": self.epoch_id,
                "eligible_arm_count": len(eligible),
                "eligibility_gate": "bnrr_sensitivity>0",
                "active_arms": list(self.active_arms),
                "candidates": [
                    {
                        "arm": arm,
                        "sensitivity": float(priors.get(arm, 0.0)),
                        "pull_count": len(self.rewards.get(arm, [])),
                    }
                    for arm in eligible
                ],
                "draws": [],
                "fallback_reason": "none",
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "reason": reason,
                "active_arms": list(self.active_arms),
                "fresh_arm_count": len(fresh),
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": "uniform_fresh",
            "selection_policy": "uniform",
            "eligibility_gate": "bnrr_sensitivity>0",
            "freshness_scope": "global",
            "reward_used_for_selection": False,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_priors": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class AgeaHubController:
    """Expose AGEA's hub-oriented allocator behind an explicit anchor contract.

    The original AGEA implementation first ranks eligible entities by
    ``degree / (pull_count + 1)``, keeps six candidates, and then performs a
    degree-weighted target draw.  This controller makes that effective target
    draw auditable and removes the query generator's hidden second sampling
    step, so allocator ablations share the same fixed-anchor query operator.
    Reward magnitudes never affect selection; only historical pull counts do.
    """

    seed: int = 42
    candidate_k: int = 6
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self._rng = random.Random(self.seed)

    @staticmethod
    def _pull_cap(degree: int) -> int:
        if degree >= 100:
            return 10
        if degree >= 50:
            return 5
        if degree >= 20:
            return 3
        return 1

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
        topology: Mapping[str, Mapping[str, float | int]] | None = None,
    ) -> str | None:
        del priors
        topology = topology or {}
        valid_available = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm)
        ]
        missing_topology = [arm for arm in valid_available if arm not in topology]
        if missing_topology:
            raise ValueError(
                f"Missing topology features for eligible arms: {missing_topology[:3]}"
            )
        eligible = [
            arm
            for arm in valid_available
            if int(topology[arm].get("degree", 0)) > 0
            and len(self.rewards.get(arm, []))
            < self._pull_cap(int(topology[arm].get("degree", 0)))
        ]
        ranked = sorted(
            eligible,
            key=lambda arm: (
                -int(topology[arm]["degree"])
                / (len(self.rewards.get(arm, [])) + 1),
                arm,
            ),
        )
        pool = ranked[: self.candidate_k]
        weights = [
            max(math.log(int(topology[arm]["degree"]) + 1), 1.0)
            / (1.0 + len(self.rewards.get(arm, [])) * 0.05)
            for arm in pool
        ]
        selected = self._rng.choices(pool, weights=weights, k=1)[0] if pool else None

        self.epoch_id += 1
        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(topology[selected]["degree"])} if selected else {}
        )
        self.epoch_end_turn = turn
        candidates = [
            {
                "arm": arm,
                "degree": int(topology[arm]["degree"]),
                "pull_count": len(self.rewards.get(arm, [])),
                "pull_cap": self._pull_cap(int(topology[arm]["degree"])),
                "priority": int(topology[arm]["degree"])
                / (len(self.rewards.get(arm, [])) + 1),
                "draw_weight": weight,
            }
            for arm, weight in zip(pool, weights, strict=True)
        ]
        common = {
            "policy": "agea_hub",
            "epoch_id": self.epoch_id,
            "eligible_arm_count": len(eligible),
            "candidate_k": self.candidate_k,
            "candidate_pool_size": len(pool),
            "eligibility_gate": "observed_degree>0_and_agea_pull_cap",
            "active_arms": list(self.active_arms),
            "candidates": candidates,
        }
        self.admission_history.append(
            {
                **common,
                "draws": [],
                "fallback_reason": "none" if selected else "no_eligible_hub",
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                **common,
                "exploit_pull": turn,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "reason": "agea_degree_weighted_hub" if selected else "no_active_arm",
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": "agea_hub",
            "selection_policy": "agea_topk_priority_then_degree_weighted_draw",
            "candidate_k": self.candidate_k,
            "eligibility_gate": "observed_degree>0_and_agea_pull_cap",
            "reward_used_for_selection": False,
            "pull_count_used_for_selection": True,
            "hidden_query_generator_target_draw": False,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_degrees": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class FrontierAlternatingFreshController:
    """Alternate a tie-safe incident frontier slot with full-pool coverage.

    Odd exploit pulls sample uniformly from the complete minimum-sensitivity
    tie group; even pulls sample uniformly from all globally fresh eligible
    anchors. Ties are never split by lexical order. Full-pool slots keep every
    positive-sensitivity anchor reachable, and rewards never affect selection.
    """

    seed: int = 42
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
    ) -> str | None:
        priors = priors or {}
        eligible = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm) and float(priors.get(arm, 0.0)) > 0.0
        ]
        self.epoch_id += 1
        fresh = [arm for arm in eligible if not self.rewards.get(arm)]
        if fresh:
            pool = fresh
            freshness_reason = "fresh"
        elif eligible:
            minimum_pulls = min(len(self.rewards.get(arm, [])) for arm in eligible)
            pool = [
                arm
                for arm in eligible
                if len(self.rewards.get(arm, [])) == minimum_pulls
            ]
            freshness_reason = "least_pulled"
        else:
            pool = []
            freshness_reason = "no_active_arm"

        minimum_sensitivity: float | None = None
        frontier_pool: list[str] = []
        if pool:
            minimum_sensitivity = min(float(priors.get(arm, 0.0)) for arm in pool)
            frontier_pool = [
                arm
                for arm in pool
                if float(priors.get(arm, 0.0)) == minimum_sensitivity
            ]
        frontier_forced = bool(pool) and self.epoch_id % 2 == 1
        selection_pool = frontier_pool if frontier_forced else pool
        selected = self._rng.choice(selection_pool) if selection_pool else None
        scope = "minimum_sensitivity_tie_group" if frontier_forced else "all"
        reason = f"frontier_alternating_{freshness_reason}_{scope}"

        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(priors.get(selected, 0.0))} if selected else {}
        )
        self.epoch_end_turn = turn
        common = {
            "frontier_forced": frontier_forced,
            "selection_scope": scope,
            "base_pool_size": len(pool),
            "frontier_pool_size": len(frontier_pool),
            "frontier_pool_share": len(frontier_pool) / len(pool) if pool else 0.0,
            "minimum_sensitivity": minimum_sensitivity,
            "tie_group_preserved": True,
        }
        self.admission_history.append(
            {
                "policy": "frontier_alternating_global_fresh",
                "epoch_id": self.epoch_id,
                "eligible_arm_count": len(eligible),
                "eligibility_gate": "bnrr_sensitivity>0",
                **common,
                "active_arms": list(self.active_arms),
                "candidates": [
                    {
                        "arm": arm,
                        "sensitivity": float(priors.get(arm, 0.0)),
                        "pull_count": len(self.rewards.get(arm, [])),
                    }
                    for arm in eligible
                ],
                "draws": [],
                "fallback_reason": "none",
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "reason": reason,
                "active_arms": list(self.active_arms),
                "fresh_arm_count": len(fresh),
                **common,
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": "frontier_alternating_fresh",
            "selection_policy": "alternate_minimum_tie_group_and_all_uniform",
            "eligibility_gate": "bnrr_sensitivity>0",
            "freshness_scope": "global",
            "reward_used_for_selection": False,
            "tie_group_preserved": True,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_priors": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class MinimumDegreeAlternatingFreshController:
    """Alternate a minimum-degree frontier slot with uniform fresh coverage.

    Odd exploit pulls sample uniformly from the complete minimum observed-degree
    tie group. Even pulls sample uniformly from every globally fresh eligible
    anchor. The controller reads no BNRR value and never uses observed rewards
    for selection; reward lists only record whether an anchor has been pulled.
    """

    seed: int = 42
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
        topology: Mapping[str, Mapping[str, float | int]] | None = None,
    ) -> str | None:
        del priors
        topology = topology or {}
        valid_available = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm)
        ]
        missing_topology = [arm for arm in valid_available if arm not in topology]
        if missing_topology:
            raise ValueError(
                f"Missing topology features for eligible arms: {missing_topology[:3]}"
            )
        eligible = [
            arm
            for arm in valid_available
            if int(topology[arm].get("degree", 0)) > 0
        ]

        self.epoch_id += 1
        fresh = [arm for arm in eligible if not self.rewards.get(arm)]
        if fresh:
            pool = fresh
            freshness_reason = "fresh"
        elif eligible:
            minimum_pulls = min(len(self.rewards.get(arm, [])) for arm in eligible)
            pool = [
                arm
                for arm in eligible
                if len(self.rewards.get(arm, [])) == minimum_pulls
            ]
            freshness_reason = "least_pulled"
        else:
            pool = []
            freshness_reason = "no_active_arm"

        minimum_degree: int | None = None
        degree_pool: list[str] = []
        if pool:
            minimum_degree = min(int(topology[arm]["degree"]) for arm in pool)
            degree_pool = [
                arm
                for arm in pool
                if int(topology[arm]["degree"]) == minimum_degree
            ]
        degree_forced = bool(pool) and self.epoch_id % 2 == 1
        selection_pool = degree_pool if degree_forced else pool
        selected = self._rng.choice(selection_pool) if selection_pool else None
        scope = "minimum_degree_tie_group" if degree_forced else "all"
        reason = f"minimum_degree_alternating_{freshness_reason}_{scope}"

        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(topology[selected]["degree"])} if selected else {}
        )
        self.epoch_end_turn = turn
        common = {
            "degree_forced": degree_forced,
            "selection_scope": scope,
            "base_pool_size": len(pool),
            "degree_pool_size": len(degree_pool),
            "degree_pool_share": len(degree_pool) / len(pool) if pool else 0.0,
            "minimum_degree": minimum_degree,
            "tie_group_preserved": True,
        }
        self.admission_history.append(
            {
                "policy": "minimum_degree_alternating_fresh",
                "epoch_id": self.epoch_id,
                "eligible_arm_count": len(eligible),
                "eligibility_gate": "observed_degree>0",
                **common,
                "active_arms": list(self.active_arms),
                "candidates": [
                    {
                        "arm": arm,
                        "degree": int(topology[arm]["degree"]),
                        "pull_count": len(self.rewards.get(arm, [])),
                    }
                    for arm in eligible
                ],
                "draws": [],
                "fallback_reason": "none",
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "reason": reason,
                "active_arms": list(self.active_arms),
                "fresh_arm_count": len(fresh),
                **common,
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": "minimum_degree_alternating_fresh",
            "selection_policy": "alternate_minimum_degree_tie_group_and_all_uniform",
            "eligibility_gate": "observed_degree>0",
            "freshness_scope": "global",
            "reward_used_for_selection": False,
            "tie_group_preserved": True,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_degrees": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class DegreeConditionedBnrrFreshController:
    """Preserve the degree marginal while testing within-degree BNRR order.

    A uniform pivot from the global fresh (or least-pulled) pool selects an
    exact observed-degree stratum.  Low and high treatments sample from the
    corresponding complete raw-BNRR tie group in that stratum; uniform and
    within-degree shuffled variants are contemporaneous controls. The hybrid
    variant uses this conditional high-BNRR rule before a frozen global-turn
    switch, then alternates a complete minimum-degree frontier slot with a
    global-fresh coverage slot. Independent unit-interval RNG streams keep
    same-state pivot and selection draws aligned across all variants.
    """

    policy_name: str = "degree_conditioned_bnrr_low_fresh"
    variant: str = "bnrr_low"
    schedule_profile: str = "linear"
    schedule_total_turns: int = 100
    schedule_start_fraction: float = 0.20
    schedule_end_fraction: float = 0.70
    seed: int = 42
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _pivot_rng: random.Random = field(init=False, repr=False)
    _selection_rng: random.Random = field(init=False, repr=False)
    _dfa_epoch_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        valid_variants = {
            "bnrr_high",
            "bnrr_high_scheduled",
            "bnrr_high_then_dfa",
            "bnrr_low",
            "shuffled_bnrr_high",
            "shuffled_bnrr_high_scheduled",
            "shuffled_bnrr_low",
            "uniform",
        }
        if self.variant not in valid_variants:
            raise ValueError(
                f"variant must be one of {sorted(valid_variants)}"
            )
        if self.schedule_profile not in {"hard_switch", "linear", "staged"}:
            raise ValueError(
                "schedule_profile must be 'hard_switch', 'linear', or 'staged'"
            )
        if self.schedule_total_turns <= 0:
            raise ValueError("schedule_total_turns must be positive")
        if not 0.0 <= self.schedule_start_fraction < self.schedule_end_fraction <= 1.0:
            raise ValueError(
                "schedule fractions must satisfy "
                "0 <= start < end <= 1"
            )
        self._pivot_rng = random.Random(self.seed)
        self._selection_rng = random.Random(self.seed + 1_000_003)

    @staticmethod
    def _pick(pool: Iterable[str], draw: float) -> str | None:
        ordered = sorted(pool)
        if not ordered:
            return None
        index = min(int(draw * len(ordered)), len(ordered) - 1)
        return ordered[index]

    def _shuffled_scores(
        self,
        degree_pool: list[str],
        raw_scores: Mapping[str, float],
    ) -> dict[str, float]:
        values = [raw_scores[arm] for arm in sorted(degree_pool)]
        shuffle_rng = random.Random(
            (self.seed + 1) * 10_000_019 + self.epoch_id
        )
        shuffle_rng.shuffle(values)
        return dict(zip(sorted(degree_pool), values))

    @staticmethod
    def _upper_tail(
        scores: Mapping[str, float], requested_fraction: float
    ) -> tuple[list[str], float | None]:
        """Return a complete upper-tail threshold group without splitting ties.

        A requested fraction of zero means the complete maximum-score tie group;
        one means the full stratum. Intermediate fractions choose the score at
        ``ceil(fraction * n)`` in descending order and include every arm tied at
        that cutoff.
        """

        if not scores:
            return [], None
        fraction = min(1.0, max(0.0, float(requested_fraction)))
        ordered_scores = sorted(scores.values(), reverse=True)
        if fraction <= 0.0:
            cutoff = ordered_scores[0]
        elif fraction >= 1.0:
            cutoff = ordered_scores[-1]
        else:
            target_size = max(1, math.ceil(fraction * len(ordered_scores)))
            cutoff = ordered_scores[target_size - 1]
        return (
            [arm for arm in sorted(scores) if scores[arm] >= cutoff],
            cutoff,
        )

    def _schedule(
        self, global_turn: int | None
    ) -> tuple[float, float, str]:
        if global_turn is None:
            raise ValueError("scheduled BNRR variants require global_turn")
        if global_turn <= 0:
            raise ValueError("global_turn must be positive")
        progress = min(1.0, global_turn / self.schedule_total_turns)
        if self.schedule_profile == "hard_switch":
            if progress <= self.schedule_end_fraction:
                return progress, 0.0, "maximum"
            return progress, 1.0, "degree_frontier"
        if self.schedule_profile == "linear":
            if progress <= self.schedule_start_fraction:
                return progress, 0.0, "maximum"
            if progress >= self.schedule_end_fraction:
                return progress, 1.0, "uniform"
            span = self.schedule_end_fraction - self.schedule_start_fraction
            fraction = (progress - self.schedule_start_fraction) / span
            return progress, fraction, "annealing"
        if progress <= 1.0 / 3.0:
            return progress, 0.0, "maximum"
        if progress <= 0.50:
            return progress, 0.25, "top_quartile"
        if progress <= 2.0 / 3.0:
            return progress, 0.50, "top_half"
        return progress, 1.0, "uniform"

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
        topology: Mapping[str, Mapping[str, float | int]] | None = None,
        global_turn: int | None = None,
    ) -> str | None:
        del priors
        topology = topology or {}
        valid_available = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm)
        ]
        missing_topology = [arm for arm in valid_available if arm not in topology]
        if missing_topology:
            raise ValueError(
                f"Missing topology features for eligible arms: {missing_topology[:3]}"
            )
        for arm in valid_available:
            degree = int(topology[arm].get("degree", 0))
            bnrr = float(topology[arm].get("bnrr", 0.0))
            if degree > 0 and (not math.isfinite(bnrr) or bnrr <= 0.0):
                raise ValueError(f"Invalid BNRR for eligible arm {arm!r}: {bnrr}")
        eligible = [
            arm
            for arm in valid_available
            if int(topology[arm].get("degree", 0)) > 0
        ]

        self.epoch_id += 1
        fresh = [arm for arm in eligible if not self.rewards.get(arm)]
        minimum_pull_count: int | None = None
        if fresh:
            pool = fresh
            freshness_reason = "fresh"
        elif eligible:
            minimum_pull_count = min(
                len(self.rewards.get(arm, [])) for arm in eligible
            )
            pool = [
                arm
                for arm in eligible
                if len(self.rewards.get(arm, [])) == minimum_pull_count
            ]
            freshness_reason = "least_pulled"
        else:
            pool = []
            freshness_reason = "no_active_arm"

        pivot_u = self._pivot_rng.random() if pool else None
        selection_u = self._selection_rng.random() if pool else None
        pivot = self._pick(pool, float(pivot_u)) if pivot_u is not None else None
        pivot_degree = int(topology[pivot]["degree"]) if pivot else None
        degree_pool = (
            [
                arm
                for arm in pool
                if int(topology[arm]["degree"]) == pivot_degree
            ]
            if pivot is not None
            else []
        )
        raw_scores = {arm: float(topology[arm]["bnrr"]) for arm in degree_pool}
        observed_levels = sorted(set(raw_scores.values()))
        minimum_bnrr = min(observed_levels) if observed_levels else None
        minimum_pool = [
            arm for arm in degree_pool if raw_scores[arm] == minimum_bnrr
        ]
        maximum_bnrr = max(observed_levels) if observed_levels else None
        maximum_pool = [
            arm for arm in degree_pool if raw_scores[arm] == maximum_bnrr
        ]
        shuffled_scores = self._shuffled_scores(degree_pool, raw_scores)
        shuffled_minimum = min(shuffled_scores.values()) if shuffled_scores else None
        shuffled_minimum_pool = [
            arm
            for arm in degree_pool
            if shuffled_scores[arm] == shuffled_minimum
        ]
        shuffled_maximum = max(shuffled_scores.values()) if shuffled_scores else None
        shuffled_maximum_pool = [
            arm
            for arm in degree_pool
            if shuffled_scores[arm] == shuffled_maximum
        ]

        global_minimum_degree = (
            min(int(topology[arm]["degree"]) for arm in pool) if pool else None
        )
        global_minimum_degree_pool = [
            arm
            for arm in pool
            if int(topology[arm]["degree"]) == global_minimum_degree
        ]

        scheduled_variant = self.variant in {
            "bnrr_high_scheduled",
            "bnrr_high_then_dfa",
            "shuffled_bnrr_high_scheduled",
        }
        schedule_progress: float | None = None
        requested_tail_fraction: float | None = None
        schedule_phase: str | None = None
        scheduled_raw_pool: list[str] = []
        scheduled_raw_cutoff: float | None = None
        scheduled_shuffled_pool: list[str] = []
        scheduled_shuffled_cutoff: float | None = None
        if scheduled_variant:
            (
                schedule_progress,
                requested_tail_fraction,
                schedule_phase,
            ) = self._schedule(global_turn)
            scheduled_raw_pool, scheduled_raw_cutoff = self._upper_tail(
                raw_scores, requested_tail_fraction
            )
            (
                scheduled_shuffled_pool,
                scheduled_shuffled_cutoff,
            ) = self._upper_tail(shuffled_scores, requested_tail_fraction)

        paired_bnrr_low_arm = (
            self._pick(minimum_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_bnrr_high_arm = (
            self._pick(maximum_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_uniform_arm = (
            self._pick(degree_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_shuffled_low_arm = (
            self._pick(shuffled_minimum_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_shuffled_high_arm = (
            self._pick(shuffled_maximum_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_scheduled_raw_arm = (
            self._pick(scheduled_raw_pool, float(selection_u))
            if selection_u is not None and scheduled_variant
            else None
        )
        paired_scheduled_shuffled_arm = (
            self._pick(scheduled_shuffled_pool, float(selection_u))
            if selection_u is not None and scheduled_variant
            else None
        )
        hybrid_degree_forced = False
        hybrid_selection_pool: list[str] = []
        paired_hybrid_arm: str | None = None
        if self.variant == "bnrr_high_then_dfa":
            if schedule_phase == "degree_frontier":
                self._dfa_epoch_id += 1
                hybrid_degree_forced = self._dfa_epoch_id % 2 == 1
                hybrid_selection_pool = (
                    global_minimum_degree_pool if hybrid_degree_forced else pool
                )
                paired_hybrid_arm = (
                    self._pick(hybrid_selection_pool, float(selection_u))
                    if selection_u is not None
                    else None
                )
            else:
                hybrid_selection_pool = scheduled_raw_pool
                paired_hybrid_arm = paired_scheduled_raw_arm
        selected_by_variant = {
            "bnrr_high": paired_bnrr_high_arm,
            "bnrr_high_scheduled": paired_scheduled_raw_arm,
            "bnrr_high_then_dfa": paired_hybrid_arm,
            "bnrr_low": paired_bnrr_low_arm,
            "shuffled_bnrr_high": paired_shuffled_high_arm,
            "shuffled_bnrr_high_scheduled": paired_scheduled_shuffled_arm,
            "shuffled_bnrr_low": paired_shuffled_low_arm,
            "uniform": paired_uniform_arm,
        }
        selected = selected_by_variant[self.variant]
        scope = {
            "bnrr_high": "maximum_bnrr_tie_group",
            "bnrr_high_scheduled": "scheduled_raw_bnrr_upper_tail",
            "bnrr_high_then_dfa": (
                "minimum_degree_tie_group"
                if hybrid_degree_forced
                else (
                    "global_fresh_uniform"
                    if schedule_phase == "degree_frontier"
                    else "maximum_bnrr_tie_group"
                )
            ),
            "bnrr_low": "minimum_bnrr_tie_group",
            "shuffled_bnrr_high": "maximum_shuffled_bnrr_tie_group",
            "shuffled_bnrr_high_scheduled": (
                "scheduled_shuffled_bnrr_upper_tail"
            ),
            "shuffled_bnrr_low": "minimum_shuffled_bnrr_tie_group",
            "uniform": "degree_stratum_uniform",
        }[self.variant]
        reason = f"{self.policy_name}_{freshness_reason}_{scope}"

        minimum_tie_share = (
            len(minimum_pool) / len(degree_pool) if degree_pool else 0.0
        )
        maximum_tie_share = (
            len(maximum_pool) / len(degree_pool) if degree_pool else 0.0
        )
        low_policy_tv = 1.0 - minimum_tie_share if degree_pool else 0.0
        high_policy_tv = 1.0 - maximum_tie_share if degree_pool else 0.0
        scheduled_raw_share = (
            len(scheduled_raw_pool) / len(degree_pool) if degree_pool else 0.0
        )
        scheduled_shuffled_share = (
            len(scheduled_shuffled_pool) / len(degree_pool)
            if degree_pool
            else 0.0
        )
        direction_policy_tv = {
            "bnrr_high": high_policy_tv,
            "bnrr_high_scheduled": (
                1.0 - scheduled_raw_share if degree_pool else 0.0
            ),
            "bnrr_high_then_dfa": (
                1.0 - len(hybrid_selection_pool) / len(pool)
                if schedule_phase == "degree_frontier" and pool
                else (1.0 - scheduled_raw_share if degree_pool else 0.0)
            ),
            "bnrr_low": low_policy_tv,
            "shuffled_bnrr_high": high_policy_tv,
            "shuffled_bnrr_high_scheduled": (
                1.0 - scheduled_shuffled_share if degree_pool else 0.0
            ),
            "shuffled_bnrr_low": low_policy_tv,
            "uniform": 0.0,
        }[self.variant]
        selected_schedule_pool = (
            hybrid_selection_pool
            if self.variant == "bnrr_high_then_dfa"
            else (
                scheduled_shuffled_pool
                if self.variant == "shuffled_bnrr_high_scheduled"
                else scheduled_raw_pool
            )
        )
        selected_schedule_cutoff = (
            None
            if self.variant == "bnrr_high_then_dfa"
            and schedule_phase == "degree_frontier"
            else (
                scheduled_shuffled_cutoff
                if self.variant == "shuffled_bnrr_high_scheduled"
                else scheduled_raw_cutoff
            )
        )
        common = {
            "selection_variant": self.variant,
            "selection_scope": scope,
            "base_pool_source": freshness_reason,
            "base_pool_size": len(pool),
            "minimum_pull_count": minimum_pull_count,
            "pivot_u": pivot_u,
            "pivot_arm": pivot,
            "pivot_degree": pivot_degree,
            "degree_stratum_size": len(degree_pool),
            "degree_stratum_share": len(degree_pool) / len(pool) if pool else 0.0,
            "selection_u": selection_u,
            "observed_unique_bnrr_count": len(observed_levels),
            "minimum_bnrr": minimum_bnrr,
            "minimum_bnrr_tie_group_size": len(minimum_pool),
            "minimum_bnrr_tie_group_share": minimum_tie_share,
            "maximum_bnrr": maximum_bnrr,
            "maximum_bnrr_tie_group_size": len(maximum_pool),
            "maximum_bnrr_tie_group_share": maximum_tie_share,
            "bnrr_informative": len(observed_levels) > 1,
            # Backward-compatible: the original field is the low-direction TV.
            "bnrr_policy_tv": low_policy_tv,
            "bnrr_low_policy_tv": low_policy_tv,
            "bnrr_high_policy_tv": high_policy_tv,
            "selected_direction_policy_tv": direction_policy_tv,
            "d1_stratum": pivot_degree == 1,
            "singleton_stratum": len(degree_pool) == 1,
            "scores_shuffled": self.variant.startswith("shuffled_bnrr_"),
            "hybrid_dfa_epoch_id": self._dfa_epoch_id,
            "hybrid_degree_forced": hybrid_degree_forced,
            "hybrid_minimum_degree": global_minimum_degree,
            "hybrid_degree_pool_size": len(global_minimum_degree_pool),
            "hybrid_selection_pool_size": len(hybrid_selection_pool),
            "schedule_profile": self.schedule_profile if scheduled_variant else None,
            "schedule_global_turn": global_turn if scheduled_variant else None,
            "schedule_total_turns": (
                self.schedule_total_turns if scheduled_variant else None
            ),
            "schedule_progress": schedule_progress,
            "schedule_phase": schedule_phase,
            "schedule_requested_tail_fraction": requested_tail_fraction,
            "schedule_tail_cutoff": selected_schedule_cutoff,
            "schedule_tail_pool_size": (
                len(selected_schedule_pool) if scheduled_variant else None
            ),
            "schedule_tail_pool_share": (
                len(selected_schedule_pool) / len(degree_pool)
                if scheduled_variant and degree_pool
                else None
            ),
            "schedule_raw_tail_pool_size": (
                len(scheduled_raw_pool) if scheduled_variant else None
            ),
            "schedule_shuffled_tail_pool_size": (
                len(scheduled_shuffled_pool) if scheduled_variant else None
            ),
            "schedule_capacity_matched": (
                len(scheduled_raw_pool) == len(scheduled_shuffled_pool)
                if scheduled_variant
                else None
            ),
            "schedule_active": (
                bool(degree_pool)
                and len(observed_levels) > 1
                and requested_tail_fraction is not None
                and requested_tail_fraction < 1.0
            ),
            "paired_bnrr_low_arm": paired_bnrr_low_arm,
            "paired_bnrr_high_arm": paired_bnrr_high_arm,
            "paired_uniform_arm": paired_uniform_arm,
            "paired_shuffled_bnrr_low_arm": paired_shuffled_low_arm,
            "paired_shuffled_bnrr_high_arm": paired_shuffled_high_arm,
            "paired_scheduled_raw_arm": paired_scheduled_raw_arm,
            "paired_scheduled_shuffled_arm": paired_scheduled_shuffled_arm,
            "paired_hybrid_arm": paired_hybrid_arm,
            "paired_scheduled_uniform_disagreement": (
                paired_scheduled_raw_arm != paired_uniform_arm
                if scheduled_variant
                else False
            ),
            "paired_scheduled_raw_shuffled_disagreement": (
                paired_scheduled_raw_arm != paired_scheduled_shuffled_arm
                if scheduled_variant
                else False
            ),
            "paired_bnrr_uniform_disagreement": (
                paired_bnrr_low_arm != paired_uniform_arm
            ),
            "paired_bnrr_high_uniform_disagreement": (
                paired_bnrr_high_arm != paired_uniform_arm
            ),
            "paired_bnrr_shuffled_disagreement": (
                paired_bnrr_low_arm != paired_shuffled_low_arm
            ),
            "paired_bnrr_high_shuffled_disagreement": (
                paired_bnrr_high_arm != paired_shuffled_high_arm
            ),
            "paired_bnrr_low_high_disagreement": (
                paired_bnrr_low_arm != paired_bnrr_high_arm
            ),
            "tie_group_preserved": True,
            "degree_marginal_preserved": (
                selected is None
                or (
                    self.variant == "bnrr_high_then_dfa"
                    and schedule_phase == "degree_frontier"
                    and selected in hybrid_selection_pool
                )
                or int(topology[selected]["degree"]) == pivot_degree
            ),
        }
        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(topology[selected]["bnrr"])} if selected else {}
        )
        self.epoch_end_turn = turn
        self.admission_history.append(
            {
                "policy": self.policy_name,
                "epoch_id": self.epoch_id,
                "eligible_arm_count": len(eligible),
                "eligibility_gate": "observed_degree>0_and_valid_bnrr",
                **common,
                "active_arms": list(self.active_arms),
                "candidates": [
                    {
                        "arm": arm,
                        "degree": int(topology[arm]["degree"]),
                        "bnrr": float(topology[arm]["bnrr"]),
                        "shuffled_selection_score": shuffled_scores.get(arm),
                        "in_pivot_degree_stratum": arm in degree_pool,
                        "in_scheduled_raw_tail": arm in scheduled_raw_pool,
                        "in_scheduled_shuffled_tail": (
                            arm in scheduled_shuffled_pool
                        ),
                        "pull_count": len(self.rewards.get(arm, [])),
                    }
                    for arm in eligible
                ],
                "draws": [
                    {"name": "pivot_u", "value": pivot_u},
                    {"name": "selection_u", "value": selection_u},
                ]
                if pool
                else [],
                "fallback_reason": (
                    "bnrr_constant_within_degree_stratum"
                    if degree_pool and len(observed_levels) <= 1
                    else (
                        "scheduled_uniform_phase"
                        if scheduled_variant
                        and requested_tail_fraction is not None
                        and requested_tail_fraction >= 1.0
                        else (
                            "scheduled_degree_frontier_phase"
                            if self.variant == "bnrr_high_then_dfa"
                            and schedule_phase == "degree_frontier"
                            else "none"
                        )
                    )
                ),
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "selected_degree": (
                    int(topology[selected]["degree"]) if selected else None
                ),
                "selected_bnrr": (
                    float(topology[selected]["bnrr"]) if selected else None
                ),
                "reason": reason,
                "active_arms": list(self.active_arms),
                "fresh_arm_count": len(fresh),
                **common,
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        scheduled = self.variant in {
            "bnrr_high_scheduled",
            "bnrr_high_then_dfa",
            "shuffled_bnrr_high_scheduled",
        }
        return {
            "policy": self.policy_name,
            "selection_policy": self.variant,
            "eligibility_gate": "observed_degree>0_and_valid_bnrr",
            "freshness_scope": "global",
            "degree_marginal": (
                "uniform_pivot_then_late_degree_frontier"
                if self.variant == "bnrr_high_then_dfa"
                else "uniform_pivot_from_base_pool"
            ),
            "reward_used_for_selection": False,
            "tie_group_preserved": True,
            "schedule_profile": (
                self.schedule_profile if scheduled else None
            ),
            "schedule_total_turns": (
                self.schedule_total_turns if scheduled else None
            ),
            "schedule_start_fraction": (
                self.schedule_start_fraction if scheduled else None
            ),
            "schedule_end_fraction": (
                self.schedule_end_fraction if scheduled else None
            ),
            "hybrid_dfa_epoch_id": self._dfa_epoch_id,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_bnrr": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class GlobalBnrrAnnealingFreshController:
    """Anneal a global BNRR rank threshold over exploit pulls.

    Unlike :class:`DegreeConditionedBnrrFreshController`, this policy does not
    draw a degree stratum before applying BNRR.  Raw BNRR can therefore change
    the selected degree marginal.  The normalized and within-degree shuffled
    variants isolate scale and score-identity effects under the same global
    fresh-pool schedule.
    """

    policy_name: str = "global_bnrr_raw_annealed_fresh"
    variant: str = "raw"
    schedule_total_turns: int = 100
    initial_pool_fraction: float = 0.10
    relax_exploit_fraction: float = 0.50
    seed: int = 42
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _selection_rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.variant not in {"raw", "normalized", "shuffled_raw"}:
            raise ValueError(
                "variant must be one of ['normalized', 'raw', 'shuffled_raw']"
            )
        if self.schedule_total_turns <= 0:
            raise ValueError("schedule_total_turns must be positive")
        if not 0.0 < self.initial_pool_fraction <= 1.0:
            raise ValueError("initial_pool_fraction must be in (0, 1]")
        if not 0.0 < self.relax_exploit_fraction <= 1.0:
            raise ValueError("relax_exploit_fraction must be in (0, 1]")
        self._selection_rng = random.Random(self.seed + 1_000_003)

    @staticmethod
    def _pick(pool: Iterable[str], draw: float) -> str | None:
        ordered = sorted(pool)
        if not ordered:
            return None
        index = min(int(draw * len(ordered)), len(ordered) - 1)
        return ordered[index]

    @staticmethod
    def _upper_tail(
        scores: Mapping[str, float], requested_fraction: float
    ) -> tuple[list[str], float | None]:
        return DegreeConditionedBnrrFreshController._upper_tail(
            scores, requested_fraction
        )

    @staticmethod
    def _degree_histogram(
        pool: Iterable[str],
        topology: Mapping[str, Mapping[str, float | int]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for arm in pool:
            degree = str(int(topology[arm]["degree"]))
            counts[degree] = counts.get(degree, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))

    def _shuffled_scores(
        self,
        pool: list[str],
        raw_scores: Mapping[str, float],
        topology: Mapping[str, Mapping[str, float | int]],
    ) -> dict[str, float]:
        shuffled: dict[str, float] = {}
        degrees = sorted({int(topology[arm]["degree"]) for arm in pool})
        for degree in degrees:
            degree_pool = sorted(
                arm for arm in pool if int(topology[arm]["degree"]) == degree
            )
            values = [raw_scores[arm] for arm in degree_pool]
            shuffle_rng = random.Random(
                (self.seed + 1) * 10_000_019
                + self.epoch_id * 1_009
                + degree * 9_176
            )
            shuffle_rng.shuffle(values)
            shuffled.update(dict(zip(degree_pool, values)))
        return shuffled

    def _schedule(self, exploit_pull: int) -> tuple[int, float, float, str]:
        if exploit_pull <= 0:
            raise ValueError("exploit_pull must be positive")
        relaxation_slots = max(
            1,
            math.ceil(self.relax_exploit_fraction * self.schedule_total_turns),
        )
        if relaxation_slots == 1:
            progress = 1.0
        else:
            progress = min(
                1.0,
                max(0.0, (exploit_pull - 1) / (relaxation_slots - 1)),
            )
        requested_fraction = self.initial_pool_fraction + (
            1.0 - self.initial_pool_fraction
        ) * progress
        if exploit_pull == 1 and requested_fraction < 1.0:
            phase = "initial_top_tail"
        elif requested_fraction < 1.0:
            phase = "annealing"
        else:
            phase = "uniform"
        return relaxation_slots, progress, requested_fraction, phase

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
        topology: Mapping[str, Mapping[str, float | int]] | None = None,
        global_turn: int | None = None,
    ) -> str | None:
        del priors
        topology = topology or {}
        valid_available = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm)
        ]
        missing_topology = [arm for arm in valid_available if arm not in topology]
        if missing_topology:
            raise ValueError(
                f"Missing topology features for eligible arms: {missing_topology[:3]}"
            )
        for arm in valid_available:
            degree = int(topology[arm].get("degree", 0))
            bnrr = float(topology[arm].get("bnrr", 0.0))
            if degree > 0 and (not math.isfinite(bnrr) or bnrr <= 0.0):
                raise ValueError(f"Invalid BNRR for eligible arm {arm!r}: {bnrr}")
        eligible = [
            arm
            for arm in valid_available
            if int(topology[arm].get("degree", 0)) > 0
        ]

        self.epoch_id += 1
        fresh = [arm for arm in eligible if not self.rewards.get(arm)]
        minimum_pull_count: int | None = None
        if fresh:
            pool = fresh
            freshness_reason = "fresh"
        elif eligible:
            minimum_pull_count = min(
                len(self.rewards.get(arm, [])) for arm in eligible
            )
            pool = [
                arm
                for arm in eligible
                if len(self.rewards.get(arm, [])) == minimum_pull_count
            ]
            freshness_reason = "least_pulled"
        else:
            pool = []
            freshness_reason = "no_active_arm"

        (
            relaxation_slots,
            schedule_progress,
            requested_fraction,
            schedule_phase,
        ) = self._schedule(turn)
        raw_scores = {arm: float(topology[arm]["bnrr"]) for arm in pool}
        normalized_scores = {
            arm: raw_scores[arm] / int(topology[arm]["degree"])
            for arm in pool
        }
        shuffled_scores = self._shuffled_scores(pool, raw_scores, topology)
        raw_pool, raw_cutoff = self._upper_tail(raw_scores, requested_fraction)
        normalized_pool, normalized_cutoff = self._upper_tail(
            normalized_scores, requested_fraction
        )
        shuffled_pool, shuffled_cutoff = self._upper_tail(
            shuffled_scores, requested_fraction
        )
        variant_scores = {
            "raw": raw_scores,
            "normalized": normalized_scores,
            "shuffled_raw": shuffled_scores,
        }[self.variant]
        variant_pool = {
            "raw": raw_pool,
            "normalized": normalized_pool,
            "shuffled_raw": shuffled_pool,
        }[self.variant]
        variant_cutoff = {
            "raw": raw_cutoff,
            "normalized": normalized_cutoff,
            "shuffled_raw": shuffled_cutoff,
        }[self.variant]

        selection_u = self._selection_rng.random() if pool else None
        paired_raw_arm = (
            self._pick(raw_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_normalized_arm = (
            self._pick(normalized_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_shuffled_arm = (
            self._pick(shuffled_pool, float(selection_u))
            if selection_u is not None
            else None
        )
        paired_uniform_arm = (
            self._pick(pool, float(selection_u))
            if selection_u is not None
            else None
        )
        selected = {
            "raw": paired_raw_arm,
            "normalized": paired_normalized_arm,
            "shuffled_raw": paired_shuffled_arm,
        }[self.variant]
        if selected is not None and selected not in variant_pool:
            raise RuntimeError("Selected arm is outside the scheduled eligible pool")

        raw_levels = sorted(set(raw_scores.values()))
        normalized_levels = sorted(set(normalized_scores.values()))
        selected_levels = sorted(set(variant_scores.values()))
        selected_share = len(variant_pool) / len(pool) if pool else 0.0
        common = {
            "selection_variant": self.variant,
            "selection_scope": "global_scheduled_upper_tail",
            "base_pool_source": freshness_reason,
            "base_pool_size": len(pool),
            "base_pool_degree_histogram": self._degree_histogram(pool, topology),
            "minimum_pull_count": minimum_pull_count,
            "selection_u": selection_u,
            "observed_unique_bnrr_count": len(raw_levels),
            "observed_unique_normalized_bnrr_count": len(normalized_levels),
            "observed_unique_selection_score_count": len(selected_levels),
            "bnrr_informative": len(selected_levels) > 1,
            "bnrr_policy_tv": 1.0 - selected_share if pool else 0.0,
            "bnrr_low_policy_tv": 0.0,
            "bnrr_high_policy_tv": 1.0 - len(raw_pool) / len(pool) if pool else 0.0,
            "selected_direction_policy_tv": 1.0 - selected_share if pool else 0.0,
            "d1_stratum": (
                int(topology[selected]["degree"]) == 1 if selected else False
            ),
            "singleton_stratum": len(pool) == 1,
            "scores_shuffled": self.variant == "shuffled_raw",
            "schedule_profile": "global_linear_quantile",
            "schedule_global_turn": global_turn,
            "schedule_exploit_pull": turn,
            "schedule_total_turns": self.schedule_total_turns,
            "schedule_relaxation_slots": relaxation_slots,
            "schedule_initial_pool_fraction": self.initial_pool_fraction,
            "schedule_relax_exploit_fraction": self.relax_exploit_fraction,
            "schedule_progress": schedule_progress,
            "schedule_phase": schedule_phase,
            "schedule_requested_tail_fraction": requested_fraction,
            "schedule_tail_cutoff": variant_cutoff,
            "schedule_tail_pool_size": len(variant_pool),
            "schedule_tail_pool_share": selected_share,
            "schedule_raw_tail_cutoff": raw_cutoff,
            "schedule_normalized_tail_cutoff": normalized_cutoff,
            "schedule_shuffled_tail_cutoff": shuffled_cutoff,
            "schedule_raw_tail_pool_size": len(raw_pool),
            "schedule_normalized_tail_pool_size": len(normalized_pool),
            "schedule_shuffled_tail_pool_size": len(shuffled_pool),
            "schedule_raw_tail_degree_histogram": self._degree_histogram(
                raw_pool, topology
            ),
            "schedule_normalized_tail_degree_histogram": self._degree_histogram(
                normalized_pool, topology
            ),
            "schedule_shuffled_tail_degree_histogram": self._degree_histogram(
                shuffled_pool, topology
            ),
            "schedule_capacity_matched": len(raw_pool) == len(shuffled_pool),
            "schedule_active": bool(pool) and requested_fraction < 1.0,
            "paired_scheduled_raw_arm": paired_raw_arm,
            "paired_scheduled_normalized_arm": paired_normalized_arm,
            "paired_scheduled_shuffled_arm": paired_shuffled_arm,
            "paired_scheduled_uniform_arm": paired_uniform_arm,
            "paired_scheduled_uniform_disagreement": (
                selected != paired_uniform_arm
            ),
            "paired_scheduled_raw_shuffled_disagreement": (
                paired_raw_arm != paired_shuffled_arm
            ),
            "paired_scheduled_raw_normalized_disagreement": (
                paired_raw_arm != paired_normalized_arm
            ),
            # Compatibility with the existing aggregate diagnostics.
            "paired_bnrr_uniform_disagreement": selected != paired_uniform_arm,
            "paired_bnrr_high_uniform_disagreement": (
                paired_raw_arm != paired_uniform_arm
            ),
            "paired_bnrr_shuffled_disagreement": selected != paired_shuffled_arm,
            "paired_bnrr_high_shuffled_disagreement": (
                paired_raw_arm != paired_shuffled_arm
            ),
            "paired_bnrr_low_high_disagreement": False,
            "tie_group_preserved": True,
            "degree_marginal_preserved": None,
        }
        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(topology[selected]["bnrr"])} if selected else {}
        )
        self.epoch_end_turn = turn
        reason = (
            f"{self.policy_name}_{freshness_reason}_"
            f"{schedule_phase}_global_upper_tail"
        )
        self.admission_history.append(
            {
                "policy": self.policy_name,
                "epoch_id": self.epoch_id,
                "eligible_arm_count": len(eligible),
                "eligibility_gate": "observed_degree>0_and_valid_bnrr",
                **common,
                "active_arms": list(self.active_arms),
                "candidates": [
                    {
                        "arm": arm,
                        "degree": int(topology[arm]["degree"]),
                        "bnrr": float(topology[arm]["bnrr"]),
                        "normalized_bnrr": (
                            float(topology[arm]["bnrr"])
                            / int(topology[arm]["degree"])
                        ),
                        "shuffled_selection_score": shuffled_scores.get(arm),
                        "in_base_pool": arm in pool,
                        "in_scheduled_raw_tail": arm in raw_pool,
                        "in_scheduled_normalized_tail": arm in normalized_pool,
                        "in_scheduled_shuffled_tail": arm in shuffled_pool,
                        "pull_count": len(self.rewards.get(arm, [])),
                    }
                    for arm in eligible
                ],
                "draws": (
                    [{"name": "selection_u", "value": selection_u}]
                    if pool
                    else []
                ),
                "fallback_reason": (
                    "selection_score_constant"
                    if pool and len(selected_levels) <= 1
                    else (
                        "scheduled_uniform_phase"
                        if requested_fraction >= 1.0
                        else "none"
                    )
                ),
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "selected_degree": (
                    int(topology[selected]["degree"]) if selected else None
                ),
                "selected_bnrr": (
                    float(topology[selected]["bnrr"]) if selected else None
                ),
                "selected_normalized_bnrr": (
                    float(topology[selected]["bnrr"])
                    / int(topology[selected]["degree"])
                    if selected
                    else None
                ),
                "selected_schedule_score": (
                    variant_scores[selected] if selected else None
                ),
                "selected_in_schedule_pool": (
                    selected in variant_pool if selected else True
                ),
                "reason": reason,
                "active_arms": list(self.active_arms),
                "fresh_arm_count": len(fresh),
                **common,
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy_name,
            "selection_policy": self.variant,
            "eligibility_gate": "observed_degree>0_and_valid_bnrr",
            "freshness_scope": "global",
            "degree_marginal": "score_induced_global_pool",
            "reward_used_for_selection": False,
            "truth_used_for_selection": False,
            "tie_group_preserved": True,
            "schedule_profile": "global_linear_quantile",
            "schedule_clock": "exploit_pull",
            "schedule_total_turns": self.schedule_total_turns,
            "schedule_initial_pool_fraction": self.initial_pool_fraction,
            "schedule_relax_exploit_fraction": self.relax_exploit_fraction,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_bnrr": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass
class OpenEgoAlternatingFreshController:
    """Alternate a topology-score slot with uniform global-fresh coverage.

    ``BNRR / degree = sqrt(degree / (degree + 2 * neighbor_edges))`` removes
    BNRR's leading degree scale and isolates observed ego non-redundancy.  Odd
    exploit pulls sample uniformly from the complete maximum-openness tie
    group; even pulls sample uniformly from the whole fresh pool.  A supported
    variant restricts only the topology slot to nodes with degree at least two,
    preventing degree-one leaves from making the BNRR term vacuous.
    """

    seed: int = 42
    minimum_topology_degree: int = 1
    policy_name: str = "open_ego_alternating_fresh"
    topology_score_name: str = "openness"
    topology_slot_period: int = 2
    rewards: dict[str, list[float]] = field(default_factory=dict)
    active_arms: tuple[str, ...] = ()
    active_priors: dict[str, float] = field(default_factory=dict)
    epoch_end_turn: int = 0
    epoch_id: int = 0
    admission_history: list[dict[str, object]] = field(default_factory=list)
    selection_history: list[dict[str, object]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.minimum_topology_degree < 1:
            raise ValueError("minimum_topology_degree must be positive")
        if self.topology_slot_period < 1:
            raise ValueError("topology_slot_period must be positive")
        if self.topology_score_name not in {
            "openness",
            "structural_opportunity",
        }:
            raise ValueError(
                "topology_score_name must be 'openness' or "
                "'structural_opportunity'"
            )
        self._rng = random.Random(self.seed)

    def select(
        self,
        available: Iterable[str],
        turn: int,
        priors: Mapping[str, float] | None = None,
        topology: Mapping[str, Mapping[str, float | int]] | None = None,
    ) -> str | None:
        priors = priors or {}
        topology = topology or {}
        eligible = [
            arm
            for arm in sorted(set(str(item) for item in available))
            if is_valid_arm(arm) and float(priors.get(arm, 0.0)) > 0.0
        ]
        missing_topology = [arm for arm in eligible if arm not in topology]
        if missing_topology:
            raise ValueError(
                f"Missing topology features for eligible arms: {missing_topology[:3]}"
            )

        self.epoch_id += 1
        fresh = [arm for arm in eligible if not self.rewards.get(arm)]
        if fresh:
            pool = fresh
            freshness_reason = "fresh"
        elif eligible:
            minimum_pulls = min(len(self.rewards.get(arm, [])) for arm in eligible)
            pool = [
                arm
                for arm in eligible
                if len(self.rewards.get(arm, [])) == minimum_pulls
            ]
            freshness_reason = "least_pulled"
        else:
            pool = []
            freshness_reason = "no_active_arm"

        supported = [
            arm
            for arm in pool
            if int(topology[arm]["degree"]) >= self.minimum_topology_degree
        ]
        support_fallback = bool(pool) and not supported
        topology_base = supported or pool
        maximum_openness: float | None = None
        maximum_topology_score: float | None = None
        open_pool: list[str] = []
        topology_pool: list[str] = []
        if topology_base:
            maximum_openness = max(
                float(topology[arm]["openness"]) for arm in topology_base
            )
            open_pool = [
                arm
                for arm in topology_base
                if float(topology[arm]["openness"]) == maximum_openness
            ]
            maximum_topology_score = max(
                float(topology[arm][self.topology_score_name])
                for arm in topology_base
            )
            topology_pool = [
                arm
                for arm in topology_base
                if float(topology[arm][self.topology_score_name])
                == maximum_topology_score
            ]

        topology_forced = bool(pool) and (
            (self.epoch_id - 1) % self.topology_slot_period == 0
        )
        selection_pool = topology_pool if topology_forced else pool
        selected = self._rng.choice(selection_pool) if selection_pool else None
        scope = (
            f"maximum_{self.topology_score_name}_tie_group"
            if topology_forced
            else "all"
        )
        reason = f"{self.policy_name}_{freshness_reason}_{scope}"

        self.active_arms = (selected,) if selected else ()
        self.active_priors = (
            {selected: float(priors.get(selected, 0.0))} if selected else {}
        )
        self.epoch_end_turn = turn
        common = {
            "topology_forced": topology_forced,
            "selection_scope": scope,
            "base_pool_size": len(pool),
            "supported_pool_size": len(supported),
            "open_pool_size": len(open_pool),
            "open_pool_share": len(open_pool) / len(pool) if pool else 0.0,
            "maximum_openness": maximum_openness,
            "topology_score_name": self.topology_score_name,
            "topology_pool_size": len(topology_pool),
            "topology_pool_share": (
                len(topology_pool) / len(pool) if pool else 0.0
            ),
            "maximum_topology_score": maximum_topology_score,
            "minimum_topology_degree": self.minimum_topology_degree,
            "topology_slot_period": self.topology_slot_period,
            "support_fallback": support_fallback,
            "tie_group_preserved": True,
        }
        self.admission_history.append(
            {
                "policy": self.policy_name,
                "epoch_id": self.epoch_id,
                "eligible_arm_count": len(eligible),
                "eligibility_gate": "bnrr_sensitivity>0",
                **common,
                "active_arms": list(self.active_arms),
                "candidates": [
                    {
                        "arm": arm,
                        "sensitivity": float(priors.get(arm, 0.0)),
                        "degree": int(topology[arm]["degree"]),
                        "neighbor_edges": int(topology[arm]["neighbor_edges"]),
                        "bnrr": float(topology[arm]["bnrr"]),
                        "openness": float(topology[arm]["openness"]),
                        "pulled_neighbor_fraction": float(
                            topology[arm].get("pulled_neighbor_fraction", 0.0)
                        ),
                        "structural_opportunity": float(
                            topology[arm].get(
                                "structural_opportunity",
                                topology[arm]["openness"],
                            )
                        ),
                        "pull_count": len(self.rewards.get(arm, [])),
                    }
                    for arm in eligible
                ],
                "draws": [],
                "fallback_reason": "none",
                "exploit_pull_start": turn,
                "exploit_pull_end": turn,
            }
        )
        self.selection_history.append(
            {
                "exploit_pull": turn,
                "epoch_id": self.epoch_id,
                "epoch_refreshed": True,
                "selected_arm": selected,
                "reason": reason,
                "active_arms": list(self.active_arms),
                "fresh_arm_count": len(fresh),
                **common,
                "filter_windows": [],
            }
        )
        if selected is not None:
            self.rewards.setdefault(selected, [])
        return selected

    def observe(self, arm: str | None, reward: float) -> None:
        if arm is not None:
            self.rewards.setdefault(arm, []).append(float(reward))

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy_name,
            "selection_policy": (
                f"maximum_{self.topology_score_name}_tie_group_every_exploit"
                if self.topology_slot_period == 1
                else (
                    f"maximum_{self.topology_score_name}_tie_group_every_"
                    f"{self.topology_slot_period}_exploit_pulls"
                )
            ),
            "topology_score": self.topology_score_name,
            "eligibility_gate": "bnrr_sensitivity>0",
            "freshness_scope": "global",
            "minimum_topology_degree": self.minimum_topology_degree,
            "topology_slot_period": self.topology_slot_period,
            "reward_used_for_selection": False,
            "tie_group_preserved": True,
            "seed": self.seed,
            "active_arms": list(self.active_arms),
            "active_priors": self.active_priors,
            "epoch_end_turn": self.epoch_end_turn,
            "epoch_id": self.epoch_id,
            "rewards": self.rewards,
            "admission_history": self.admission_history,
            "selection_history": self.selection_history,
        }


@dataclass(frozen=True)
class ModeDecision:
    """Auditable state for one explore/exploit decision."""

    mode: str
    reason: str
    epsilon: float
    effective_htsn_threshold: float
    recent_htsn: float | None
    recent_explore_success_rate: float | None
    explore_suppressed: bool
    available_arm_count: int

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "epsilon": self.epsilon,
            "effective_htsn_threshold": self.effective_htsn_threshold,
            "recent_htsn": self.recent_htsn,
            "recent_explore_success_rate": self.recent_explore_success_rate,
            "explore_suppressed": self.explore_suppressed,
            "available_arm_count": self.available_arm_count,
        }


@dataclass
class AdaptiveModeController:
    """Use HTSN for mode choice without allowing an absorbing explore state.

    A fixed HTSN threshold can create an absorbing explore state when repeated
    broad queries keep the recent score at zero. This controller keeps HTSN as
    the decision signal while adding an adaptive threshold, an exploration-success
    guard, and a failed-explore streak cap.
    """

    initial_epsilon: float = 0.30
    epsilon_decay: float = 0.98
    min_epsilon: float = 0.05
    htsn_threshold: float = 0.15
    htsn_window: int = 5
    adaptive_threshold: bool = True
    explore_success_window: int = 20
    explore_success_min_samples: int = 5
    explore_success_threshold: float = 0.20
    max_consecutive_failed_explore: int = 2
    sampling_policy: str = "stochastic_epsilon"
    deterministic_explore_surplus_cap: float = 0.0
    seed: int = 42
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        valid_policies = {"deterministic_epsilon_deficit", "stochastic_epsilon"}
        if self.sampling_policy not in valid_policies:
            raise ValueError(
                f"sampling_policy must be one of {sorted(valid_policies)}"
            )
        if self.deterministic_explore_surplus_cap < 0.0:
            raise ValueError(
                "deterministic_explore_surplus_cap must be non-negative"
            )
        self._rng = random.Random(self.seed)

    def epsilon(self, turn: int) -> float:
        return max(
            self.min_epsilon,
            self.initial_epsilon * self.epsilon_decay ** max(0, turn - 1),
        )

    def choose(
        self,
        turn: int,
        available_arms: Iterable[str],
        htsn_history: list[float],
        mode_history: list[str] | None = None,
        meaningful_gain_history: list[bool] | None = None,
    ) -> ModeDecision:
        arms = list(available_arms)
        mode_history = mode_history or []
        meaningful_gain_history = meaningful_gain_history or []
        epsilon = self.epsilon(turn)
        threshold = self.htsn_threshold
        if self.adaptive_threshold and self.initial_epsilon > 0.0:
            threshold *= epsilon / self.initial_epsilon
        recent = htsn_history[-max(1, self.htsn_window) :]
        recent_htsn = sum(recent) / len(recent) if recent else None

        def decision(
            mode: str,
            reason: str,
            *,
            success_rate: float | None = None,
            explore_suppressed: bool = False,
        ) -> ModeDecision:
            return ModeDecision(
                mode=mode,
                reason=reason,
                epsilon=epsilon,
                effective_htsn_threshold=threshold,
                recent_htsn=recent_htsn,
                recent_explore_success_rate=success_rate,
                explore_suppressed=explore_suppressed,
                available_arm_count=len(arms),
            )

        if turn == 1:
            return decision("explore", "seed_turn")
        if not arms:
            return decision("explore", "no_anchor_available")

        recent_start = max(0, len(mode_history) - self.explore_success_window)
        recent_explore_gains = [
            bool(meaningful_gain_history[index])
            for index in range(
                recent_start,
                min(len(mode_history), len(meaningful_gain_history)),
            )
            if mode_history[index] == "explore"
        ]
        success_rate = None
        if len(recent_explore_gains) >= self.explore_success_min_samples:
            success_rate = sum(recent_explore_gains) / len(recent_explore_gains)
            if success_rate < self.explore_success_threshold:
                return decision(
                    "exploit",
                    f"low_explore_success({success_rate:.3f})",
                    success_rate=success_rate,
                    explore_suppressed=True,
                )

        # Productive broad queries are allowed to continue. The cap is only a
        # rescue mechanism for consecutive explore turns with no new relation
        # and no positive history-anchored sensitive mass.
        consecutive_failed_explore = 0
        paired_history = zip(mode_history, meaningful_gain_history)
        for previous_mode, meaningful_gain in reversed(list(paired_history)):
            if previous_mode != "explore":
                break
            if meaningful_gain:
                break
            consecutive_failed_explore += 1
        if (
            self.max_consecutive_failed_explore > 0
            and consecutive_failed_explore
            >= self.max_consecutive_failed_explore
        ):
            return decision(
                "exploit",
                f"failed_explore_streak_cap({self.max_consecutive_failed_explore})",
                success_rate=success_rate,
                explore_suppressed=True,
            )

        if self.sampling_policy == "deterministic_epsilon_deficit":
            expected_explores = sum(
                self.epsilon(candidate_turn)
                for candidate_turn in range(2, turn + 1)
            )
            realized_explores = sum(
                previous_mode == "explore" for previous_mode in mode_history[1:]
            )
            if realized_explores + 0.5 < expected_explores:
                return decision(
                    "explore",
                    "deterministic_epsilon_deficit"
                    f"(expected={expected_explores:.3f},"
                    f"realized={realized_explores})",
                    success_rate=success_rate,
                )
            if (
                recent_htsn is not None
                and recent_htsn < threshold
                and realized_explores + 0.5
                < expected_explores + self.deterministic_explore_surplus_cap
            ):
                return decision(
                    "explore",
                    "deterministic_novelty_surplus"
                    f"(expected={expected_explores:.3f},"
                    f"realized={realized_explores},"
                    f"cap={self.deterministic_explore_surplus_cap:.3f})",
                    success_rate=success_rate,
                )
            return decision(
                "exploit",
                "deterministic_epsilon_quota_met"
                f"(expected={expected_explores:.3f},"
                f"realized={realized_explores})",
                success_rate=success_rate,
            )
        if recent_htsn is not None and recent_htsn < threshold:
            return decision(
                "explore",
                f"recent_htsn_below_threshold({threshold:.4f})",
                success_rate=success_rate,
            )
        if self._rng.random() < epsilon:
            return decision(
                "explore",
                f"epsilon_sample({epsilon:.4f})",
                success_rate=success_rate,
            )
        return decision(
            "exploit",
            f"epsilon_complement({epsilon:.4f})",
            success_rate=success_rate,
        )


def rotting_diagnostics(
    rewards: Mapping[str, list[float]], tolerance: float = 1e-12
) -> dict[str, float | int | None]:
    """Measure observed violations of the FEWA non-increasing-yield assumption."""

    comparisons = 0
    increases = 0
    repeated_arms = 0
    for values in rewards.values():
        if len(values) < 2:
            continue
        repeated_arms += 1
        for previous, current in zip(values, values[1:]):
            comparisons += 1
            increases += int(current > previous + tolerance)
    return {
        "arms_with_repeated_queries": repeated_arms,
        "adjacent_reward_comparisons": comparisons,
        "reward_increases": increases,
        "violation_rate": increases / comparisons if comparisons else None,
    }
