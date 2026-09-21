"""Topology-sensitive candidate admission for exploit epochs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_GENERIC_TERMS = {
    "SUMMARY",
    "ORGANIZATIONS",
    "ROLES",
    "OVERVIEW",
    "ENTITY NAME",
    "ENTITY",
    "RELATIONSHIPS",
    "ENTITIES",
    "REPORTS",
    "INFORMATION",
    "DATA",
    "CONTENT",
    "UNKNOWN",
    "N/A",
    "NULL",
    "EMPTY",
}


def is_valid_arm(entity: str) -> bool:
    """Reject parser artifacts and generic labels as exploit targets."""

    entity = str(entity)
    if len(entity.strip()) < 2:
        return False
    if (
        entity.startswith("ENTITIES (")
        or entity.startswith("RELATIONSHIPS (")
        or entity.startswith("REPORTS (")
        or (entity.startswith("[") and entity.endswith("]"))
    ):
        return False
    return entity.upper().strip() not in _GENERIC_TERMS


@dataclass(frozen=True)
class AdmissionDecision:
    """A frozen epoch arm set sampled by ordered Plackett--Luce draws."""

    policy: str
    epoch_id: int
    eligible_arm_count: int
    active_arms: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]
    draws: tuple[dict[str, Any], ...]
    fallback_reason: str
    epoch_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "epoch_id": self.epoch_id,
            "eligible_arm_count": self.eligible_arm_count,
            "active_arms": list(self.active_arms),
            "candidates": list(self.candidates),
            "draws": list(self.draws),
            "fallback_reason": self.fallback_reason,
            "epoch_seed": self.epoch_seed,
        }


@dataclass(frozen=True)
class TopologyPlackettLuceAdmission:
    """Sample a fixed FEWA arm set from topology and prior pull counts only."""

    seed: int = 42

    def admit(
        self,
        available: Iterable[str],
        priors: Mapping[str, float],
        pull_counts: Mapping[str, int],
        *,
        max_arms: int,
        epoch_id: int,
    ) -> AdmissionDecision:
        candidates: list[dict[str, Any]] = []
        for arm in sorted(set(str(item) for item in available)):
            if not is_valid_arm(arm):
                continue
            sensitivity = min(1.0, max(0.0, float(priors.get(arm, 0.0))))
            if sensitivity <= 0.0:
                continue
            pull_count = max(0, int(pull_counts.get(arm, 0)))
            # Admission uses only the frozen topology prior and under-pull
            # correction. Observed reward remains exclusively FEWA's signal.
            weight = sensitivity / math.sqrt(1.0 + pull_count)
            candidates.append(
                {
                    "arm": arm,
                    "sensitivity": sensitivity,
                    "pull_count": pull_count,
                    "weight": weight,
                }
            )

        epoch_seed = self.seed + 1_000_003 * max(0, epoch_id)
        rng = random.Random(epoch_seed)
        selected, draws, fallback = _plackett_luce_sample(
            rng,
            candidates,
            min(max(0, max_arms), len(candidates)),
        )
        return AdmissionDecision(
            policy="topology_plackett_luce",
            epoch_id=epoch_id,
            eligible_arm_count=len(candidates),
            active_arms=tuple(str(item["arm"]) for item in selected),
            candidates=tuple(candidates),
            draws=tuple(draws),
            fallback_reason=fallback,
            epoch_seed=epoch_seed,
        )


def _plackett_luce_sample(
    rng: random.Random,
    candidates: Sequence[Mapping[str, Any]],
    count: int,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], str]:
    """Return ordered draws and their conditional probabilities."""

    remaining = [dict(item) for item in candidates]
    selected: list[Mapping[str, Any]] = []
    draws: list[dict[str, Any]] = []
    fallback_reason = "none"
    for draw_index in range(count):
        total = sum(max(0.0, float(item["weight"])) for item in remaining)
        uniform_fallback = total <= 0.0 or not math.isfinite(total)
        draw = rng.random()
        if uniform_fallback:
            fallback_reason = "uniform_nonpositive_or_nonfinite_weight"
            selected_index = min(int(draw * len(remaining)), len(remaining) - 1)
            chosen = remaining[selected_index]
            conditional_probability = 1.0 / len(remaining)
        else:
            threshold = draw * total
            cumulative = 0.0
            selected_index = len(remaining) - 1
            for index, item in enumerate(remaining):
                cumulative += max(0.0, float(item["weight"]))
                if threshold <= cumulative:
                    selected_index = index
                    break
            chosen = remaining[selected_index]
            conditional_probability = float(chosen["weight"]) / total

        draws.append(
            {
                "draw_index": draw_index + 1,
                "uniform_draw": draw,
                "remaining": [
                    {"arm": item["arm"], "weight": item["weight"]}
                    for item in remaining
                ],
                "remaining_total_weight": total,
                "selected_arm": chosen["arm"],
                "selected_weight": chosen["weight"],
                "conditional_probability": conditional_probability,
                "uniform_fallback": uniform_fallback,
            }
        )
        selected.append(chosen)
        remaining.pop(selected_index)
    return selected, draws, fallback_reason
