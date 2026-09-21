"""Learning-free Incident/Closure scheduling and candidate construction."""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from ..metrics.graph import NodeScore, compute_node_scores
from ..models import normalize_label
from .actions import ActionKind, EntityPair, QueryAction
from .admission import is_valid_arm


TRI_ACTION_VARIANTS = {
    "incident_only",
    "uniform_closure",
    "bnrr_closure",
    "raw_bnrr_greedy",
}


def _neighbors(graph: nx.Graph, node: str) -> set[str]:
    if graph.is_directed():
        return {
            normalize_label(item)
            for item in (*graph.predecessors(node), *graph.successors(node))
            if normalize_label(item) and normalize_label(item) != normalize_label(node)
        }
    return {
        normalize_label(item)
        for item in graph.neighbors(node)
        if normalize_label(item) and normalize_label(item) != normalize_label(node)
    }


def _known_pairs(graph: nx.Graph) -> set[EntityPair]:
    return {
        EntityPair.from_values(source, target)
        for source, target in graph.edges()
        if normalize_label(source)
        and normalize_label(target)
        and normalize_label(source) != normalize_label(target)
    }


def _degree_bin(degree: int) -> int:
    return int(math.floor(math.log2(max(1, degree))))


def _midrank(value: float, reference: Sequence[float]) -> float:
    ordered = sorted(float(item) for item in reference)
    if not ordered:
        return 0.0
    lower = bisect.bisect_left(ordered, value)
    upper = bisect.bisect_right(ordered, value)
    return (lower + 0.5 * (upper - lower)) / len(ordered)


@dataclass(frozen=True)
class LocalCandidateSnapshot:
    node_scores: Mapping[str, NodeScore]
    incident_eligible: tuple[str, ...]
    fresh_incident: tuple[str, ...]
    open_pairs: Mapping[str, tuple[EntityPair, ...]]
    raw_open_pair_counts: Mapping[str, int]
    pair_evidence_scores: Mapping[EntityPair, int]
    closure_executable: tuple[str, ...]
    closure_eligible: tuple[str, ...]
    degree_bins: Mapping[str, int]
    degree_bin_sizes: Mapping[str, int]
    degree_conditioned_percentiles: Mapping[str, float]
    all_tie_bins: frozenset[int]

    @property
    def local_entities(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.incident_eligible) | set(self.closure_eligible)))


@dataclass
class LearningFreeTriActionController:
    """Use binary structural eligibility, independent freshness, and rotation."""

    variant: str = "bnrr_closure"
    seed: int = 42
    closure_bundle_size: int = 1
    closure_bnrr_percentile: float = 0.5
    prevent_consecutive_closure: bool = True
    closure_pair_policy: str = "uniform"
    closure_min_shared_text_units: int = 0
    max_closure_local_share: float = 0.5
    entity_text_units: Mapping[str, frozenset[str]] = field(
        default_factory=dict, repr=False
    )
    incident_pull_counts: dict[str, int] = field(default_factory=dict)
    closure_anchor_pull_counts: dict[str, int] = field(default_factory=dict)
    closure_pair_pull_counts: dict[EntityPair, int] = field(default_factory=dict)
    global_topic_pull_counts: dict[str, int] = field(default_factory=dict)
    last_local_action: ActionKind | None = None
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)
    _closure_schedule_credit: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.variant not in TRI_ACTION_VARIANTS:
            raise ValueError(
                f"Unknown tri-action variant {self.variant!r}; "
                f"expected one of {sorted(TRI_ACTION_VARIANTS)}"
            )
        if self.closure_bundle_size <= 0:
            raise ValueError("closure_bundle_size must be positive")
        if not 0.0 <= self.closure_bnrr_percentile <= 1.0:
            raise ValueError("closure_bnrr_percentile must be in [0, 1]")
        if self.closure_pair_policy not in {"uniform", "text_unit_overlap"}:
            raise ValueError(self.closure_pair_policy)
        if self.closure_min_shared_text_units < 0:
            raise ValueError("closure_min_shared_text_units must be non-negative")
        if not 0.0 <= self.max_closure_local_share <= 1.0:
            raise ValueError("max_closure_local_share must be in [0, 1]")
        self._rng = random.Random(self.seed)
        self._closure_schedule_credit = 0.0

    def _pair_evidence_score(self, pair: EntityPair) -> int:
        left_units = self.entity_text_units.get(pair.left, frozenset())
        right_units = self.entity_text_units.get(pair.right, frozenset())
        return len(left_units & right_units)

    def build_snapshot(self, graph: nx.MultiDiGraph) -> LocalCandidateSnapshot:
        scores = compute_node_scores(graph)
        incident = tuple(
            sorted(
                node
                for node, score in scores.items()
                if is_valid_arm(node) and score.degree > 0
            )
        )
        fresh_incident = tuple(
            node for node in incident if self.incident_pull_counts.get(node, 0) == 0
        )

        known = _known_pairs(graph)
        open_pairs: dict[str, tuple[EntityPair, ...]] = {}
        raw_open_pair_counts: dict[str, int] = {}
        pair_evidence_scores: dict[EntityPair, int] = {}
        if self.variant != "incident_only":
            for node in incident:
                neighbors = sorted(_neighbors(graph, node))
                if len(neighbors) < 2:
                    continue
                raw_candidates = tuple(
                    pair
                    for source_index, source in enumerate(neighbors)
                    for target in neighbors[source_index + 1 :]
                    for pair in (EntityPair(source, target),)
                    if pair not in known
                    and self.closure_pair_pull_counts.get(pair, 0) == 0
                )
                raw_open_pair_counts[node] = len(raw_candidates)
                if self.closure_pair_policy == "text_unit_overlap":
                    scored = [
                        (pair, self._pair_evidence_score(pair))
                        for pair in raw_candidates
                    ]
                    pair_evidence_scores.update(scored)
                    candidates = tuple(
                        pair
                        for pair, score in scored
                        if score >= self.closure_min_shared_text_units
                    )
                else:
                    candidates = raw_candidates
                if len(candidates) >= self.closure_bundle_size:
                    open_pairs[node] = candidates

        executable = tuple(sorted(open_pairs))
        bins: dict[int, list[str]] = defaultdict(list)
        degree_bins: dict[str, int] = {}
        for node in executable:
            bin_id = _degree_bin(scores[node].degree)
            degree_bins[node] = bin_id
            bins[bin_id].append(node)

        percentiles: dict[str, float] = {}
        bin_sizes: dict[str, int] = {}
        all_tie_bins: set[int] = set()
        for bin_id, nodes in bins.items():
            raw = [scores[node].bnrr for node in nodes]
            if len(set(raw)) == 1:
                all_tie_bins.add(bin_id)
            for node in nodes:
                percentiles[node] = _midrank(scores[node].bnrr, raw)
                bin_sizes[node] = len(nodes)

        if self.variant == "incident_only":
            closure_eligible: tuple[str, ...] = ()
        elif self.variant == "uniform_closure":
            closure_eligible = executable
        elif self.variant == "raw_bnrr_greedy":
            if executable:
                maximum = max(scores[node].bnrr for node in executable)
                closure_eligible = tuple(
                    node for node in executable if scores[node].bnrr == maximum
                )
            else:
                closure_eligible = ()
        else:
            closure_eligible = tuple(
                node
                for node in executable
                if percentiles[node] >= self.closure_bnrr_percentile
            )

        return LocalCandidateSnapshot(
            node_scores=scores,
            incident_eligible=incident,
            fresh_incident=fresh_incident,
            open_pairs=open_pairs,
            raw_open_pair_counts=raw_open_pair_counts,
            pair_evidence_scores=pair_evidence_scores,
            closure_executable=executable,
            closure_eligible=closure_eligible,
            degree_bins=degree_bins,
            degree_bin_sizes=bin_sizes,
            degree_conditioned_percentiles=percentiles,
            all_tie_bins=frozenset(all_tie_bins),
        )

    def select_global(
        self,
        topics: Iterable[str],
        *,
        turn: int,
        seed_topic: str | None = None,
    ) -> QueryAction:
        if seed_topic is not None:
            return QueryAction(
                kind=ActionKind.GLOBAL,
                topic=seed_topic,
                metadata={"turn": turn, "seed": True, "reason": "seed_turn"},
            )
        available = tuple(str(topic) for topic in topics)
        if not available:
            raise ValueError("At least one global topic is required")
        minimum = min(self.global_topic_pull_counts.get(topic, 0) for topic in available)
        pool = [
            topic
            for topic in available
            if self.global_topic_pull_counts.get(topic, 0) == minimum
        ]
        topic = self._rng.choice(pool)
        self.global_topic_pull_counts[topic] = self.global_topic_pull_counts.get(topic, 0) + 1
        return QueryAction(
            kind=ActionKind.GLOBAL,
            topic=topic,
            metadata={
                "turn": turn,
                "seed": False,
                "reason": "uniform_least_pulled_topic",
                "minimum_topic_pulls": minimum,
                "candidate_topics": len(pool),
            },
        )

    def select_local(
        self,
        snapshot: LocalCandidateSnapshot,
        *,
        turn: int,
    ) -> QueryAction | None:
        incident_available = bool(snapshot.incident_eligible)
        closure_available = bool(snapshot.closure_eligible)
        if closure_available:
            self._closure_schedule_credit = min(
                1.0,
                self._closure_schedule_credit + self.max_closure_local_share,
            )
        closure_under_cap = self._closure_schedule_credit >= 1.0
        closure_blocked = bool(
            self.prevent_consecutive_closure
            and self.last_local_action == ActionKind.CLOSURE
        )

        if closure_available and closure_under_cap and not closure_blocked:
            action = self._select_closure(snapshot, turn=turn)
        elif incident_available:
            action = self._select_incident(snapshot, turn=turn)
        elif closure_available:
            action = self._select_closure(snapshot, turn=turn)
        else:
            return None

        if action.kind == ActionKind.CLOSURE:
            self._closure_schedule_credit = max(
                0.0, self._closure_schedule_credit - 1.0
            )
        self.last_local_action = action.kind
        self.decision_history.append(action.to_dict())
        return action

    def _select_incident(
        self, snapshot: LocalCandidateSnapshot, *, turn: int
    ) -> QueryAction:
        if snapshot.fresh_incident:
            pool = list(snapshot.fresh_incident)
            reason = "uniform_fresh_incident"
        else:
            minimum = min(
                self.incident_pull_counts.get(node, 0)
                for node in snapshot.incident_eligible
            )
            pool = [
                node
                for node in snapshot.incident_eligible
                if self.incident_pull_counts.get(node, 0) == minimum
            ]
            reason = "uniform_least_pulled_incident"
        anchor = self._rng.choice(pool)
        previous = self.incident_pull_counts.get(anchor, 0)
        self.incident_pull_counts[anchor] = previous + 1
        score = snapshot.node_scores[anchor]
        return QueryAction(
            kind=ActionKind.INCIDENT,
            anchor=anchor,
            metadata={
                "turn": turn,
                "reason": reason,
                "pool_size": len(pool),
                "incident_eligible_count": len(snapshot.incident_eligible),
                "fresh_incident_count": len(snapshot.fresh_incident),
                "previous_incident_pulls": previous,
                "degree": score.degree,
                "bnrr": score.bnrr,
            },
        )

    def _select_closure(
        self, snapshot: LocalCandidateSnapshot, *, turn: int
    ) -> QueryAction:
        eligible = list(snapshot.closure_eligible)
        if self.variant == "raw_bnrr_greedy":
            pool = eligible
            reason = "raw_bnrr_greedy"
        else:
            fresh = [
                node
                for node in eligible
                if self.closure_anchor_pull_counts.get(node, 0) == 0
            ]
            if fresh:
                pool = fresh
                reason = "uniform_fresh_closure_anchor"
            else:
                minimum = min(
                    self.closure_anchor_pull_counts.get(node, 0)
                    for node in eligible
                )
                pool = [
                    node
                    for node in eligible
                    if self.closure_anchor_pull_counts.get(node, 0) == minimum
                ]
                reason = "uniform_least_pulled_closure_anchor"
        anchor = self._rng.choice(pool)
        previous = self.closure_anchor_pull_counts.get(anchor, 0)
        self.closure_anchor_pull_counts[anchor] = previous + 1
        available_pairs = list(snapshot.open_pairs[anchor])
        bundle_size = min(self.closure_bundle_size, len(available_pairs))
        if self.closure_pair_policy == "text_unit_overlap":
            ranked = sorted(
                available_pairs,
                key=lambda pair: (
                    -snapshot.pair_evidence_scores.get(pair, 0),
                    pair.left,
                    pair.right,
                ),
            )
            cutoff_score = snapshot.pair_evidence_scores.get(
                ranked[bundle_size - 1], 0
            )
            above_cutoff = [
                pair
                for pair in ranked
                if snapshot.pair_evidence_scores.get(pair, 0) > cutoff_score
            ]
            tied_cutoff = [
                pair
                for pair in ranked
                if snapshot.pair_evidence_scores.get(pair, 0) == cutoff_score
            ]
            remaining = bundle_size - len(above_cutoff)
            pairs = tuple(
                [*above_cutoff, *self._rng.sample(tied_cutoff, remaining)]
            )
        else:
            pairs = tuple(self._rng.sample(available_pairs, bundle_size))
        for pair in pairs:
            self.closure_pair_pull_counts[pair] = (
                self.closure_pair_pull_counts.get(pair, 0) + 1
            )
        score = snapshot.node_scores[anchor]
        bin_id = snapshot.degree_bins[anchor]
        return QueryAction(
            kind=ActionKind.CLOSURE,
            anchor=anchor,
            pairs=pairs,
            metadata={
                "turn": turn,
                "reason": reason,
                "variant": self.variant,
                "anchor_pool_size": len(pool),
                "closure_executable_count": len(snapshot.closure_executable),
                "closure_eligible_count": len(snapshot.closure_eligible),
                "previous_closure_anchor_pulls": previous,
                "open_pair_count": len(available_pairs),
                "raw_open_pair_count": snapshot.raw_open_pair_counts.get(
                    anchor, len(available_pairs)
                ),
                "bundle_size": bundle_size,
                "closure_pair_policy": self.closure_pair_policy,
                "pair_evidence_scores": [
                    snapshot.pair_evidence_scores.get(pair, 0) for pair in pairs
                ],
                "max_closure_local_share": self.max_closure_local_share,
                "degree": score.degree,
                "neighbor_edges": score.neighbor_edges,
                "clustering": score.clustering,
                "bnrr": score.bnrr,
                "degree_bin": bin_id,
                "degree_bin_size": snapshot.degree_bin_sizes[anchor],
                "degree_conditioned_bnrr_percentile": (
                    snapshot.degree_conditioned_percentiles[anchor]
                ),
                "singleton_bin": snapshot.degree_bin_sizes[anchor] == 1,
                "all_tie_bin": bin_id in snapshot.all_tie_bins,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": "learning_free_tri_action",
            "variant": self.variant,
            "seed": self.seed,
            "closure_bundle_size": self.closure_bundle_size,
            "closure_bnrr_percentile": self.closure_bnrr_percentile,
            "prevent_consecutive_closure": self.prevent_consecutive_closure,
            "closure_pair_policy": self.closure_pair_policy,
            "closure_min_shared_text_units": self.closure_min_shared_text_units,
            "max_closure_local_share": self.max_closure_local_share,
            "closure_schedule_credit": self._closure_schedule_credit,
            "last_local_action": (
                self.last_local_action.value if self.last_local_action else None
            ),
            "incident_pull_counts": dict(self.incident_pull_counts),
            "closure_anchor_pull_counts": dict(self.closure_anchor_pull_counts),
            "closure_pair_pull_counts": {
                f"{pair.left}\t{pair.right}": count
                for pair, count in self.closure_pair_pull_counts.items()
            },
            "global_topic_pull_counts": dict(self.global_topic_pull_counts),
            "decision_history": list(self.decision_history),
        }
