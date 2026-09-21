"""Shared exposure machinery and historical absolute-threshold controller.

New runs select RankBnrrController through their explicit rank_schedule manifest.

Only pre-query observed topology enters the footprint. Exposure is updated once
per successfully committed exploit, never on a request retry or failed parse.
No truth, generated query text, or response gain enters policy decisions.
"""
from dataclasses import dataclass, field
import math
import hashlib
import random

from ..metrics.graph import compute_node_scores, simple_projection
from ..models import normalize_label

def keyed_rng(seed: int, turn: int, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{turn}:{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def nearest_rank(values: list[float], q: float) -> float:
    if not values or not 0 < q <= 1:
        raise ValueError("nearest_rank needs a nonempty sample and q in (0,1]")
    return sorted(values)[math.ceil(q * len(values)) - 1]


PROTOCOL = 'bnrr-soft-self-onehop'

def weighted_choice(labels, weights, rng):
    if not labels or any(not math.isfinite(weights[v]) or weights[v] <= 0 for v in labels):
        raise ValueError('Positive finite weights and nonempty candidates required')
    total = sum(weights[v] for v in labels)
    draw, cumulative = rng.random() * total, 0.0
    for v in labels:
        cumulative += weights[v]
        if draw < cumulative:
            return v
    return labels[-1]

@dataclass
class BnrrController:
    horizon: int
    seed: int = 42
    q_hi: float = 0.9
    q_lo: float = 0.1
    rho: float = 0.2
    gate_policy: str = 'residual_mass'
    calibration_round: int | None = None
    tau_hi: float | None = None
    tau_lo: float | None = None
    neighbor_penalty: float = 0.5
    completed: set[str] = field(default_factory=set)
    exposure: dict[str, float] = field(default_factory=dict)
    query_counts: dict[str, int] = field(default_factory=dict)
    _pending_anchor: str | None = None
    _pending_neighbors: tuple[str, ...] = ()

    def __post_init__(self):
        if self.horizon < 1 or not 0 < self.q_lo <= self.q_hi <= 1:
            raise ValueError('invalid horizon or quantile endpoints')
        if not 0 <= self.rho <= 1:
            raise ValueError('rho must be in [0,1]')
        if self.gate_policy != 'residual_mass' or not 0 <= self.neighbor_penalty <= 1:
            raise ValueError('Soft controller requires residual_mass and alpha in [0,1]')

    def threshold(self, turn: int) -> float | None:
        if self.calibration_round is None:
            return None
        span = self.horizon - self.calibration_round - 1
        u = min(1.0, max(0.0, (turn - self.calibration_round - 1) / span)) if span > 0 else 0.0
        return (1 - u) * self.tau_hi + u * self.tau_lo


    def snapshot(self, graph):
        # Previously queried anchors remain eligible. Isolates follow baseline.
        return {v:{'degree':s.degree,'neighbor_edges':s.neighbor_edges,'bnrr':s.bnrr}
            for v,s in sorted(compute_node_scores(graph).items()) if s.degree>0}

    def decide(self, graph, turn, *, moderation_excluded=None):
        if not 1 <= turn <= self.horizon or self._pending_anchor is not None:
            raise ValueError('Invalid turn or previous exploit not completed')
        pool=self.snapshot(graph)
        if turn>1 and pool and self.calibration_round is None:
            values=[s['bnrr'] for s in pool.values()]
            self.calibration_round=turn-1
            self.tau_hi=nearest_rank(values,self.q_hi)
            self.tau_lo=nearest_rank(values,self.q_lo)
        tau=self.threshold(turn)
        eligible=[v for v,s in pool.items() if tau is not None and s['bnrr']>=tau]
        weights={v:1.0/(1.0+self.exposure.get(v,0.0)) for v in pool}
        total=sum(s['bnrr'] for s in pool.values())
        raw_eligible=sum(pool[v]['bnrr'] for v in eligible)
        residual=sum(pool[v]['bnrr']*weights[v] for v in eligible)
        probability=residual/total if total else 0.0
        draw=keyed_rng(self.seed,turn,'bnrr_gate').random() if turn>1 else None
        mode,anchor,branch='explore',None,'seed' if turn==1 else 'no_eligible'
        if turn>1 and eligible:
            branch='bnrr_mass_explore'
            if draw<probability:
                mode='exploit'
                branch='random_soft' if keyed_rng(self.seed,turn,'mixture').random()<self.rho else 'bnrr_eligible_soft'
                choices=list(pool) if branch=='random_soft' else eligible
                anchor=weighted_choice(choices,weights,keyed_rng(self.seed,turn,'anchor'))
        fallback = {}
        if moderation_excluded is not None:
            alternatives = sorted(set(pool) - set(moderation_excluded))
            anchor = keyed_rng(self.seed, turn, 'moderation_anchor').choice(alternatives) if alternatives else None
            mode = 'exploit' if anchor is not None else 'explore'
            branch = 'moderation_random_anchor' if anchor is not None else 'moderation_no_other_anchor'
            fallback = {'moderation_excluded': sorted(moderation_excluded),
                'moderation_candidates': alternatives, 'anchor_sampling': 'uniform',
                'bnrr_exploit_probability': probability}
            probability = 1.0 if alternatives else 0.0
        neighbors=tuple(sorted(simple_projection(graph).neighbors(anchor))) if anchor is not None else ()
        self._pending_anchor=anchor
        self._pending_neighbors=neighbors
        return {'turn':turn,'mode':mode,'anchor':anchor,'branch':branch,
            'fresh':pool,'eligible':eligible,'fresh_size':len(pool),'eligible_size':len(eligible),
            'max_bnrr':max((s['bnrr'] for s in pool.values()),default=None),
            'tau':tau,'tau_hi':self.tau_hi,'tau_lo':self.tau_lo,'calibration_round':self.calibration_round,
            'gate_policy':self.gate_policy,'total_bnrr_mass':total,'raw_eligible_bnrr_mass':raw_eligible,
            'eligible_fresh_bnrr_mass':residual,'exploit_probability':probability,'gate_draw':draw,
            'coverage_protocol':PROTOCOL,'candidate_scope':'all_positive_degree_including_queried',
            'neighbor_penalty':self.neighbor_penalty,'exposure':{v:self.exposure.get(v,0.0) for v in pool},
            'sampling_weights':weights,'pre_query_neighbors':list(neighbors),
            'selected_previous_query_count':self.query_counts.get(anchor,0) if anchor else 0, **fallback}

    def complete(self, anchor):
        if anchor is None:
            if self._pending_anchor is not None:raise ValueError('Missing selected anchor')
            return
        anchor=normalize_label(anchor)
        if anchor!=self._pending_anchor:raise ValueError('No matching pending exploit')
        self.exposure[anchor]=self.exposure.get(anchor,0.0)+1.0
        for v in self._pending_neighbors:
            self.exposure[v]=self.exposure.get(v,0.0)+self.neighbor_penalty
        self.query_counts[anchor]=self.query_counts.get(anchor,0)+1
        self.completed.add(anchor)  # Diagnostic compatibility only; never a mask.
        self._pending_anchor=None
        self._pending_neighbors=()

    def skip(self, anchor):
        """Release a failed attempt without charging semantic exposure."""
        if (normalize_label(anchor) if anchor is not None else None) != self._pending_anchor:
            raise ValueError('No matching pending exploit to skip')
        self._pending_anchor = None
        self._pending_neighbors = ()
