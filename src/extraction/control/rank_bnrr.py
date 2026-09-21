"""Fractional rank admission with soft exposure; the default schedule tightens."""
from dataclasses import dataclass
from fractions import Fraction
import math

from .bnrr import BnrrController, PROTOCOL, keyed_rng, weighted_choice
from ..metrics.graph import simple_projection

RANK_PROTOCOL = 'rank-fractional-boundary'
DIRECTIONS = {'relax': (.1, .9), 'tighten': (.9, .1)}


def specification(direction):
    start, end = DIRECTIONS[direction]
    return {'protocol': RANK_PROTOCOL, 'direction': direction,
        'start_fraction': start, 'end_fraction': end,
        'progress': 'global-postseed-rounds', 'quota': 'ceil', 'boundary_ties': 'fractional'}


def validate_spec(config):
    if not isinstance(config, dict) or config.get('direction') not in DIRECTIONS:
        raise ValueError('Unknown rank schedule')
    if config != specification(config['direction']):
        raise ValueError('Rank schedule differs from the frozen experiment contract')


def fraction_at(turn, horizon, direction):
    if not 1 <= turn <= horizon or horizon < 2:
        raise ValueError('Invalid scheduled round')
    start, end = (Fraction(str(v)) for v in DIRECTIONS[direction])
    progress = Fraction(max(0, turn-2), horizon-2) if horizon > 2 else Fraction(0)
    return start*(1-progress) + end*progress


def admission(scores, fraction):
    if not scores:
        return {}, 0, None, 0.
    # Exact rational arithmetic avoids ceil(3.0000000000000004) becoming four.
    desired = fraction*len(scores)
    quota = (desired.numerator + desired.denominator-1)//desired.denominator
    boundary = sorted(scores.values(), reverse=True)[quota-1]
    above = sum(value > boundary for value in scores.values())
    tied = sum(value == boundary for value in scores.values())
    boundary_weight = (quota-above)/tied
    membership = {v: 1. if b > boundary else boundary_weight if b == boundary else 0.
        for v,b in scores.items()}
    if not math.isclose(sum(membership.values()), quota, abs_tol=1e-8):
        raise RuntimeError('Effective rank admission does not match quota')
    return membership, quota, boundary, boundary_weight


@dataclass
class RankBnrrController(BnrrController):
    direction: str = 'tighten'

    def __post_init__(self):
        super().__post_init__()
        if self.direction not in DIRECTIONS or self.horizon < 2:
            raise ValueError('Invalid rank controller configuration')

    def decide(self, graph, turn, *, moderation_excluded=None):
        if not 1 <= turn <= self.horizon or self._pending_anchor is not None:
            raise ValueError('Invalid turn or previous exploit not completed')
        pool = self.snapshot(graph)
        fraction = fraction_at(turn, self.horizon, self.direction)
        membership, quota, boundary, boundary_weight = admission(
            {v:s['bnrr'] for v,s in pool.items()}, fraction)
        eligible = [v for v in pool if membership[v] > 0]
        weights = {v:1/(1+self.exposure.get(v,0.)) for v in pool}
        guided = {v:membership[v]*weights[v] for v in eligible}
        total = sum(s['bnrr'] for s in pool.values())
        raw = sum(membership[v]*pool[v]['bnrr'] for v in eligible)
        residual = sum(guided[v]*pool[v]['bnrr'] for v in eligible)
        probability = residual/total if total else 0.
        draw = keyed_rng(self.seed,turn,'bnrr_gate').random() if turn>1 else None
        mode,anchor,branch = 'explore',None,'seed' if turn==1 else 'no_eligible'
        if turn>1 and eligible:
            branch = 'bnrr_mass_explore'
            if draw < probability:
                mode = 'exploit'
                branch = 'random_soft' if keyed_rng(self.seed,turn,'mixture').random()<self.rho else 'rank_eligible_soft'
                choices, sampling = (list(pool),weights) if branch=='random_soft' else (eligible,guided)
                anchor = weighted_choice(choices,sampling,keyed_rng(self.seed,turn,'anchor'))
        fallback = {}
        if moderation_excluded is not None:
            alternatives = sorted(set(pool)-set(moderation_excluded))
            anchor = keyed_rng(self.seed,turn,'moderation_anchor').choice(alternatives) if alternatives else None
            mode = 'exploit' if anchor is not None else 'explore'
            branch = 'moderation_random_anchor' if anchor is not None else 'moderation_no_other_anchor'
            fallback = {'moderation_excluded':sorted(moderation_excluded), 'moderation_candidates':alternatives,
                'anchor_sampling':'uniform', 'bnrr_exploit_probability':probability}
            probability = 1. if alternatives else 0.
        neighbors = tuple(sorted(simple_projection(graph).neighbors(anchor))) if anchor is not None else ()
        self._pending_anchor,self._pending_neighbors = anchor,neighbors
        return {'turn':turn,'mode':mode,'anchor':anchor,'branch':branch,'fresh':pool,'eligible':eligible,
            'fresh_size':len(pool),'eligible_size':len(eligible),
            'max_bnrr':max((s['bnrr'] for s in pool.values()),default=None),
            'tau':boundary,'tau_hi':None,'tau_lo':None,'calibration_round':None,
            'gate_policy':self.gate_policy,'total_bnrr_mass':total,'raw_eligible_bnrr_mass':raw,
            'eligible_fresh_bnrr_mass':residual,'exploit_probability':probability,'gate_draw':draw,
            'coverage_protocol':PROTOCOL,'candidate_scope':'all_positive_degree_including_queried',
            'neighbor_penalty':self.neighbor_penalty,'exposure':{v:self.exposure.get(v,0.) for v in pool},
            'sampling_weights':weights,'eligible_sampling_weights':guided,'pre_query_neighbors':list(neighbors),
            'selected_previous_query_count':self.query_counts.get(anchor,0) if anchor else 0,
            'rank_protocol':RANK_PROTOCOL,'rank_direction':self.direction,'rank_fraction':float(fraction),
            'rank_quota':quota,'effective_eligible_size':sum(membership.values()),
            'effective_eligible_fraction':quota/len(pool) if pool else 0.,
            'admission_weights':membership,'boundary_weight':boundary_weight, **fallback}
