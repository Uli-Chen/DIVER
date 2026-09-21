"""Canonical defaults for new BNRR experiments; frozen runs keep their own manifests."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BnrrConfig:
    rounds: int = 100
    seeds: tuple[int, ...] = (42, 43, 44)
    prompt_profile: str = "agea_adapted_null"
    parser: str = "bnrr"
    parser_protocol: str = "bnrr-null-record-local"
    controller_protocol: str = "bnrr-soft-self-onehop"
    query_memory: str = "recent_exclusion_desc"
    gate_policy: str = "residual_mass"
    neighbor_penalty: float = 0.5
    rank_direction: str = "tighten"
    # Used only when replaying frozen manifests without rank_schedule.
    q_hi: float = 0.9
    q_lo: float = 0.1
    rho: float = 0.2
    extraction_max_tokens: int = 16384
    query_max_tokens: int = 1024
    thinking: bool = False
    extra_llm_filter: bool = False
    moderation_policy: str = "retry-then-random-other-anchor"
    moderation_attempts: int = 3

    @property
    def rank_schedule(self):
        from extraction.control.rank_bnrr import specification
        return specification(self.rank_direction)

    def to_dict(self):
        return {**asdict(self), "rank_schedule": self.rank_schedule}


DEFAULTS = BnrrConfig()
