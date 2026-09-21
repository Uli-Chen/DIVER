"""The official BNRR prompt bundle and observed-history memory contract."""
from pathlib import Path
from .prompts import PromptLibrary
from .bnrr_config import DEFAULTS

PROFILE = DEFAULTS.prompt_profile
MEMORY_PROFILE = DEFAULTS.query_memory
PROFILES = (PROFILE,)


def call_query(function, *args, memory_profile=MEMORY_PROFILE, **kwargs):
    return function(*args, memory_profile=memory_profile, **kwargs)


def profile_settings(profile=PROFILE):
    if profile != PROFILE:
        raise ValueError("Unsupported BNRR prompt profile; replay historical runs with their frozen source")
    return {"profile": profile, "memory_profile": MEMORY_PROFILE}


def load_profile(project: str | Path, profile=PROFILE) -> PromptLibrary:
    profile_settings(profile)
    return PromptLibrary(Path(project) / "configs/prompts/bnrr", profile="bnrr")


def manifest_profile(manifest):
    config = profile_settings(manifest.get("bnrr_prompt_profile", PROFILE))
    if manifest.get("query_memory_profile", MEMORY_PROFILE) != MEMORY_PROFILE:
        raise ValueError("Frozen prompt and query-memory profiles are inconsistent")
    return config
