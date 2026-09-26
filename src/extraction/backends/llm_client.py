"""OpenAI-compatible query-writing client for DIVER reproduction.

Retains the provider compatibility options from the locally patched AGEA
snapshot (https://github.com/shuashua0608/AGEA, revision
c9c27dd15d55fb2a64fef89d2f9f7cda80038ffe).
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from openai import OpenAI


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def get_openai_client() -> OpenAI:
    """Use the same chat provider as GraphRAG answer generation."""
    key = _env("PROVIDER_API_KEY")
    endpoint = _env("PROVIDER_API_BASE")
    if not key or not endpoint:
        raise ValueError("Set PROVIDER_API_KEY and PROVIDER_API_BASE before reproduction")
    return OpenAI(api_key=key, base_url=endpoint.rstrip("/"), max_retries=2, timeout=120)


def resolve_agent_model(role_env: str, fallback: str) -> str:
    """Resolve the shared chat model; the role argument is compatibility-only."""
    model = _env("PROVIDER_CHAT_MODEL") or fallback
    return model.split("#", 1)[0].strip()


def agent_completion_options(model: str) -> dict:
    """Select the non-reasoning parameter accepted by the configured provider."""
    if "deepseek-v4" not in model.casefold():
        return {}
    if urlsplit(_env("PROVIDER_API_BASE")).hostname == "api.deepseek.com":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {"extra_body": {"enable_thinking": False}}
