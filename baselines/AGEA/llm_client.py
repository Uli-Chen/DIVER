"""Provider-neutral chat client construction for AGEA agents.

AGEA only relies on the OpenAI-compatible ``chat.completions`` surface.  This
module keeps Azure OpenAI support while allowing providers such as DeepSeek,
SiliconFlow, OpenRouter, and locally hosted compatible servers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


ChatClient = Union[OpenAI, AzureOpenAI]

_MODEL_ALIASES = {
    # Paratera's OpenAI-compatible endpoint treats model IDs as
    # case-sensitive. Canonicalize the legacy AGEA spellings without
    # lowercasing arbitrary provider model names.
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
}


def _env(*names: str) -> Optional[str]:
    """Return the first non-empty environment value among ``names``."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _is_azure_endpoint(endpoint: str) -> bool:
    hostname = (urlparse(endpoint).hostname or "").lower()
    return hostname.endswith(".openai.azure.com") or hostname.endswith(".cognitiveservices.azure.com")


def _load_dataset_env(dataset_path: Optional[str]) -> None:
    if not dataset_path:
        return
    env_path = Path(dataset_path) / ".env"
    if env_path.exists():
        # Repository-level variables loaded by the caller remain authoritative.
        load_dotenv(env_path, override=False)


def resolve_chat_backend(dataset_path: Optional[str] = None) -> Tuple[str, str, str, Optional[str]]:
    """Resolve provider, API key, endpoint/base URL, and Azure API version.

    Preferred generic configuration:

    - ``AGEA_LLM_PROVIDER=openai_compatible``
    - ``AGEA_API_KEY`` and ``AGEA_API_BASE``

    Standard ``OPENAI_*`` and GraphRAG ``GRAPHRAG_*`` names are accepted as
    fallbacks.  In ``auto`` mode, legacy ``AZURE_OPENAI_*`` variables pointing
    at a non-Azure host are interpreted as an OpenAI-compatible endpoint.  This
    preserves compatibility with older AGEA environment files that used Azure
    variable names for providers such as DeepSeek.
    """
    _load_dataset_env(dataset_path)

    provider = (_env("AGEA_LLM_PROVIDER") or "auto").lower().replace("-", "_")
    if provider == "openai":
        provider = "openai_compatible"
    if provider not in {"auto", "openai_compatible", "azure"}:
        raise ValueError(
            "AGEA_LLM_PROVIDER must be one of: auto, openai_compatible, azure"
        )

    # Keep AGEA's agent endpoint isolated from GraphRAG's answer-generation
    # endpoint. The namespaces may coexist in the shared repository .env;
    # treating GRAPHRAG_* as an equal-priority generic credential could still
    # switch the query generator to a different provider.
    generic_key = _env("AGEA_API_KEY", "OPENAI_API_KEY")
    generic_base = _env("AGEA_API_BASE", "OPENAI_BASE_URL")
    graphrag_key = _env("GRAPHRAG_API_KEY")
    graphrag_base = _env("GRAPHRAG_API_BASE")
    azure_key = _env("AZURE_OPENAI_API_KEY")
    azure_endpoint = _env("AZURE_OPENAI_ENDPOINT")
    azure_version = _env("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

    if provider == "openai_compatible":
        api_key = generic_key or azure_key or graphrag_key
        api_base = generic_base or azure_endpoint or graphrag_base
        if not api_key or not api_base:
            raise ValueError(
                "OpenAI-compatible AGEA client requires AGEA_API_KEY and AGEA_API_BASE "
                "(OPENAI_* and GRAPHRAG_* fallbacks are also supported)."
            )
        return "openai_compatible", api_key, api_base.rstrip("/"), None

    if provider == "azure":
        if not azure_key or not azure_endpoint:
            raise ValueError(
                "Azure AGEA client requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )
        return "azure", azure_key, azure_endpoint.rstrip("/"), azure_version

    # Auto mode prefers explicitly named AGEA/OpenAI credentials.
    if generic_key and generic_base:
        return "openai_compatible", generic_key, generic_base.rstrip("/"), None

    # Backward compatibility: some existing AGEA .env files store a DeepSeek
    # or other compatible endpoint under the old AZURE_OPENAI_* names.
    if azure_key and azure_endpoint and not _is_azure_endpoint(azure_endpoint):
        return "openai_compatible", azure_key, azure_endpoint.rstrip("/"), None

    if azure_key and azure_endpoint:
        return "azure", azure_key, azure_endpoint.rstrip("/"), azure_version

    # GraphRAG credentials are compatibility fallbacks only. In particular,
    # they must not override an AGEA endpoint loaded before GraphRAG starts.
    if graphrag_key and graphrag_base:
        return "openai_compatible", graphrag_key, graphrag_base.rstrip("/"), None

    raise ValueError(
        "No AGEA chat API configured. Set AGEA_API_KEY and AGEA_API_BASE for an "
        "OpenAI-compatible provider, or configure AZURE_OPENAI_* for Azure."
    )


def create_chat_client(dataset_path: Optional[str] = None) -> ChatClient:
    provider, api_key, endpoint, api_version = resolve_chat_backend(dataset_path)
    if provider == "azure":
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            max_retries=10,
        )
    return OpenAI(api_key=api_key, base_url=endpoint, max_retries=10)


def resolve_agent_model(role_env: str, fallback: str) -> str:
    """Resolve a role-specific model with common provider fallbacks."""

    raw_model = _env(role_env, "AGEA_CHAT_MODEL", "GRAPHRAG_CHAT_MODEL") or fallback
    model = raw_model.split("#", 1)[0].strip()
    # Paratera treats these legacy model IDs as case-sensitive, whereas other
    # compatible providers (notably DashScope) expose a distinct lowercase ID.
    # Applying the Paratera spelling globally turns a valid DashScope model
    # into a 404 and silently forces AGEA's query generator onto its fallback.
    endpoint = _env(
        "AGEA_API_BASE",
        "OPENAI_BASE_URL",
        "GRAPHRAG_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
    )
    hostname = (urlparse(endpoint).hostname or "").casefold() if endpoint else ""
    if hostname == "paratera.com" or hostname.endswith(".paratera.com"):
        return _MODEL_ALIASES.get(model.casefold(), model)
    return model


def agent_completion_options(model: str) -> dict:
    """Return provider options that keep agent calls in chat/output mode.

    DeepSeek-V4 reasoning can consume the entire completion budget before
    emitting the structured query or KEEP/DISCARD decisions that AGEA needs.
    The provider exposes ``enable_thinking=false`` for the corresponding chat
    behavior, which matches the non-reasoning agent models used by AGEA's
    original experiment.
    """

    if "deepseek-v4" in model.casefold():
        style = (_env("AGEA_THINKING_CONTROL_STYLE") or "compat").casefold()
        if style == "enable_thinking":
            return {"extra_body": {"enable_thinking": False}}
        if style == "thinking_disabled":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        if style == "reasoning_disabled":
            return {"extra_body": {"reasoning": {"enabled": False}}}
        if style == "reasoning_effort":
            return {"reasoning_effort": "none"}
        if style not in {"compat", "none"}:
            raise ValueError(
                "AGEA_THINKING_CONTROL_STYLE must be one of: "
                "compat, enable_thinking, reasoning_disabled, "
                "reasoning_effort, none"
            )
        if style == "none":
            return {}
        return {
            "reasoning_effort": "none",
            "extra_body": {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "thinking": {"type": "disabled"},
            },
        }
    return {}
