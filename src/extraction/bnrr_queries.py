"""AGEA-style online query writing without a second mode/anchor policy."""
from __future__ import annotations

import json
import re
from typing import Any

from .prompts import PromptLibrary


class QueryGenerationError(ValueError):
    """A rejected query retains its original model output for diagnosis."""
    def __init__(self, message: str, audit: dict):
        super().__init__(message)
        self.audit = audit


def exploration_messages(library: PromptLibrary, graph: Any, history: list[dict], *, memory_profile="recent_exclusion_desc") -> list[dict]:
    """Only observed semantic history enters the writer, never topology or truth."""
    if memory_profile != "recent_exclusion_desc":
        raise ValueError("Unsupported query memory profile")
    from .query_described_exclusion import exploration_context
    context = json.dumps(exploration_context(graph, history), ensure_ascii=False)
    return [{"role": "system", "content": library.templates["query_generator_system"]},
            {"role": "user", "content": library.templates["explore_query_generator"].format(context=context)}]



def generate_explore_query(library: PromptLibrary, graph: Any, history: list[dict], *,
                           client: Any, model: str, completion_options: dict, memory_profile="recent_exclusion_desc") -> tuple[str, dict]:
    """One query-writing call before retrieval; errors never trigger a topic fallback."""
    messages = exploration_messages(library, graph, history, memory_profile=memory_profile)
    return _generate(messages, client=client, model=model, completion_options=completion_options,
                     temperature=0.3, action="explore")


def exploitation_messages(library: PromptLibrary, graph: Any, history: list[dict], anchor: str, *, memory_profile="recent_exclusion_desc") -> list[dict]:
    if not anchor or anchor not in graph:
        raise ValueError("Exploit requires a fixed observed anchor")
    if memory_profile != "recent_exclusion_desc":
        raise ValueError("Unsupported query memory profile")
    from .query_described_exclusion import exploitation_context
    context = json.dumps(exploitation_context(graph, history, anchor), ensure_ascii=False)
    return [{"role": "system", "content": library.templates["query_generator_system"]},
            {"role": "user", "content": library.templates["exploit_query_generator"].format(context=context)}]



def generate_exploit_query(library: PromptLibrary, graph: Any, history: list[dict], *,
                           anchor: str, client: Any, model: str, completion_options: dict, memory_profile="recent_exclusion_desc") -> tuple[str, dict]:
    return _generate(exploitation_messages(library, graph, history, anchor, memory_profile=memory_profile), client=client,
        model=model, completion_options=completion_options, temperature=0.2, action="exploit", anchor=anchor,
        anchor_protocol="anchor-single-quote-escape-v1" if memory_profile == "recent_exclusion_desc" else "literal-v1",
        observed_labels=tuple(graph.nodes) if memory_profile == "recent_exclusion_desc" else ())


def _generate(messages: list[dict], *, client: Any, model: str, completion_options: dict,
              temperature: float, action: str, anchor: str | None = None,
              anchor_protocol="literal-v1", observed_labels=()) -> tuple[str, dict]:
    if anchor_protocol not in {"literal-v1", "anchor-single-quote-escape-v1"}:
        raise ValueError("Unknown anchor normalization protocol")
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, top_p=1.0,
        max_tokens=1024, **completion_options,
    )
    choice = response.choices[0]
    content = choice.message.content
    audit = {"messages": messages, "raw_query": content, "model": model,
             "temperature": temperature, "max_tokens": 1024, "action": action,
             "anchor": anchor, "finish_reason": getattr(choice, "finish_reason", None)}
    def reject(message, reason="invalid_query"):
        audit["rejection_reason"] = reason
        raise QueryGenerationError(message, audit)
    if getattr(choice.message, "reasoning_content", None):
        reject("Query writer returned reasoning despite no-thinking contract", "unexpected_reasoning")
    if getattr(choice, "finish_reason", None) == "length":
        reject("Query writer returned a truncated query", "query_output_length")
    if not isinstance(content, str) or not content.strip():
        reject("Query writer returned no query")
    # Preserve punctuation at an unusual anchor's end for the new protocol;
    # whole-response strip('"') could itself delete part of its literal label.
    query = " ".join((content.strip() if anchor_protocol == "anchor-single-quote-escape-v1"
                      else content.strip().strip('"').strip("'")).split())
    if anchor is not None and anchor_protocol == "anchor-single-quote-escape-v1":
        from .query_anchor import canonicalize_anchor
        try:
            canonical, normalization = canonicalize_anchor(query, anchor, observed_labels)
        except ValueError as error:
            reject(str(error))
        audit["anchor_normalization"] = normalization
        if normalization["changed"]:
            audit["query_before_anchor_normalization"] = query
        query = canonical
    if not query or len(query) > 800 or "```" in query or re.search(r'\b(?:ENTITY|Source|Target)\s*:', query, re.I):
        reject("Query writer returned extraction records or invalid query text")
    if anchor is not None and anchor_protocol == "literal-v1":
        pattern = r'(?<!\w)' + re.escape(anchor) + r'(?!\w)'
        if not re.search(pattern, query, re.IGNORECASE):
            reject("Exploit query writer omitted or changed the fixed anchor")
        # Graph entity labels are case-insensitive. Restore their exact stored
        # spelling without accepting aliases, omissions, or a different target.
        canonical = re.sub(pattern, lambda match: anchor, query, flags=re.IGNORECASE)
        if canonical != query:
            audit["query_before_anchor_case_normalization"] = query
        query = canonical
    audit["query"] = query
    return query, audit
