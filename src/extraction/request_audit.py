"""HTTP-level usage accounting, including GraphRAG's internal SSE responses.

No prompt, response text, headers or credentials are persisted. Missing usage
is unknown, never a zero-cost request. Usage chunks are cumulative snapshots,
not token deltas. The final valid snapshot is counted once per HTTP attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def response_events(response):
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        body = None
    if isinstance(body, dict):
        return [body], "json"
    events = []
    data = []
    for line in [*response.text.splitlines(), ""]:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                continue
            try:
                item = json.loads(payload)
            except ValueError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return events, "sse"


def complete_usage(usage, kind):
    fields = ("prompt_tokens", "completion_tokens") if kind == "chat" else ("prompt_tokens",)
    return isinstance(usage, dict) and all(
        type(usage.get(k)) is int and usage[k] >= 0 for k in fields)


def request_stage(row):
    if row["kind"] == "embedding":
        return "embedding"
    return "query_generation" if row.get("max_tokens") in (1024, 3072) else "extraction"


def cost_summary(rows):
    """Disjoint stages; known totals remain separate from complete totals."""
    stages = {}
    for row in rows:
        stage = request_stage(row)
        out = stages.setdefault(stage, {"requests": 0, "errors": 0,
            "unknown_usage_requests": 0, "known_input_tokens": 0,
            "known_output_tokens": 0, "known_cached_input_tokens": 0, "seconds": 0.0})
        out["requests"] += 1
        out["errors"] += int(bool(row.get("error")) or (row.get("status") or 0) >= 400)
        out["seconds"] += row.get("seconds", 0)
        usage = row.get("usage") or {}
        out["unknown_usage_requests"] += int(not complete_usage(usage, row["kind"]))
        for source, target in (("prompt_tokens", "known_input_tokens"), ("completion_tokens", "known_output_tokens")):
            if type(usage.get(source)) is int and usage[source] >= 0:
                out[target] += usage[source]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if type(cached) is int and cached >= 0:
            out["known_cached_input_tokens"] += cached
    for out in stages.values():
        out["tokens_complete"] = out["unknown_usage_requests"] == 0
        out["total_tokens"] = (out["known_input_tokens"] + out["known_output_tokens"]
            if out["tokens_complete"] else None)
    chat = [v for k, v in stages.items() if k != "embedding"]
    return {"stages": stages, "chat_tokens_complete": all(v["tokens_complete"] for v in chat),
        "chat_total_tokens": sum(v["total_tokens"] for v in chat) if all(v["tokens_complete"] for v in chat) else None,
        "embedding_tokens_are_separate": True}


class RequestAudit:
    def __init__(self, path):
        self.path, self.turn = Path(path), 0
        self.count = self.errors = self.input_tokens = self.output_tokens = 0
        self.unknown_usage = 0

    def prepare_request(self, request):
        """Request provider usage, preserving all generation settings and headers."""
        if not request.url.path.endswith("/chat/completions"):
            return request
        body = json.loads(request.content)
        if not body.get("stream"):
            return request
        import httpx
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        headers = request.headers.copy()
        headers.pop("content-length", None)
        return httpx.Request(request.method, request.url, headers=headers,
            content=json.dumps(body, ensure_ascii=False).encode(), extensions=request.extensions)

    def start(self, request):
        if not request.url.path.endswith(("/chat/completions", "/embeddings")):
            return None
        body = json.loads(request.content)
        chat = request.url.path.endswith("/chat/completions")
        native_deepseek = request.url.host == "api.deepseek.com"
        thinking_disabled = ((body.get("thinking") or {}).get("type") == "disabled"
            if native_deepseek else body.get("enable_thinking") is False)
        if chat and not thinking_disabled:
            raise RuntimeError("Refusing chat request without explicit provider-specific thinking disable")
        item = {"turn": self.turn, "kind": "chat" if chat else "embedding", "model": body.get("model"),
            "host": request.url.host, "thinking_disabled": thinking_disabled if chat else None,
            "thinking_control": "thinking.type=disabled" if native_deepseek and chat else "enable_thinking=false" if chat else None,
            "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"), "top_p": body.get("top_p"),
            "stream": bool(body.get("stream")), "include_usage": (body.get("stream_options") or {}).get("include_usage"),
            "request_body_sha256": hashlib.sha256(request.content).hexdigest(), "started": time.time()}
        item["stage"] = request_stage(item)
        if getattr(self, "journal_starts", False):
            item["attempt_id"] = uuid.uuid4().hex
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_name("request_attempts.jsonl").open("a") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return item

    def finish(self, item, response=None, error=None):
        if item is None:
            return
        self.count += 1
        item.update(request_index=self.count, seconds=time.time() - item["started"],
            status=response.status_code if response is not None else None, error=error)
        events, wire_format = response_events(response) if response is not None else ([], None)
        provider_errors = [e["error"] for e in events if e.get("error")]
        if provider_errors:
            # SSE can carry a provider error inside an HTTP 200 response. Store
            # only a fixed classification, never provider text or credentials.
            codes = [e.get("code") for e in provider_errors if isinstance(e, dict)]
            item["error"] = error or ("data_inspection_failed" if any(
                str(code).replace("_", "").lower() == "datainspectionfailed" for code in codes) else "ProviderError")
        self.errors += int(bool(item["error"]) or (response is not None and response.status_code >= 400))
        usage_events = [e["usage"] for e in events if complete_usage(e.get("usage"), item["kind"])]
        usage = usage_events[-1] if usage_events else {}
        item.update(usage=usage, usage_source=wire_format if usage_events else "missing",
            usage_complete=bool(usage_events))
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        self.unknown_usage += int(not usage_events)
        choices = [c for e in events for c in (e.get("choices") or [])]
        reasoned = any(bool((c.get(part) or {}).get("reasoning_content"))
            for c in choices for part in ("message", "delta"))
        item["reasoning_content_nonempty"] = reasoned
        item["finish_reasons"] = sorted({c["finish_reason"] for c in choices if c.get("finish_reason")})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if getattr(self, "journal_starts", False):
                handle.flush()
                os.fsync(handle.fileno())
        if reasoned:
            raise RuntimeError("Provider returned reasoning content despite disabled thinking")

    def install(self):
        import httpx
        sync, async_send = httpx.Client.send, httpx.AsyncClient.send
        def send(client, request, *args, **kwargs):
            request = self.prepare_request(request)
            item = self.start(request)
            try:
                response = sync(client, request, *args, **kwargs)
                if item is not None:
                    response.read()
            except Exception as error:
                self.finish(item, error=type(error).__name__)
                raise
            self.finish(item, response)
            return response
        async def asend(client, request, *args, **kwargs):
            request = self.prepare_request(request)
            item = self.start(request)
            try:
                response = await async_send(client, request, *args, **kwargs)
                if item is not None:
                    await response.aread()
            except Exception as error:
                self.finish(item, error=type(error).__name__)
                raise
            self.finish(item, response)
            return response
        httpx.Client.send, httpx.AsyncClient.send = send, asend
