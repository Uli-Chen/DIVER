import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from extraction.request_audit import RequestAudit, cost_summary


def req(**kwargs):
    return httpx.Request("POST", "https://example.test/v1/chat/completions",
        headers={"Authorization": "Bearer secret-do-not-log"},
        json={"model": "test", "enable_thinking": False, "max_tokens": 16384, **kwargs})


def sse(events):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
        content="".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n")


def test_terminal_stream_usage_counted_once_and_content_unchanged(tmp_path):
    audit = RequestAudit(tmp_path / "requests.jsonl")
    response = sse([
        {"choices": [{"delta": {"content": "private-content"}}], "usage": None},
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
    ])
    original = response.content
    audit.finish(audit.start(req(stream=True)), response)
    row = json.loads(audit.path.read_text())
    assert row["usage_source"] == "sse" and row["usage_complete"]
    assert row["finish_reasons"] == ["stop"]
    assert audit.input_tokens == 10 and audit.output_tokens == 4
    assert response.content == original
    assert "private-content" not in audit.path.read_text()
    assert "secret-do-not-log" not in audit.path.read_text()


def test_usage_option_preserves_generation_and_headers(tmp_path):
    audit = RequestAudit(tmp_path / "requests.jsonl")
    original = req(stream=True, temperature=0, stream_options={"other": True})
    prepared = audit.prepare_request(original)
    body = json.loads(prepared.content)
    assert body["stream_options"] == {"other": True, "include_usage": True}
    assert body["temperature"] == 0 and body["max_tokens"] == 16384
    assert prepared.headers["authorization"] == original.headers["authorization"]
    assert int(prepared.headers["content-length"]) == len(prepared.content)
    ordinary = req(stream=False)
    assert audit.prepare_request(ordinary) is ordinary


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 20}, {"prompt_tokens": -1, "completion_tokens": 2}])
def test_missing_or_partial_usage_not_reported_as_zero(tmp_path, usage):
    audit = RequestAudit(tmp_path / "requests.jsonl")
    audit.finish(audit.start(req()), httpx.Response(200, json={"usage": usage}))
    assert audit.unknown_usage == 1
    result = cost_summary([json.loads(audit.path.read_text())])
    assert result["chat_total_tokens"] is None
    assert not result["chat_tokens_complete"]


def test_reasoning_detected_in_stream_delta(tmp_path):
    audit = RequestAudit(tmp_path / "requests.jsonl")
    with pytest.raises(RuntimeError, match="reasoning"):
        audit.finish(audit.start(req()), sse([{"choices": [{"delta": {"reasoning_content": "hidden"}}]}]))
    assert json.loads(audit.path.read_text())["reasoning_content_nonempty"]


def test_errors_and_embedding_separate(tmp_path):
    audit = RequestAudit(tmp_path / "requests.jsonl")
    audit.finish(audit.start(req()), error="ReadTimeout")
    audit.finish(audit.start(req()), httpx.Response(500, json={"error": "unavailable"}))
    embedding = httpx.Request("POST", "https://example.test/v1/embeddings", json={"model": "embed"})
    audit.finish(audit.start(embedding), httpx.Response(200, json={"usage": {"prompt_tokens": 8}}))
    assert audit.count == 3 and audit.errors == 2 and audit.unknown_usage == 2
    summary = cost_summary([json.loads(l) for l in audit.path.read_text().splitlines()])
    assert summary["stages"]["embedding"]["total_tokens"] == 8
    assert summary["chat_total_tokens"] is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_http_hook_with_real_httpx_stream_transport(tmp_path, monkeypatch, asynchronous):
    audit = RequestAudit(tmp_path / "requests.jsonl")
    # Register originals with monkeypatch before install to restore afterwards.
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    audit.install()
    def handler(request):
        assert json.loads(request.content)["stream_options"]["include_usage"] is True
        return sse([{"choices": [{"delta": {"content": "OK"}}]},
            {"usage": {"prompt_tokens": 11, "completion_tokens": 1}}])
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async def run():
            async with httpx.AsyncClient(transport=transport) as client:
                response = await client.send(req(stream=True), stream=True)
                assert "OK" in (await response.aread()).decode()
        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as client:
            response = client.send(req(stream=True), stream=True)
            assert "OK" in response.read().decode()
    assert audit.count == 1 and audit.input_tokens == 11 and audit.output_tokens == 1


def test_metered_launcher_dispatch_and_old_manifest_guard(tmp_path, monkeypatch):
    from extraction import accounting as module
    assert Path(module.pilot.__file__).name == "run_bnrr.py"
    assert module.pilot.RequestAudit is RequestAudit
    monkeypatch.setattr(module, "original_check", lambda root: {})
    with pytest.raises(RuntimeError, match="historical"):
        module.check_manifest(tmp_path)
    monkeypatch.setattr(module, "original_environment", lambda: None)
    monkeypatch.setenv("AGEA_QUERY_MAX_TOKENS", "3072")
    module.environment()
    import os
    assert os.environ["AGEA_QUERY_MAX_TOKENS"] == "1024"


@pytest.mark.parametrize("missing_seed_usage", [False, True])
def test_efficiency_summary_charges_initialization_and_keeps_unknown(tmp_path, monkeypatch, missing_seed_usage):
    from extraction import accounting as module
    manifest = {"methods": ["AGEA_R", "FULL"], "seeds": [42, 43, 44],
        "accounting_protocol": module.PROTOCOL, "agea_query_max_tokens": 1024}
    monkeypatch.setattr(module, "original_summarize", lambda root: None)
    monkeypatch.setattr(module, "original_check", lambda root: manifest)
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    request = {"kind": "chat", "max_tokens": 16384, "status": 200, "seconds": 1,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    for seed in manifest["seeds"]:
        initial = {**request, "turn": 1}
        if missing_seed_usage and seed == 43:
            initial["usage"] = {}
        write_rows(tmp_path / f"seed_generation/seed{seed}/requests.jsonl", [initial])
        for method in manifest["methods"]:
            run = tmp_path / f"runs/{method}_seed{seed}"
            write_rows(run / "requests.jsonl", [{**request, "turn": 2}])
            write_rows(run / "metrics.jsonl", [{"turn": t, **{k: 0.5 for k in module.pilot.METRICS}} for t in (1, 2)])
    module.summarize(tmp_path)
    report = json.loads((tmp_path / "EFFICIENCY.json").read_text())
    for method in manifest["methods"]:
        assert report["methods"][method]["chat_total_tokens_mean"] == (None if missing_seed_usage else 24)
        run = report["runs"][f"{method}_seed42"]
        assert run["quality_vs_cost"][0]["cumulative_cost"]["chat_total_tokens"] == 12
        assert run["quality_vs_cost"][1]["cumulative_cost"]["chat_total_tokens"] == 24
