"""Durable handling of exhausted provider moderation failures; no graph mutation."""
import hashlib
import json
import os
import time
from pathlib import Path

POLICY = "retry-then-random-other-anchor"


def is_moderation_error(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        body = getattr(error, "body", None)
        detail = body.get("error", body) if isinstance(body, dict) else {}
        codes = [getattr(error, "code", None), detail.get("code") if isinstance(detail, dict) else None]
        if any(str(code).replace("_", "").lower() == "datainspectionfailed" for code in codes):
            return True
        # OpenAI's SSE APIError may retain the message but discard the code.
        if type(error).__module__.startswith("openai") and any(message in str(error) for message in (
                "Input data may contain inappropriate content", "Output data may contain inappropriate content",
                "Input or output data may contain inappropriate content")):
            return True
        error = error.__cause__
    return False


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def failure_path(root, stage, turn):
    return Path(root) / "moderation" / f"{stage}_{turn}.json"


def load_failure(root, stage, turn, payload, attempts):
    path = failure_path(root, stage, turn)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    expected = {"policy": POLICY, "turn": turn, "stage": stage,
        "request_identity": identity(payload), "attempts": attempts, "error_code": "data_inspection_failed"}
    if record != expected:
        raise RuntimeError("Moderation failure evidence does not match the frozen request")
    return dict(sorted(record.items()))


def save_failure(root, stage, turn, payload, attempts):
    record = {"policy": POLICY, "turn": turn, "stage": stage,
        "request_identity": identity(payload), "attempts": attempts, "error_code": "data_inspection_failed"}
    path = failure_path(root, stage, turn)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return dict(sorted(record.items()))


def skip_stats(record):
    return {"source": "provider_error", "parse_status": "skipped", "round_skipped": True,
        "skip_reason": "content_moderation", "skip_stage": record["stage"],
        "skipped_records": [], "finish_reasons": [], "moderation_failure": record}


def call(operation, root, stage, turn, payload, attempts):
    record = load_failure(root, stage, turn, payload, attempts)
    if record is not None:
        return None, record
    # The extraction adapter already makes `attempts` attempts. The query writer
    # needs its own bounded moderation retries (the SDK does not retry HTTP 400).
    limit = attempts if stage == "query_generation" else 1
    for attempt in range(1, limit + 1):
        try:
            return operation(), None
        except Exception as error:
            if not is_moderation_error(error):
                raise
            if attempt == limit:
                return None, save_failure(root, stage, turn, payload, attempts)
            time.sleep(min(2 ** (attempt - 1), 8))


def rejected_query(payload, action, anchor):
    return {**payload, "action": action, "anchor": anchor, "query": None,
        "raw_query": None, "rejection_reason": "content_moderation"}


def choose_decision(controller, graph, turn, history, enabled):
    excluded = None
    if enabled and history and history[-1]["parse_stats"].get("skip_reason") == "content_moderation":
        excluded = set()
        for row in reversed(history):
            if row["parse_stats"].get("skip_reason") != "content_moderation":
                break
            if row.get("anchor") is not None:
                excluded.add(row["anchor"])
    return controller.decide(graph, turn, moderation_excluded=excluded)
