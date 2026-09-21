"""FULL-only, fail-closed round journal. No network calls or method decisions."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

PROTOCOL = "bnrr-resume-v1"
LOGS = ("decisions.jsonl", "graph_delta.jsonl", "history.jsonl", "metrics.jsonl", "query_generation.jsonl")


def rows(path):
    if not path.exists():
        return []
    # Torn/invalid records are never silently discarded or interpreted as success.
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RoundJournal:
    def __init__(self, out, manifest_sha, *, resume, parent_manifest_sha=None, config_manifest_sha=None,
                 commit_manifest_sha=None):
        self.out, self.manifest_sha = Path(out), manifest_sha
        self.lock = (self.out / ".worker.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another worker holds this run lock") from None
        self.data = {name: rows(self.out / name) for name in LOGS}
        self.completed = 0
        if resume:
            config = json.loads((self.out / "config.json").read_text())
            allowed = {manifest_sha, parent_manifest_sha, config_manifest_sha} - {None}
            if config["manifest_sha256"] not in allowed:
                raise RuntimeError("Resume manifest does not match saved run")
            commit_path = self.out / "COMMIT.json"
            if commit_path.exists():
                commit = json.loads(commit_path.read_text())
                allowed_commits = allowed | ({commit_manifest_sha} if commit_manifest_sha else set())
                if commit["protocol"] != PROTOCOL or commit["manifest_sha256"] not in allowed_commits:
                    raise RuntimeError("Commit protocol/manifest mismatch")
                self.completed = commit["round"]
                for name, info in commit["files"].items():
                    content = (self.out / name).read_bytes()
                    if len(content) < info["bytes"] or digest(content[:info["bytes"]]) != info["sha256"]:
                        raise RuntimeError(f"Committed log changed: {name}")
            else:
                # Legacy adoption requires equal, uninterrupted completed logs.
                lengths = [len(self.data[n]) for n in LOGS[1:4]]
                if len(set(lengths)) != 1:
                    raise RuntimeError("Legacy partial write requires explicit recovery; refusing to guess")
                self.completed = lengths[0]
            for name, values in self.data.items():
                start = 2 if name == "query_generation.jsonl" else 1
                if [x["turn"] for x in values] != list(range(start, start + len(values))):
                    raise RuntimeError(f"Non-contiguous/duplicate rounds in {name}")
                required = max(0, self.completed - start + 1)
                if not required <= len(values) <= required + 1:
                    raise RuntimeError(f"Invalid uncommitted tail in {name}")
        elif any(self.data.values()):
            raise RuntimeError("New run contains existing records")

    def get(self, name, turn):
        return next((r for r in self.data[name] if r["turn"] == turn), None)

    def append(self, name, value):
        old = self.get(name, value["turn"])
        if old is not None:
            if old != value:
                raise RuntimeError(f"Uncommitted record mismatch: {name} round {value['turn']}")
            return
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.data[name].append(value)

    def commit(self, turn, write_json):
        if turn != self.completed + 1:
            raise RuntimeError("Non-sequential commit")
        files = {}
        for name, values in self.data.items():
            start = 2 if name == "query_generation.jsonl" else 1
            if [x["turn"] for x in values] != list(range(start, turn + 1)):
                raise RuntimeError(f"Incomplete round transaction: {name}")
            path = self.out / name
            if path.exists():
                content = path.read_bytes()
                files[name] = {"bytes": len(content), "sha256": digest(content)}
        # write_json uses temp + replace; fsync preserves the commit across crashes.
        write_json(self.out / "COMMIT.json", {"protocol": PROTOCOL, "round": turn,
            "manifest_sha256": self.manifest_sha, "files": files})
        with (self.out / "COMMIT.json").open("rb") as handle:
            os.fsync(handle.fileno())
        fd = os.open(self.out, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.completed = turn

    def rebuild_csv(self):
        values = self.data["metrics.jsonl"]
        if values:
            path = self.out / "metrics.csv"
            temp = path.with_suffix(".csv.tmp")
            with temp.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
            temp.replace(path)

    def close(self):
        self.lock.close()


def restore_accounting(audit):
    values = rows(audit.path)
    if [r["request_index"] for r in values] != list(range(1, len(values) + 1)):
        raise RuntimeError("Request journal is not contiguous")
    finished = {r.get("attempt_id") for r in values}
    for started in rows(audit.path.with_name("request_attempts.jsonl")):
        if started["attempt_id"] not in finished:
            # The remote may have billed an interrupted request. Never call it free.
            uncertain = {**started, "request_index": len(values) + 1, "status": None,
                "error": "InterruptedRequestOutcomeUnknown", "usage": {}, "usage_complete": False,
                "usage_source": "missing", "finish_reasons": [], "reasoning_content_nonempty": False,
                "seconds": 0.0, "duration_unknown": True}
            with audit.path.open("a") as handle:
                handle.write(json.dumps(uncertain, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            values.append(uncertain)
            finished.add(started["attempt_id"])
    audit.count = len(values)
    audit.errors = sum(bool(r.get("error")) or (r.get("status") or 0) >= 400 for r in values)
    audit.input_tokens = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in values)
    audit.output_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in values)
    audit.unknown_usage = sum(not r.get("usage") for r in values)


def saved_query(library, graph, history, row, decision, memory_profile):
    """Validate persisted writer input/output without a second writer call."""
    from .bnrr_queries import exploration_messages, exploitation_messages, _generate
    from .bnrr_prompt_profiles import call_query
    import inspect
    from types import SimpleNamespace
    mode, anchor = decision["mode"], decision["anchor"]
    messages = (call_query(exploitation_messages, library, graph, history, anchor, memory_profile=memory_profile)
        if mode == "exploit" else call_query(exploration_messages, library, graph, history, memory_profile=memory_profile))
    if row["messages"] != messages or row["action"] != mode or row["anchor"] != anchor:
        raise RuntimeError("Saved query does not match restored pre-query state")
    # Reuse the exact normalization/validation implementation, with no network client.
    choice = SimpleNamespace(message=SimpleNamespace(content=row["raw_query"], reasoning_content=None),
                             finish_reason=row["finish_reason"])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[choice]))))
    options = {}
    if "anchor_protocol" in inspect.signature(_generate).parameters:
        options.update(anchor_protocol="anchor-single-quote-escape-v1" if memory_profile == "recent_exclusion_desc" else "literal-v1",
            observed_labels=graph.nodes)
    elif memory_profile != "recent":
        raise RuntimeError("Legacy query normalization cannot replay this memory profile")
    text, replay = _generate(messages, client=client, model=row["model"], completion_options={},
        temperature=.2 if mode == "exploit" else .3, action=mode, anchor=anchor, **options)
    if {k: v for k, v in row.items() if k != "turn"} != replay:
        raise RuntimeError("Saved query normalization/output mismatch")
    return text, replay
