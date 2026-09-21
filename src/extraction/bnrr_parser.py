"""Record-local BNRR parser. Only explicit fields supply graph identities.

Missing attributes stay None in JSON records. No truth data, model calls or
legacy numbered-list fallback participates in parsing or recovery.
"""
from __future__ import annotations

import re

from .models import normalize_label

PROTOCOL = "bnrr-null-record-local"
FIELD = re.compile(r"^\s*(?:#{1,6}\s*)?(?:(?:[-*+] |\d+[.)]\s+)\s*)?(ENTITY|Relationships|Source|Target|Description)\s*:\s*(.*?)\s*$", re.I)
CITATION = re.compile(r"\[Data:[^\[\]]*\]", re.I)
BOUNDARY = re.compile(r"^\s*(?:#{1,6}\s+|\d+[.)]\s+|```|[-_]{3,}\s*$|\*\*[^*]+\*\*\s*$)")
INLINE = re.compile(r"^Source\s*:\s*(.*?)\s*(?:,\s*|→\s*|->\s*)Target\s*:\s*(.*?)(?:,\s*Description\s*:\s*|\s+[—–]\s+|\s+--\s+)(.*?)$", re.I)


def missing(value):
    if value is None:
        return True
    clean = value.strip()
    while len(clean) >= 2 and (clean[0], clean[-1]) in {("[", "]"), ("(", ")"), ('"', '"'), ("'", "'")}:
        clean = clean[1:-1].strip()
    # UNKNOWN and NA can be real entity names; never reserve them here.
    return not clean or clean.lower() in {"null", "none", "n/a"} or bool(re.fullmatch(
        r"(?:none\b\s+.*|no (?:(?:direct|supported|outgoing|incoming|explicit)\s+)*relationships?\b.*)", clean, re.I))


def identity(value):
    if missing(value) or re.search(r"\b(?:Source|Target|Description)\s*:", value, re.I):
        return None
    value = CITATION.sub("", value).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    return normalize_label(value) or None


def description(parts):
    value = CITATION.sub("", "\n".join(parts)).strip()
    if missing(value) or re.fullmatch(
        r"[([]?\s*no\s+(?:(?:entity|relationship|additional|explicit|separate)\s+)*description\b.*", value, re.I):
        return None
    return value


def parse_response(text, *, extract_body=lambda value: value, finish_reasons=()):
    body = extract_body(text)
    body = re.sub(r"\AFull GraphRAG Response \(including retrieved context\):\s*\n", "", body, flags=re.I)
    reasons = sorted(set(finish_reasons))
    stats = {"parser_protocol": PROTOCOL, "parse_status": "parsed", "syntax_errors": [],
        "syntax_repairs": [], "skipped_records": [], "entity_records_skipped": 0,
        "relationship_records_skipped": 0, "inherited_source_records": 0,
        "compact_format_edges": 0, "citation_rejected_compact_edges": 0,
        "finish_reasons": reasons, "completion_verified": bool(reasons), "round_skipped": False}
    if "length" in reasons:
        stats.update(parse_status="skipped", round_skipped=True, skip_reason="output_length")
        return [], [], stats
    if body.strip() == "NO_SUPPORTED_RECORDS":
        stats["parse_status"] = "explicit_empty"
        return [], [], stats

    nodes, edges = {}, {}
    entity = relation = None
    active = None

    def skip(kind, record, reason):
        stats[kind + "_records_skipped"] += 1
        stats["skipped_records"].append({"record_type": kind, "line": record["line"], "reason": reason})

    def finish_entity():
        nonlocal entity
        if entity is None:
            return
        label = identity(entity["name"])
        if label is None:
            skip("entity", entity, "missing_or_malformed_name")
        else:
            value = None if entity.get("duplicate_description") else description(entity["description"])
            node = {"id": label, "label": label, "description": value, "type": "extracted",
                "content": f"ENTITY: {label}\nDescription: {value if value is not None else 'null'}"}
            if label not in nodes or (value and not nodes[label]["description"]):
                nodes[label] = node
        entity = None

    def finish_relation():
        nonlocal relation
        if relation is None:
            return None
        source, target = identity(relation.get("source")), identity(relation.get("target"))
        inherited = None
        if not source or not target:
            skip("relationship", relation, "missing_or_malformed_endpoint")
        elif relation.get("duplicate_description"):
            skip("relationship", relation, "duplicate_description")
        elif source == target:
            skip("relationship", relation, "self_loop")
        else:
            value = description(relation["description"])
            edge = {"source": source, "target": target, "rel": "related_to", "description": value,
                "weight": 1.0, "type": "extracted"}
            key = source, target
            if key not in edges or (value and not edges[key]["description"]):
                edges[key] = edge
            # Only a completed, explicit Target/Description group authorizes
            # another Target in the same Source block. No cross-record guessing.
            if relation.get("description_seen"):
                inherited = source
        relation = None
        return inherited

    lines = []
    for number, raw in enumerate(body.splitlines(), 1):
        plain = raw.replace("**", "").replace("`", "")
        field = FIELD.match(plain)
        inline = INLINE.match("Source: " + field[2]) if field and field[1].lower() == "source" else None
        if inline:
            lines.extend((number, f"{name}: {value}") for name, value in zip(("Source", "Target", "Description"), inline.groups()))
            stats["compact_format_edges"] += 1
        else:
            lines.append((number, raw))

    for number, raw in lines:
        field = FIELD.match(raw.replace("**", "").replace("`", ""))
        if not field:
            if BOUNDARY.match(raw) or raw.strip() in {"NO_SUPPORTED_RECORDS", "No supported relationships reported."}:
                finish_entity()
                finish_relation()
                active = None
                if raw.strip() == "NO_SUPPORTED_RECORDS":
                    stats["syntax_errors"].append({"kind": "mixed_empty_sentinel", "line": number})
            elif active is not None and active.get("description_seen") and raw.strip():
                active["description"].append(raw.strip())
            continue
        name, value = field[1].lower(), field[2].strip()
        if name == "entity":
            finish_entity()
            finish_relation()
            entity = {"name": value, "line": number, "description": []}
            active = entity
        elif name == "relationships":
            finish_entity()
            finish_relation()
            active = None
        elif name == "source":
            finish_entity()
            finish_relation()
            relation = {"source": value, "line": number, "description": []}
            active = relation
        elif name == "target":
            finish_entity()
            if relation is None or "target" in relation:
                source = finish_relation()
                relation = {"source": source, "line": number, "description": []}
                if source:
                    stats["inherited_source_records"] += 1
                    stats["syntax_repairs"].append({"kind": "inherited_source", "line": number})
            relation["target"] = value
            active = relation
        elif active is not None:
            if active.get("description_seen"):
                active["duplicate_description"] = True
                stats["syntax_errors"].append({"kind": "duplicate_description", "line": number})
            else:
                active["description_seen"] = True
                active["description"] = [value]
        else:
            stats["syntax_errors"].append({"kind": "orphan_description", "line": number})
    finish_entity()
    finish_relation()
    stats.update(null_description_entities=sum(n["description"] is None for n in nodes.values()),
        null_description_edges=sum(e["description"] is None for e in edges.values()))
    if not nodes and not edges:
        stats.update(parse_status="skipped", round_skipped=True, skip_reason="no_parseable_records")
    elif stats["skipped_records"] or stats["syntax_errors"]:
        stats["parse_status"] = "partial"
    return list(nodes.values()), list(edges.values()), stats


def extraction_finish_reasons(request_path, turn):
    """Completion of the last successful extraction attempt, not writer/retries."""
    import json
    from pathlib import Path
    from .request_audit import request_stage
    path = Path(request_path)
    if not path.exists():
        # Reused initialization carries frozen completion metadata even when
        # its provider accounting remains in the originating cohort.
        sidecar = path.parent / "extraction_completion.json"
        return json.loads(sidecar.read_text())["finish_reasons"] if turn == 1 and sidecar.exists() else []
    successful = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if (row.get("turn") == turn and (row.get("stage") or request_stage(row)) == "extraction"
                and not row.get("error") and 200 <= (row.get("status") or 0) < 300):
            successful.append(row)
    return successful[-1].get("finish_reasons", []) if successful else []
