"""Auditable input-side evidence packets for grounded Ego Closure.

The packet builder deliberately uses only online endpoint labels, GraphRAG's
input-side entity-to-text-unit membership, and the raw text units.  Relationship
labels and truth edges are never consulted during admission or packing.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd
import tiktoken

from .control.actions import EntityPair
from .models import normalize_label


ACCOUNTING_TOKENIZER = "tiktoken==0.13.0/cl100k_base"
ACCOUNTING_ENCODING = "cl100k_base"
MAX_JOINT_WINDOW_TOKENS = 192
MAX_WINDOWS_PER_PAIR = 2


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _list_values(value: Any) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _neighbors(graph: nx.MultiDiGraph, node: str) -> set[str]:
    if graph.is_directed():
        values = (*graph.predecessors(node), *graph.successors(node))
    else:
        values = tuple(graph.neighbors(node))
    canonical = normalize_label(node)
    return {
        normalized
        for value in values
        for normalized in (normalize_label(value),)
        if normalized and normalized != canonical
    }


def _known_unordered_pairs(graph: nx.MultiDiGraph) -> set[EntityPair]:
    return {
        EntityPair.from_values(source, target)
        for source, target in graph.edges()
        if normalize_label(source)
        and normalize_label(target)
        and normalize_label(source) != normalize_label(target)
    }


def _abbreviations(title: str) -> tuple[str, ...]:
    values = {
        normalize_label(match)
        for match in re.findall(r"\(([A-Za-z0-9][A-Za-z0-9+./-]{1,11})\)", title)
    }
    return tuple(sorted(value for value in values if value))


def _mention_forms(
    endpoint: str,
    known_titles: frozenset[str],
) -> tuple[str, ...]:
    """Return exact title plus a unique, non-conflicting parenthetical alias."""

    alias_owners: dict[str, set[str]] = {}
    for title in known_titles:
        for alias in _abbreviations(title):
            alias_owners.setdefault(alias, set()).add(title)
    allowed_aliases = {
        alias
        for alias, owners in alias_owners.items()
        if owners == {endpoint} and alias not in known_titles
    }
    return (endpoint, *sorted(allowed_aliases))


def mention_forms_by_title(
    known_titles: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Build the conflict-aware mention map once for a complete online state."""

    titles = frozenset(normalize_label(title) for title in known_titles)
    alias_owners: dict[str, set[str]] = {}
    for title in titles:
        for alias in _abbreviations(title):
            alias_owners.setdefault(alias, set()).add(title)
    allowed_by_title: dict[str, list[str]] = {title: [] for title in titles}
    for alias, owners in alias_owners.items():
        if len(owners) == 1 and alias not in titles:
            allowed_by_title[next(iter(owners))].append(alias)
    return {
        title: (title, *sorted(allowed_by_title[title])) for title in titles
    }


def _find_mentions(text: str, forms: Sequence[str]) -> list[tuple[int, int, str]]:
    found: set[tuple[int, int, str]] = set()
    for form in forms:
        pieces = [re.escape(piece) for piece in form.split()]
        pattern = r"(?<![A-Za-z0-9])" + r"\s+".join(pieces) + r"(?![A-Za-z0-9])"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.add((match.start(), match.end(), form))
    return sorted(found)


def _char_span_to_token_span(
    offsets: Sequence[int],
    start: int,
    end: int,
) -> tuple[int, int]:
    token_start = max(0, bisect.bisect_right(offsets, start) - 1)
    token_end = bisect.bisect_left(offsets, end)
    return token_start, max(token_start + 1, token_end)


@dataclass(frozen=True)
class EvidenceWindow:
    text_unit_id: str
    text: str
    token_start: int
    token_end: int
    source_text_sha256: str
    left_mention: Mapping[str, Any]
    right_mention: Mapping[str, Any]

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_unit_id": self.text_unit_id,
            "tokenizer": ACCOUNTING_TOKENIZER,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "source_text_sha256": self.source_text_sha256,
            "left_mention": dict(self.left_mention),
            "right_mention": dict(self.right_mention),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceWindow":
        return cls(
            text_unit_id=str(value["text_unit_id"]),
            text=str(value["text"]),
            token_start=int(value["token_start"]),
            token_end=int(value["token_end"]),
            source_text_sha256=str(value["source_text_sha256"]),
            left_mention=dict(value["left_mention"]),
            right_mention=dict(value["right_mention"]),
        )


@dataclass(frozen=True)
class PairEvidencePacket:
    pair: EntityPair
    shared_text_unit_count: int
    common_neighbor_count: int
    windows: tuple[EvidenceWindow, ...]

    @property
    def shortest_window_tokens(self) -> int:
        return min(window.token_count for window in self.windows)

    @property
    def evidence_text_unit_ids(self) -> frozenset[str]:
        return frozenset(window.text_unit_id for window in self.windows)

    @property
    def rank_key(self) -> tuple[Any, ...]:
        return (
            -self.shared_text_unit_count,
            self.shortest_window_tokens,
            -self.common_neighbor_count,
            self.pair.left,
            self.pair.right,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.pair.left,
            "right": self.pair.right,
            "shared_text_unit_count": self.shared_text_unit_count,
            "common_neighbor_count": self.common_neighbor_count,
            "windows": [window.to_dict() for window in self.windows],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairEvidencePacket":
        return cls(
            pair=EntityPair(str(value["left"]), str(value["right"])),
            shared_text_unit_count=int(value["shared_text_unit_count"]),
            common_neighbor_count=int(value["common_neighbor_count"]),
            windows=tuple(
                EvidenceWindow.from_dict(window) for window in value["windows"]
            ),
        )


class EvidenceCorpus:
    """Read-only view over entity memberships and raw text units."""

    def __init__(self, data_dir: str | Path) -> None:
        root = Path(data_dir).resolve()
        entities = pd.read_parquet(
            root / "entities.parquet", columns=["title", "text_unit_ids"]
        )
        text_units = pd.read_parquet(
            root / "text_units.parquet", columns=["id", "text"]
        )
        memberships: dict[str, set[str]] = {}
        for record in entities.to_dict(orient="records"):
            title = normalize_label(record.get("title"))
            if title:
                memberships.setdefault(title, set()).update(
                    _list_values(record.get("text_unit_ids"))
                )
        self.memberships = {
            title: frozenset(values) for title, values in memberships.items()
        }
        self.text_units = {
            str(record["id"]): str(record["text"])
            for record in text_units.to_dict(orient="records")
        }
        self.encoding = tiktoken.get_encoding(ACCOUNTING_ENCODING)
        self._joint_window_cache: dict[
            tuple[str, tuple[str, ...], tuple[str, ...]], EvidenceWindow | None
        ] = {}

    def raw_open_pairs(
        self,
        graph: nx.MultiDiGraph,
        anchor: str,
        *,
        known_pairs: set[EntityPair] | None = None,
        neighbor_map: Mapping[str, set[str]] | None = None,
    ) -> tuple[EntityPair, ...]:
        known = known_pairs if known_pairs is not None else _known_unordered_pairs(graph)
        neighbors = (
            neighbor_map.get(anchor, set())
            if neighbor_map is not None
            else _neighbors(graph, anchor)
        )
        return tuple(
            pair
            for left, right in combinations(sorted(neighbors), 2)
            for pair in (EntityPair(left, right),)
            if pair not in known
        )

    def packets_for_anchor(
        self,
        graph: nx.MultiDiGraph,
        anchor: str,
        *,
        forms_by_title: Mapping[str, tuple[str, ...]] | None = None,
        neighbor_map: Mapping[str, set[str]] | None = None,
        known_pairs: set[EntityPair] | None = None,
    ) -> tuple[PairEvidencePacket, ...]:
        known_titles = frozenset(normalize_label(node) for node in graph.nodes)
        forms = forms_by_title or mention_forms_by_title(known_titles)
        packets = [
            packet
            for pair in self.raw_open_pairs(
                graph, anchor, known_pairs=known_pairs, neighbor_map=neighbor_map
            )
            for packet in (
                self.packet_for_pair(
                    graph,
                    pair,
                    known_titles,
                    forms_by_title=forms,
                    neighbor_map=neighbor_map,
                ),
            )
            if packet is not None
        ]
        return tuple(sorted(packets, key=lambda packet: packet.rank_key))

    def packet_for_pair(
        self,
        graph: nx.MultiDiGraph,
        pair: EntityPair,
        known_titles: frozenset[str] | None = None,
        *,
        forms_by_title: Mapping[str, tuple[str, ...]] | None = None,
        neighbor_map: Mapping[str, set[str]] | None = None,
    ) -> PairEvidencePacket | None:
        left_units = self.memberships.get(pair.left)
        right_units = self.memberships.get(pair.right)
        if not left_units or not right_units:
            return None
        shared = sorted(left_units & right_units)
        if not shared:
            return None
        titles = known_titles or frozenset(normalize_label(node) for node in graph.nodes)
        forms = forms_by_title or mention_forms_by_title(titles)
        left_forms = forms.get(pair.left, (pair.left,))
        right_forms = forms.get(pair.right, (pair.right,))
        windows = [
            window
            for unit_id in shared
            for window in (
                self._joint_window(unit_id, left_forms, right_forms),
            )
            if window is not None
        ]
        if not windows:
            return None
        windows.sort(
            key=lambda window: (
                window.token_count,
                window.text_unit_id,
                window.token_start,
            )
        )
        neighbors = neighbor_map or {}
        left_neighbors = neighbors.get(pair.left) or _neighbors(graph, pair.left)
        right_neighbors = neighbors.get(pair.right) or _neighbors(graph, pair.right)
        common_neighbors = len(left_neighbors & right_neighbors)
        return PairEvidencePacket(
            pair=pair,
            shared_text_unit_count=len(shared),
            common_neighbor_count=common_neighbors,
            windows=tuple(windows[:MAX_WINDOWS_PER_PAIR]),
        )

    def _joint_window(
        self,
        text_unit_id: str,
        left_forms: Sequence[str],
        right_forms: Sequence[str],
    ) -> EvidenceWindow | None:
        cache = getattr(self, "_joint_window_cache", None)
        if cache is None:
            cache = {}
            self._joint_window_cache = cache
        cache_key = (text_unit_id, tuple(left_forms), tuple(right_forms))
        if cache_key in cache:
            return cache[cache_key]
        source_text = self.text_units.get(text_unit_id)
        if source_text is None:
            cache[cache_key] = None
            return None
        left_mentions = _find_mentions(source_text, left_forms)
        right_mentions = _find_mentions(source_text, right_forms)
        if not left_mentions or not right_mentions:
            cache[cache_key] = None
            return None
        tokens = self.encoding.encode(source_text)
        decoded, offsets = self.encoding.decode_with_offsets(tokens)
        if decoded != source_text:
            raise ValueError(f"Tokenizer round-trip failed for text unit {text_unit_id}")

        candidates: list[tuple[Any, ...]] = []
        for left in left_mentions:
            left_token = _char_span_to_token_span(offsets, left[0], left[1])
            for right in right_mentions:
                right_token = _char_span_to_token_span(offsets, right[0], right[1])
                token_start = min(left_token[0], right_token[0])
                token_end = max(left_token[1], right_token[1])
                token_count = token_end - token_start
                if token_count <= MAX_JOINT_WINDOW_TOKENS:
                    candidates.append(
                        (
                            token_count,
                            token_start,
                            token_end,
                            left[0],
                            right[0],
                            left,
                            right,
                        )
                    )
        if not candidates:
            cache[cache_key] = None
            return None
        _, token_start, token_end, _, _, left, right = min(candidates)
        window_char_start = offsets[token_start]
        window_char_end = offsets[token_end] if token_end < len(offsets) else len(source_text)
        result = EvidenceWindow(
            text_unit_id=text_unit_id,
            text=self.encoding.decode(tokens[token_start:token_end]),
            token_start=token_start,
            token_end=token_end,
            source_text_sha256=_sha256_text(source_text),
            left_mention={
                "form": left[2],
                "char_start": left[0],
                "char_end": left[1],
                "window_char_start": left[0] - window_char_start,
                "window_char_end": left[1] - window_char_start,
            },
            right_mention={
                "form": right[2],
                "char_start": right[0],
                "char_end": right[1],
                "window_char_start": right[0] - window_char_start,
                "window_char_end": right[1] - window_char_start,
            },
        )
        cache[cache_key] = result
        return result

    def validate_window(self, window: EvidenceWindow) -> bool:
        source = self.text_units.get(window.text_unit_id)
        if source is None or _sha256_text(source) != window.source_text_sha256:
            return False
        tokens = self.encoding.encode(source)
        if not 0 <= window.token_start < window.token_end <= len(tokens):
            return False
        if window.token_count > MAX_JOINT_WINDOW_TOKENS:
            return False
        if self.encoding.decode(tokens[window.token_start : window.token_end]) != window.text:
            return False
        for mention in (window.left_mention, window.right_mention):
            start, end = int(mention["char_start"]), int(mention["char_end"])
            if not 0 <= start < end <= len(source):
                return False
            if normalize_label(source[start:end]) != normalize_label(mention["form"]):
                return False
        return True


def evidence_packets_json(packets: Sequence[PairEvidencePacket]) -> str:
    """Render only data the verifier is allowed to use."""

    payload = [
        {
            "left": packet.pair.left,
            "right": packet.pair.right,
            "evidence_windows": [window.to_dict() for window in packet.windows],
        }
        for packet in packets
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def parse_grounded_closure_response(
    response: str,
    packets: Sequence[PairEvidencePacket],
    corpus: EvidenceCorpus,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the JSON contract as one fail-closed response transaction."""

    expected = {packet.pair: packet for packet in packets}
    reasons: list[str] = []
    parsed: Any = None
    try:
        parsed = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        reasons.append("invalid_json")
    if parsed is not None:
        if not isinstance(parsed, dict) or set(parsed) != {"pairs"}:
            reasons.append("invalid_top_level_schema")
        elif not isinstance(parsed["pairs"], list):
            reasons.append("pairs_not_list")

    records = parsed.get("pairs", []) if isinstance(parsed, dict) else []
    seen: set[EntityPair] = set()
    provisional_edges: list[dict[str, Any]] = []
    supported_count = 0
    unsupported_count = 0
    valid_keys = {
        "left",
        "right",
        "supported",
        "source",
        "target",
        "description",
        "evidence_text_unit_ids",
    }
    if isinstance(records, list):
        for index, record in enumerate(records):
            prefix = f"pair_{index}"
            if not isinstance(record, dict) or set(record) != valid_keys:
                reasons.append(f"{prefix}:invalid_object_schema")
                continue
            try:
                pair = EntityPair(str(record["left"]), str(record["right"]))
            except ValueError:
                reasons.append(f"{prefix}:invalid_pair")
                continue
            if pair not in expected:
                reasons.append(f"{prefix}:off_pair")
                continue
            if pair in seen:
                reasons.append(f"{prefix}:duplicate_pair")
                continue
            seen.add(pair)
            if type(record["supported"]) is not bool:
                reasons.append(f"{prefix}:supported_not_boolean")
                continue
            citations = record["evidence_text_unit_ids"]
            if not isinstance(citations, list) or not all(
                isinstance(value, str) for value in citations
            ):
                reasons.append(f"{prefix}:invalid_citation_list")
                continue
            if not record["supported"]:
                unsupported_count += 1
                if any(
                    record[field] not in (None, "")
                    for field in ("source", "target", "description")
                ) or citations:
                    reasons.append(f"{prefix}:unsupported_fields_not_empty")
                continue

            supported_count += 1
            source = normalize_label(record["source"])
            target = normalize_label(record["target"])
            description = str(record["description"] or "").strip()
            if not source or not target or source == target:
                reasons.append(f"{prefix}:invalid_endpoints")
                continue
            if not pair.contains_directed(source, target):
                reasons.append(f"{prefix}:endpoint_mismatch")
                continue
            if not description:
                reasons.append(f"{prefix}:empty_description")
                continue
            if _negative_description(description):
                reasons.append(f"{prefix}:negative_description")
                continue
            packet = expected[pair]
            allowed = packet.evidence_text_unit_ids
            if not citations or any(citation not in allowed for citation in citations):
                reasons.append(f"{prefix}:invalid_citation")
                continue
            cited_windows = {
                window.text_unit_id: window for window in packet.windows
            }
            if any(
                not corpus.validate_window(cited_windows[citation])
                for citation in citations
            ):
                reasons.append(f"{prefix}:evidence_integrity_failure")
                continue
            provisional_edges.append(
                {
                    "source": source,
                    "target": target,
                    "rel": "related_to",
                    "description": description,
                    "evidence_text_unit_ids": citations,
                }
            )

    missing = set(expected) - seen
    if missing:
        reasons.append("missing_pairs")
    if len(records) != len(expected):
        reasons.append("pair_count_mismatch")
    schema_valid = not reasons
    accepted = provisional_edges if schema_valid else []
    stats = {
        "closure_parser": "grounded_json_v1",
        "closure_schema_valid": schema_valid,
        "closure_schema_rejection_reasons": sorted(set(reasons)),
        "closure_expected_pair_count": len(expected),
        "closure_reported_pair_count": len(seen),
        "closure_pair_coverage": len(seen) / len(expected) if expected else 0.0,
        "closure_supported_objects": supported_count,
        "closure_unsupported_objects": unsupported_count,
        "closure_accepted_edges": len(accepted),
        "closure_fail_closed_edges": len(provisional_edges) - len(accepted),
        "closure_negative_accepted_edges": 0,
    }
    return accepted, stats


def _negative_description(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    patterns = (
        r"\bno (?:direct )?relationship\b",
        r"\bnot (?:directly )?(?:related|associated|connected)\b",
        r"\b(?:cannot|can't|unable to) (?:determine|confirm|establish)\b",
        r"\b(?:insufficient|no) evidence\b",
        r"\b(?:unknown|unclear|unsupported)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def count_tokens(value: str) -> int:
    return len(tiktoken.get_encoding(ACCOUNTING_ENCODING).encode(value))


def packet_truth_free_manifest(
    packets: Iterable[PairEvidencePacket],
) -> list[dict[str, Any]]:
    return [packet.to_dict() for packet in packets]
