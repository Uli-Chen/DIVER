"""Typed data structures shared by extraction and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


def normalize_label(value: Any) -> str:
    """Apply the same case-insensitive entity normalization used by AGEA."""

    raw = "" if value is None else str(value)
    return " ".join(raw.strip().split()).upper()


def normalize_relation(value: Any) -> str:
    """Normalize relation identifiers without altering their semantic wording."""

    normalized = " ".join(str(value or "related_to").strip().split()).lower()
    return normalized or "related_to"


@dataclass(frozen=True, order=True)
class EdgeAtom:
    """A normalized directed relation triple used for exact novelty checks."""

    source: str
    relation: str
    target: str

    @property
    def pair(self) -> tuple[str, str]:
        """Return the directed endpoint pair used by GraphRAG truth evaluation."""

        return self.source, self.target


@dataclass
class CandidateBatch:
    """A normalized, within-batch deduplicated set of nodes and relation triples."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[EdgeAtom, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        nodes: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        *,
        label_resolver: Callable[[str], str] | None = None,
    ) -> "CandidateBatch":
        """Build a batch and include edge endpoints in its candidate node set."""

        resolve = label_resolver or (lambda label: label)

        def canonical(value: Any) -> str:
            normalized = normalize_label(value)
            return normalize_label(resolve(normalized)) if normalized else ""

        node_map: dict[str, dict[str, Any]] = {}
        for record in nodes:
            label = canonical(record.get("label") or record.get("id"))
            if not label:
                continue
            attrs = dict(record)
            attrs["label"] = label
            attrs["id"] = label
            node_map.setdefault(label, attrs)

        edge_map: dict[EdgeAtom, dict[str, Any]] = {}
        for record in edges:
            source = canonical(record.get("source") or record.get("src"))
            target = canonical(record.get("target") or record.get("dst"))
            if not source or not target:
                continue
            relation = normalize_relation(record.get("rel") or record.get("relation"))
            atom = EdgeAtom(source, relation, target)
            attrs = dict(record)
            attrs.update({"source": source, "target": target, "rel": relation})
            edge_map.setdefault(atom, attrs)
            if attrs.get("description") and not edge_map[atom].get("description"):
                edge_map[atom]["description"] = attrs["description"]

            # AGEA's response format may mention an endpoint only in a relationship.
            node_map.setdefault(source, {"id": source, "label": source, "type": "endpoint"})
            node_map.setdefault(target, {"id": target, "label": target, "type": "endpoint"})

        return cls(nodes=node_map, edges=edge_map)

    def to_records(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return JSON-serializable node and edge records."""

        return list(self.nodes.values()), list(self.edges.values())

    @property
    def atoms(self) -> tuple[set[str], set[EdgeAtom]]:
        return set(self.nodes), set(self.edges)


def edge_atoms_from_graph(graph: Any) -> set[EdgeAtom]:
    """Read normalized relation triples from a NetworkX graph."""

    atoms: set[EdgeAtom] = set()
    if graph.is_multigraph():
        iterator: Iterable[tuple[Any, Any, Any, Mapping[str, Any]]] = graph.edges(
            keys=True, data=True
        )
        for source, target, key, attrs in iterator:
            relation = attrs.get("rel") or attrs.get("relation") or key
            atoms.add(
                EdgeAtom(
                    normalize_label(source),
                    normalize_relation(relation),
                    normalize_label(target),
                )
            )
    else:
        for source, target, attrs in graph.edges(data=True):
            relation = attrs.get("rel") or attrs.get("relation") or "related_to"
            atoms.add(
                EdgeAtom(
                    normalize_label(source),
                    normalize_relation(relation),
                    normalize_label(target),
                )
            )
    return atoms
