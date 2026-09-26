"""BNRR on the observed simple graph and cumulative response-graph updates."""
from __future__ import annotations
import bisect
import math
from dataclasses import dataclass
from typing import Mapping
import networkx as nx
from ..models import CandidateBatch, normalize_label


@dataclass(frozen=True)
class NodeScore:
    degree: int
    neighbor_edges: int
    clustering: float
    collision_probability: float
    effective_diversity: float
    burt_effective_size: float
    burt_balanced: float
    bnrr: float
    sensitivity: float


def simple_projection(graph: nx.Graph) -> nx.Graph:
    """Project a directed multi-relation graph to a loop-free simple graph."""

    projected = nx.Graph()
    projected.add_nodes_from(normalize_label(node) for node in graph.nodes)
    for source, target in graph.edges():
        u, v = normalize_label(source), normalize_label(target)
        if u and v and u != v:
            projected.add_edge(u, v)
    return projected


def _midrank_values(raw_scores: Mapping[str, float]) -> dict[str, float]:
    """Compute empirical midrank CDF values, assigning isolates score zero."""

    positive = sorted(value for value in raw_scores.values() if value > 0.0)
    count = len(positive)
    if count == 0:
        return {node: 0.0 for node in raw_scores}

    result: dict[str, float] = {}
    for node, value in raw_scores.items():
        if value <= 0.0:
            result[node] = 0.0
            continue
        lower = bisect.bisect_left(positive, value)
        upper = bisect.bisect_right(positive, value)
        result[node] = (lower + 0.5 * (upper - lower)) / count
    return result


def compute_node_scores(graph: nx.Graph) -> dict[str, NodeScore]:
    """Compute collision diversity, BNRR, and empirical sensitivity per node.

    For degree ``d`` and ``T`` edges among the neighbors, the exact redundant
    collision probability is ``(d + 2T) / d^2``.  Its inverse is the collision
    effective diversity, and BNRR is ``sqrt(d * e_col)``.
    """

    projected = simple_projection(graph)
    raw: dict[str, float] = {}
    local: dict[str, tuple[int, int, float, float, float, float, float]] = {}

    for node in projected.nodes:
        neighbors = set(projected.neighbors(node))
        degree = len(neighbors)
        if degree == 0:
            local[node] = (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            raw[node] = 0.0
            continue

        neighbor_edges = projected.subgraph(neighbors).number_of_edges()
        clustering = (
            2.0 * neighbor_edges / (degree * (degree - 1)) if degree >= 2 else 0.0
        )
        denominator = degree + 2 * neighbor_edges
        collision_probability = denominator / (degree * degree)
        effective_diversity = degree * degree / denominator
        # Burt effective size is the proposal's linear-redundancy competitor.
        burt_effective_size = degree - 2.0 * neighbor_edges / degree
        burt_balanced = math.sqrt(degree * burt_effective_size)
        bnrr = math.sqrt(degree * effective_diversity)
        local[node] = (
            degree,
            neighbor_edges,
            clustering,
            collision_probability,
            effective_diversity,
            burt_effective_size,
            burt_balanced,
        )
        raw[node] = bnrr

    sensitivity = _midrank_values(raw)
    return {
        node: NodeScore(*local[node], raw[node], sensitivity[node])
        for node in projected.nodes
    }


def merge_batch(graph: nx.MultiDiGraph, batch: CandidateBatch) -> None:
    """Merge a normalized batch into cumulative memory in place."""

    for label, attrs in batch.nodes.items():
        if label in graph:
            current = graph.nodes[label]
            if attrs.get("description") and not current.get("description"):
                current["description"] = attrs["description"]
        else:
            # JSON deltas retain null; GraphML has no null attribute type.
            graph.add_node(label, **{k: v for k, v in attrs.items() if v is not None})

    for atom, attrs in batch.edges.items():
        edge_attrs = {k: v for k, v in attrs.items() if v is not None}
        edge_attrs["rel"] = atom.relation
        current = graph.get_edge_data(atom.source, atom.target, key=atom.relation) or {}
        if not edge_attrs.get("description") and current.get("description"):
            edge_attrs["description"] = current["description"]
        graph.add_edge(atom.source, atom.target, key=atom.relation, **edge_attrs)
