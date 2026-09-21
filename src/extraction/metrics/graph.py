"""BNRR, historical anchoring, HTSN, and retrospective scoring.

The implemented definitions are documented in the repository-level
``proposal.md``. All topology statistics are computed on a simple undirected
projection: directions, relation types, and parallel edges are retained in
memory but collapsed for local topology only.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import networkx as nx

from ..models import CandidateBatch, EdgeAtom, edge_atoms_from_graph, normalize_label


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


@dataclass
class BatchScore:
    candidate_nodes: int
    candidate_edges: int
    new_nodes: int
    new_edges: int
    historical_node_mass: float
    historical_edge_mass: float
    historical_edge_mass_noisy_or: float
    y_ha: float
    htsn_node: float
    htsn_edge: float
    htsn_combined: float
    htsn_edge_noisy_or: float
    htsn_combined_noisy_or: float
    count_novelty: float
    retrospective_node_mass: float
    retrospective_edge_mass: float
    retrospective_edge_mass_noisy_or: float
    rtsn_combined: float
    rtsn_combined_noisy_or: float
    retrospective_gain: float
    new_node_weights: dict[str, float]
    new_edge_weights: dict[EdgeAtom, float]
    new_edge_weights_noisy_or: dict[EdgeAtom, float]
    arrival_bnrr: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        # ``dataclasses.asdict`` recursively turns EdgeAtom dictionary keys into
        # dictionaries, which are unhashable.  Copy fields shallowly and encode
        # only those keys explicitly for JSON.
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["new_edge_weights"] = {
            "\t".join((edge.source, edge.relation, edge.target)): value
            for edge, value in self.new_edge_weights.items()
        }
        payload["new_edge_weights_noisy_or"] = {
            "\t".join((edge.source, edge.relation, edge.target)): value
            for edge, value in self.new_edge_weights_noisy_or.items()
        }
        return payload


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


def midrank_cdf(value: float, reference: list[float]) -> float:
    """Evaluate the historical empirical midrank CDF at an arrival BNRR value."""

    positive = sorted(score for score in reference if score > 0.0)
    if value <= 0.0 or not positive:
        return 0.0
    lower = bisect.bisect_left(positive, value)
    upper = bisect.bisect_right(positive, value)
    return (lower + 0.5 * (upper - lower)) / len(positive)


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


def _arrival_bnrr(
    node: str,
    batch: CandidateBatch,
    historical_nodes: set[str],
    historical_projection: nx.Graph,
) -> float:
    """Score a new node using only its links to the pre-turn graph.

    Edges between two nodes arriving in the same batch are intentionally ignored;
    this is the historical-anchoring rule that prevents batch self-credit.
    """

    anchors: set[str] = set()
    for edge in batch.edges:
        if edge.source == node and edge.target in historical_nodes:
            anchors.add(edge.target)
        elif edge.target == node and edge.source in historical_nodes:
            anchors.add(edge.source)

    anchor_count = len(anchors)
    if anchor_count == 0:
        return 0.0
    anchor_edges = historical_projection.subgraph(anchors).number_of_edges()
    return anchor_count ** 1.5 / math.sqrt(anchor_count + 2 * anchor_edges)


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


def _delta(
    graph: nx.MultiDiGraph, batch: CandidateBatch
) -> tuple[set[str], set[EdgeAtom]]:
    historical_nodes = {normalize_label(node) for node in graph.nodes}
    historical_edges = edge_atoms_from_graph(graph)
    return set(batch.nodes) - historical_nodes, set(batch.edges) - historical_edges


def compute_batch_scores(graph: nx.MultiDiGraph, batch: CandidateBatch) -> BatchScore:
    """Compute pre-merge HTSN and post-merge retrospective TSN for one batch."""

    historical_nodes = {normalize_label(node) for node in graph.nodes}
    historical_projection = simple_projection(graph)
    historical_scores = compute_node_scores(graph)
    historical_raw = [score.bnrr for score in historical_scores.values()]
    new_nodes, new_edges = _delta(graph, batch)

    node_weights: dict[str, float] = {}
    arrival_bnrr: dict[str, float] = {}
    for node in new_nodes:
        raw = _arrival_bnrr(node, batch, historical_nodes, historical_projection)
        arrival_bnrr[node] = raw
        node_weights[node] = midrank_cdf(raw, historical_raw)

    def endpoint_weight(node: str) -> float:
        if node in historical_scores:
            return historical_scores[node].sensitivity
        return node_weights.get(node, 0.0)

    edge_weights = {
        edge: max(endpoint_weight(edge.source), endpoint_weight(edge.target))
        for edge in new_edges
    }
    edge_weights_noisy_or = {
        edge: 1.0
        - (1.0 - endpoint_weight(edge.source))
        * (1.0 - endpoint_weight(edge.target))
        for edge in new_edges
    }
    node_mass = sum(node_weights.values())
    edge_mass = sum(edge_weights.values())
    edge_mass_noisy_or = sum(edge_weights_noisy_or.values())
    candidate_node_count = len(batch.nodes)
    candidate_edge_count = len(batch.edges)
    atom_count = candidate_node_count + candidate_edge_count

    htsn_node = node_mass / candidate_node_count if candidate_node_count else 0.0
    htsn_edge = edge_mass / candidate_edge_count if candidate_edge_count else 0.0
    htsn_combined = (node_mass + edge_mass) / atom_count if atom_count else 0.0
    htsn_edge_noisy_or = (
        edge_mass_noisy_or / candidate_edge_count if candidate_edge_count else 0.0
    )
    htsn_combined_noisy_or = (
        (node_mass + edge_mass_noisy_or) / atom_count if atom_count else 0.0
    )
    count_novelty = (len(new_nodes) + len(new_edges)) / atom_count if atom_count else 0.0

    merged = graph.copy()
    merge_batch(merged, batch)
    retrospective_scores = compute_node_scores(merged)
    retrospective_node_mass = sum(
        retrospective_scores[node].sensitivity for node in new_nodes
    )
    retrospective_edge_mass = sum(
        max(
            retrospective_scores[edge.source].sensitivity,
            retrospective_scores[edge.target].sensitivity,
        )
        for edge in new_edges
    )
    retrospective_edge_mass_noisy_or = sum(
        1.0
        - (1.0 - retrospective_scores[edge.source].sensitivity)
        * (1.0 - retrospective_scores[edge.target].sensitivity)
        for edge in new_edges
    )
    rtsn_combined = (
        (retrospective_node_mass + retrospective_edge_mass) / atom_count
        if atom_count
        else 0.0
    )
    rtsn_combined_noisy_or = (
        (retrospective_node_mass + retrospective_edge_mass_noisy_or) / atom_count
        if atom_count
        else 0.0
    )

    return BatchScore(
        candidate_nodes=candidate_node_count,
        candidate_edges=candidate_edge_count,
        new_nodes=len(new_nodes),
        new_edges=len(new_edges),
        historical_node_mass=node_mass,
        historical_edge_mass=edge_mass,
        historical_edge_mass_noisy_or=edge_mass_noisy_or,
        y_ha=node_mass + edge_mass,
        htsn_node=htsn_node,
        htsn_edge=htsn_edge,
        htsn_combined=htsn_combined,
        htsn_edge_noisy_or=htsn_edge_noisy_or,
        htsn_combined_noisy_or=htsn_combined_noisy_or,
        count_novelty=count_novelty,
        retrospective_node_mass=retrospective_node_mass,
        retrospective_edge_mass=retrospective_edge_mass,
        retrospective_edge_mass_noisy_or=retrospective_edge_mass_noisy_or,
        rtsn_combined=rtsn_combined,
        rtsn_combined_noisy_or=rtsn_combined_noisy_or,
        retrospective_gain=rtsn_combined - htsn_combined,
        new_node_weights=node_weights,
        new_edge_weights=edge_weights,
        new_edge_weights_noisy_or=edge_weights_noisy_or,
        arrival_bnrr=arrival_bnrr,
    )


def ftp_scores(
    graph: nx.Graph, query_counts: Mapping[str, int]
) -> dict[str, float]:
    """Frequency-penalized topology baseline: sensitivity / (1 + queries)."""

    scores = compute_node_scores(graph)
    return {
        node: score.sensitivity / (1 + query_counts.get(node, 0))
        for node, score in scores.items()
    }
