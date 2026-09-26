"""Ground-truth graph recovery metrics for DIVER."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd

from extraction.metrics.graph import NodeScore, compute_node_scores
from extraction.models import normalize_label


@dataclass
class TruthData:
    """Frozen GraphRAG truth and the weights derived from its topology."""

    graph: nx.DiGraph
    nodes: set[str]
    edges: set[tuple[str, str]]
    aliases: dict[str, str]
    node_scores: dict[str, NodeScore]
    pagerank: dict[str, float]

    @classmethod
    def load(cls, output_dir: str | Path) -> "TruthData":
        output_path = Path(output_dir)
        entities = pd.read_parquet(output_path / "entities.parquet")
        relationships = pd.read_parquet(output_path / "relationships.parquet")

        graph = nx.DiGraph()
        for record in entities.to_dict(orient="records"):
            label = normalize_label(record.get("title"))
            if label:
                graph.add_node(label, **_json_safe_attributes(record))
        for record in relationships.to_dict(orient="records"):
            source = normalize_label(record.get("source"))
            target = normalize_label(record.get("target"))
            if source and target:
                graph.add_edge(source, target, **_json_safe_attributes(record))

        nodes = set(graph.nodes)
        edges = set(graph.edges)
        aliases = _build_alias_map(nodes)
        node_scores = compute_node_scores(graph)
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        return cls(graph, nodes, edges, aliases, node_scores, pagerank)

    def canonical(self, value: str) -> str:
        normalized = normalize_label(value)
        if normalized in self.nodes:
            return normalized
        return self.aliases.get(normalized, normalized)

    @property
    def node_rank_weights(self) -> dict[str, float]:
        return {node: score.sensitivity for node, score in self.node_scores.items()}

    @property
    def node_raw_weights(self) -> dict[str, float]:
        return {node: score.bnrr for node, score in self.node_scores.items()}

    @property
    def degree_weights(self) -> dict[str, float]:
        # Count degree on the simple undirected projection.
        return {node: float(score.degree) for node, score in self.node_scores.items()}

    @property
    def burt_effective_weights(self) -> dict[str, float]:
        return {
            node: score.burt_effective_size for node, score in self.node_scores.items()
        }

    @property
    def collision_diversity_weights(self) -> dict[str, float]:
        return {
            node: score.effective_diversity for node, score in self.node_scores.items()
        }

    @property
    def burt_balanced_weights(self) -> dict[str, float]:
        return {node: score.burt_balanced for node, score in self.node_scores.items()}

    def edge_weights(self, kind: str = "rank") -> dict[tuple[str, str], float]:
        node_weights = self.node_rank_weights if kind == "rank" else self.node_raw_weights
        return {
            edge: max(node_weights.get(edge[0], 0.0), node_weights.get(edge[1], 0.0))
            for edge in self.edges
        }

    def edge_noisy_or_rank_weights(self) -> dict[tuple[str, str], float]:
        node_weights = self.node_rank_weights
        return {
            edge: 1.0
            - (1.0 - node_weights.get(edge[0], 0.0))
            * (1.0 - node_weights.get(edge[1], 0.0))
            for edge in self.edges
        }


def _json_safe_attributes(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep useful scalar parquet fields out of NumPy/object serialization traps."""

    result: dict[str, Any] = {}
    for key, value in record.items():
        if key == "text_unit_ids":
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
    return result


def _build_alias_map(nodes: set[str]) -> dict[str, str]:
    """Build conservative aliases from unique parenthesized abbreviations."""

    candidates: dict[str, set[str]] = {}
    for node in sorted(nodes):
        if "(" not in node or ")" not in node:
            continue
        start, end = node.find("("), node.rfind(")")
        abbreviation = normalize_label(node[start + 1 : end])
        if abbreviation and len(abbreviation) <= 10:
            candidates.setdefault(abbreviation, set()).add(node)
    return {
        alias: next(iter(values))
        for alias, values in candidates.items()
        if len(values) == 1 and alias not in nodes
    }


def weighted_coverage(
    recovered: set[Any], truth: set[Any], weights: Mapping[Any, float]
) -> float:
    denominator = sum(weights.get(item, 0.0) for item in truth)
    if denominator <= 0.0:
        return 0.0
    return sum(weights.get(item, 0.0) for item in recovered & truth) / denominator


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks with exact tie handling."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute Spearman's rho without adding SciPy as a project dependency."""

    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = _average_ranks(left), _average_ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    scale_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    scale_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    return covariance / (scale_x * scale_y)


def autc(recovered: set[str], truth_weights: Mapping[str, float]) -> float:
    """Average top-k recall over the complete deterministic truth ranking."""

    ranking = sorted(truth_weights, key=lambda node: (-truth_weights[node], node))
    if not ranking:
        return 0.0
    cumulative_hits = 0
    recalls: list[float] = []
    for k, node in enumerate(ranking, start=1):
        cumulative_hits += int(node in recovered)
        recalls.append(cumulative_hits / k)
    return sum(recalls) / len(recalls)


def _topk_hits(
    recovered: set[str], weights: Mapping[str, float], values: Iterable[int]
) -> dict[str, int]:
    ranking = sorted(weights, key=lambda node: (-weights[node], node))
    return {
        f"top{k}_sensitive_nodes_recovered": len(recovered & set(ranking[:k]))
        for k in values
    }


def evaluate_recovery(
    recovered_graph: nx.MultiDiGraph,
    truth: TruthData,
) -> dict[str, Any]:
    """Evaluate cumulative recovery using count, topology, and ranking metrics."""

    recovered_nodes = {truth.canonical(str(node)) for node in recovered_graph.nodes}
    recovered_edges = {
        (truth.canonical(str(source)), truth.canonical(str(target)))
        for source, target in recovered_graph.edges()
    }
    matched_nodes = recovered_nodes & truth.nodes
    matched_edges = recovered_edges & truth.edges

    truth_rank = truth.node_rank_weights
    truth_raw = truth.node_raw_weights
    edge_rank = truth.edge_weights("rank")
    edge_raw = truth.edge_weights("raw")
    edge_noisy_or_rank = truth.edge_noisy_or_rank_weights()
    online = compute_node_scores(recovered_graph)

    online_by_canonical: dict[str, NodeScore] = {}
    for node, online_score in online.items():
        canonical = truth.canonical(node)
        if canonical in truth.node_scores:
            online_by_canonical.setdefault(canonical, online_score)
    comparable = [
        (online_by_canonical[node], truth.node_scores[node])
        for node in sorted(online_by_canonical)
    ]

    undirected_truth = {frozenset(edge) for edge in truth.edges}
    undirected_recovered = {frozenset(edge) for edge in recovered_edges}
    undirected_matches = undirected_truth & undirected_recovered

    node_precision = _safe_ratio(len(matched_nodes), len(recovered_nodes))
    node_recall = _safe_ratio(len(matched_nodes), len(truth.nodes))
    edge_pair_precision = _safe_ratio(len(matched_edges), len(recovered_edges))
    edge_pair_recall = _safe_ratio(len(matched_edges), len(truth.edges))
    node_f1 = _f1(node_precision, node_recall)
    edge_pair_f1 = _f1(edge_pair_precision, edge_pair_recall)
    metrics: dict[str, Any] = {
        "recovered_nodes": len(recovered_nodes),
        "recovered_edge_pairs": len(recovered_edges),
        "matched_nodes": len(matched_nodes),
        "matched_directed_edge_pairs": len(matched_edges),
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_pair_precision": edge_pair_precision,
        "edge_pair_recall": edge_pair_recall,
        "edge_pair_f1": edge_pair_f1,
        "balanced_graph_quality": node_f1 * edge_pair_f1,
        "undirected_edge_pair_precision": _safe_ratio(
            len(undirected_matches), len(undirected_recovered)
        ),
        "undirected_edge_pair_recall": _safe_ratio(
            len(undirected_matches), len(undirected_truth)
        ),
        "node_tsc_rank": weighted_coverage(matched_nodes, truth.nodes, truth_rank),
        "node_tsc_raw": weighted_coverage(matched_nodes, truth.nodes, truth_raw),
        "edge_tsc_rank": weighted_coverage(matched_edges, truth.edges, edge_rank),
        "edge_tsc_raw": weighted_coverage(matched_edges, truth.edges, edge_raw),
        "edge_tsc_noisy_or_rank": weighted_coverage(
            matched_edges, truth.edges, edge_noisy_or_rank
        ),
        "node_tsc_degree": weighted_coverage(
            matched_nodes, truth.nodes, truth.degree_weights
        ),
        "node_tsc_burt_effective": weighted_coverage(
            matched_nodes, truth.nodes, truth.burt_effective_weights
        ),
        "node_tsc_collision_diversity": weighted_coverage(
            matched_nodes, truth.nodes, truth.collision_diversity_weights
        ),
        "node_tsc_burt_balanced": weighted_coverage(
            matched_nodes, truth.nodes, truth.burt_balanced_weights
        ),
        "degree_weighted_node_coverage": weighted_coverage(
            matched_nodes, truth.nodes, truth.degree_weights
        ),
        "pagerank_weighted_node_coverage": weighted_coverage(
            matched_nodes, truth.nodes, truth.pagerank
        ),
        "node_autc_rank": autc(matched_nodes, truth_rank),
        "spearman_bnrr_online_vs_truth": spearman_correlation(
            [item[0].bnrr for item in comparable],
            [item[1].bnrr for item in comparable],
        ),
        "spearman_sensitivity_online_vs_truth": spearman_correlation(
            [item[0].sensitivity for item in comparable],
            [item[1].sensitivity for item in comparable],
        ),
        "spearman_matched_nodes": len(comparable),
    }
    metrics.update(_topk_hits(matched_nodes, truth_rank, (5, 10, 20, 50)))
    return metrics


def aggregate_trajectory(turn_metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate budget-indexed curves; AUTSC is their arithmetic mean."""

    if not turn_metrics:
        return {}
    curve_keys = (
        "node_tsc_rank",
        "node_tsc_raw",
        "edge_tsc_rank",
        "edge_tsc_raw",
        "node_recall",
        "edge_pair_recall",
        "node_f1",
        "edge_pair_f1",
        "balanced_graph_quality",
        "node_autc_rank",
    )
    return {
        f"au_{key}": sum(float(turn.get(key, 0.0)) for turn in turn_metrics)
        / len(turn_metrics)
        for key in curve_keys
    }
