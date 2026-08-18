from __future__ import annotations

import collections
import json
import time

import networkx as nx
import numpy as np

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode


def _spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def _petersen_calculator() -> Yamada:
    processor = PDCode(_spring_embedding(nx.petersen_graph(), 9))
    processor.compute(
        rotation_angles=(-134.58074129795634, 55.40942502382338, 0.0)
    )
    if len(processor.crossings) != 6:
        raise AssertionError(f"expected Petersen c=6, got {len(processor.crossings)}")
    return Yamada.from_PDCode(processor)


def _attributed_simple(graph: CompactGraph) -> nx.Graph:
    out = nx.Graph()
    for node in range(graph.n):
        out.add_node(node, loops=int(graph.rows[node][node]))
    for i in range(graph.n):
        for j in range(i + 1, graph.n):
            multiplicity = int(graph.rows[i][j])
            if multiplicity:
                out.add_edge(i, j, multiplicity=multiplicity)
    return out


def _cheap_signature(graph: CompactGraph):
    nx_graph = _attributed_simple(graph)
    wl = nx.weisfeiler_lehman_graph_hash(
        nx_graph,
        node_attr="loops",
        edge_attr="multiplicity",
        iterations=4,
    )
    return (
        graph.n,
        graph.edge_count,
        tuple(sorted(graph.degree(i) for i in range(graph.n))),
        tuple(sorted(graph.rows[i][i] for i in range(graph.n))),
        wl,
    )


def _exact_isomorphic(left: CompactGraph, right: CompactGraph) -> bool:
    left_nx = _attributed_simple(left)
    right_nx = _attributed_simple(right)
    return nx.is_isomorphic(
        left_nx,
        right_nx,
        node_match=nx.algorithms.isomorphism.categorical_node_match("loops", 0),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("multiplicity", 0),
    )


def _fill_memo():
    calculator = _petersen_calculator()
    evaluator = PythonCompactYamadaEvaluator()
    start = time.perf_counter()
    for graph, _ in calculator._iter_compact_states():
        evaluator.compute_laurent(graph)
    return evaluator, time.perf_counter() - start


def main():
    evaluator, evaluation_s = _fill_memo()
    graphs = list(evaluator.memo)
    buckets: dict[object, list[CompactGraph]] = collections.defaultdict(list)
    for graph in graphs:
        buckets[_cheap_signature(graph)].append(graph)

    potential_buckets = [bucket for bucket in buckets.values() if len(bucket) > 1]
    exact_classes = 0
    exact_representatives = 0
    exact_duplicates = 0
    comparisons = 0
    start = time.perf_counter()
    for bucket in potential_buckets:
        representatives: list[CompactGraph] = []
        counts: list[int] = []
        for graph in bucket:
            for index, representative in enumerate(representatives):
                comparisons += 1
                if _exact_isomorphic(graph, representative):
                    counts[index] += 1
                    break
            else:
                representatives.append(graph)
                counts.append(1)
        exact_classes += len(representatives)
        exact_representatives += len(representatives)
        exact_duplicates += sum(count - 1 for count in counts)
    iso_s = time.perf_counter() - start

    singleton_buckets = sum(1 for bucket in buckets.values() if len(bucket) == 1)
    unique_up_to_iso = singleton_buckets + exact_representatives
    result = {
        "case": "petersen_c6_structural_memo",
        "labeled_memo_entries": len(graphs),
        "signature_buckets": len(buckets),
        "potential_collision_buckets": len(potential_buckets),
        "exact_isomorphic_duplicates": exact_duplicates,
        "unique_up_to_exact_isomorphism": unique_up_to_iso,
        "maximum_possible_memo_reduction_fraction": (
            exact_duplicates / len(graphs) if graphs else 0.0
        ),
        "exact_isomorphism_comparisons": comparisons,
        "state_evaluation_s": evaluation_s,
        "posthoc_exact_isomorphism_analysis_s": iso_s,
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
