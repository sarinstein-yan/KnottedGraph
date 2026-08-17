from __future__ import annotations

import json
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def median(fn, repeats=5):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), result


def spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    pos = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    out = nx.MultiGraph()
    for node, point in pos.items():
        out.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        out.add_edge(u, v, pts=np.vstack([pos[u], pos[v]]))
    return out


def multi_crossing_theta(component_count=5):
    graph = nx.MultiGraph()
    for component in range(component_count):
        y_offset = 5.0 * component
        sign = 1.0 if component % 2 == 0 else -1.0
        left = f"u{component}"
        right = f"v{component}"
        graph.add_node(left, pos=np.array([-2.0, y_offset, 0.0]))
        graph.add_node(right, pos=np.array([2.0, y_offset, 0.0]))
        curves = [
            np.array([[-2, 0, 0], [-1, -1, 0.5 * sign], [1, 1, 0.5 * sign], [2, 0, 0]], dtype=float),
            np.array([[-2, 0, 0], [-1, 1, -0.5 * sign], [1, -1, -0.5 * sign], [2, 0, 0]], dtype=float),
            np.array([[-2, 0, 0], [-1, 2, 0], [1, 2, 0], [2, 0, 0]], dtype=float),
        ]
        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)
    return graph


def measure_case(name, graph, rotation, expected_crossings):
    def projection_only():
        processor = PDCode(graph)
        processor.compute(rotation_angles=rotation)
        return processor

    projection_s, processor = median(projection_only, 7)
    crossings = len(processor.crossings)
    if crossings != expected_crossings:
        raise AssertionError(f"{name}: expected {expected_crossings} crossings, got {crossings}")
    if len(Yamada.from_PDCode(processor)._diagram_blocks()) != (5 if name == "decomposable_c5" else 1):
        raise AssertionError(f"{name}: unexpected block factorization")

    invariant_s, polynomial = median(
        lambda: Yamada.from_PDCode(processor).compute(A, normalize=False, n_jobs=1, method="negami"),
        7,
    )

    def complete():
        p = PDCode(graph)
        p.compute(rotation_angles=rotation)
        return Yamada.from_PDCode(p).compute(A, normalize=False, n_jobs=1, method="negami")

    total_s, total_poly = median(complete, 7)
    if sp.expand(polynomial - total_poly) != 0:
        raise AssertionError("pipeline breakdown changed result")
    row = {
        "case": name,
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "crossings": crossings,
        "projection_s": projection_s,
        "invariant_s": invariant_s,
        "total_s": total_s,
        "projection_fraction": projection_s / total_s,
        "invariant_fraction": invariant_s / total_s,
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def main():
    rows = []
    rows.append(measure_case("decomposable_c5", multi_crossing_theta(5), (0.0, 0.0, 0.0), 5))

    k4 = spring_embedding(nx.complete_graph(4), 7)
    rows.append(measure_case("connected_K4", k4, (0.0, 89.70158313251306, 0.0), 1))

    petersen = spring_embedding(nx.petersen_graph(), 9)
    rows.append(
        measure_case(
            "connected_petersen",
            petersen,
            (-134.58074129795634, 55.40942502382338, 0.0),
            6,
        )
    )
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
