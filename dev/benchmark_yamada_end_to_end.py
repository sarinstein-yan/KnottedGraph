from __future__ import annotations

import json
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial

A = sp.Symbol("A")


def multi_crossing_theta(component_count=3):
    """Exact nondegenerate fixture used by the regression test suite."""
    graph = nx.MultiGraph()

    for component in range(component_count):
        y_offset = 5.0 * component
        sign = 1.0 if component % 2 == 0 else -1.0
        left = f"u{component}"
        right = f"v{component}"
        graph.add_node(left, pos=np.array([-2.0, y_offset, 0.0]))
        graph.add_node(right, pos=np.array([2.0, y_offset, 0.0]))

        curves = [
            np.array([
                [-2.0, 0.0, 0.0],
                [-1.0, -1.0, 0.5 * sign],
                [1.0, 1.0, 0.5 * sign],
                [2.0, 0.0, 0.0],
            ]),
            np.array([
                [-2.0, 0.0, 0.0],
                [-1.0, 1.0, -0.5 * sign],
                [1.0, -1.0, -0.5 * sign],
                [2.0, 0.0, 0.0],
            ]),
            np.array([
                [-2.0, 0.0, 0.0],
                [-1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
            ]),
        ]

        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)

    return graph


def timed(fn, repeats=1):
    values = []
    answer = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - t0)
    return statistics.median(values), answer


def main():
    rows = []
    for crossings in range(1, 6):
        graph = multi_crossing_theta(crossings)
        elapsed, answer = timed(
            lambda: compute_yamada_polynomial(
                graph,
                A,
                rotation_angles=(0.0, 0.0, 0.0),
                normalize=True,
                n_jobs=1,
                method="negami",
                return_result=True,
            )
        )
        if answer.projection.num_crossings != crossings:
            raise AssertionError(
                f"fixture expected {crossings} crossings, got "
                f"{answer.projection.num_crossings}"
            )
        row = {
            "crossings": crossings,
            "states": 3**crossings,
            "V": graph.number_of_nodes(),
            "E": graph.number_of_edges(),
            "runtime_s": elapsed,
            "polynomial": str(sp.expand(answer.polynomial)),
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
