from __future__ import annotations

import argparse
import json
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial


def coords_ring(n: int) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    r = 5 * (1 + 0.05 * np.sin(3 * t + 0.2))
    z = 0.8 * (np.sin(3 * t + 0.31) + 0.3 * np.cos(5 * t - 0.2))
    return np.c_[r * np.cos(t), r * np.sin(t), z]


def matching(n: int) -> list[tuple[int, int]]:
    # Deterministic non-adjacent perfect matching with modest projected crossing count.
    candidates = []
    for shift in range(2, n // 2 + 1):
        used = set()
        edges = []
        ok = True
        for i in range(n):
            if i in used:
                continue
            j = (i + shift) % n
            if j in used or j == i or (i - j) % n in (1, n - 1):
                ok = False
                break
            used.add(i)
            used.add(j)
            edges.append(tuple(sorted((i, j))))
        if ok and len(edges) == n // 2:
            candidates.append(edges)
    if candidates:
        return candidates[0]

    # Fallback fixed seeded search.
    rng = np.random.default_rng(1000 + n)
    for _ in range(10000):
        p = list(map(int, rng.permutation(n)))
        edges = []
        good = True
        for k in range(0, n, 2):
            a, b = p[k], p[k + 1]
            if (a - b) % n in (1, n - 1):
                good = False
                break
            edges.append(tuple(sorted((a, b))))
        if good:
            return sorted(edges)
    raise RuntimeError(f"matching failure for n={n}")


def graph_for(n: int):
    xyz = coords_ring(n)
    bridges = matching(n)
    graph = nx.MultiGraph()
    for i, p in enumerate(xyz):
        graph.add_node(i, pos=p)
    for i in range(n):
        j = (i + 1) % n
        graph.add_edge(i, j, pts=np.vstack([xyz[i], xyz[j]]))
    for a, b in bridges:
        graph.add_edge(a, b, pts=np.vstack([xyz[a], xyz[b]]))
    return xyz, bridges, graph


def run_knottedgraph(repeats: int):
    A = sp.Symbol("A")
    rows = []
    for n in (6, 8, 10, 12):
        _, _, graph = graph_for(n)
        times = []
        result = None
        crossings = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            answer = compute_yamada_polynomial(
                graph,
                A,
                rotation_angles=(0.0, 0.0, 0.0),
                normalize=True,
                n_jobs=1,
                method="negami",
                return_result=True,
            )
            times.append(time.perf_counter() - t0)
            result = str(sp.expand(answer.polynomial))
            crossings = answer.projection.num_crossings
        rows.append(
            dict(
                package="knottedgraph",
                V=n,
                E=n + n // 2,
                crossings=crossings,
                runtime_s=statistics.median(times),
                polynomial=result,
            )
        )
    return rows


def run_topoly(repeats: int):
    from topoly import yamada
    from topoly.params import Closure, ReduceMethod, Translate

    rows = []
    for n in (6, 8, 10, 12):
        xyz, bridges, _ = graph_for(n)
        times = []
        result = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = yamada(
                xyz.tolist(),
                closure=Closure.CLOSED,
                tries=1,
                reduce_method=ReduceMethod.NO,
                max_cross=200,
                poly_reduce=True,
                translate=Translate.NO,
                hide_trivial=False,
                hide_rare=False,
                minimal=False,
                cuda=False,
                run_parallel=False,
                parallel_workers=1,
                bridges=bridges,
                breaks=[],
            )
            times.append(time.perf_counter() - t0)
        rows.append(
            dict(
                package="topoly",
                V=n,
                E=n + n // 2,
                runtime_s=statistics.median(times),
                polynomial=str(result),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", choices=("knottedgraph", "topoly"), required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    rows = run_knottedgraph(args.repeats) if args.package == "knottedgraph" else run_topoly(args.repeats)
    print(json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
