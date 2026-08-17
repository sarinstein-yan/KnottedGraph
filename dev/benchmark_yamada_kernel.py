from __future__ import annotations

import json
import statistics
import time

import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.fast import (
    FastNegamiSpecializedEvaluator,
    FastYamadaEvaluator,
)
from knotted_graph.invariants.yamada.recursive import (
    NegamiRecursiveEvaluator,
    YamadaRecursiveEvaluator,
)

A = sp.Symbol("A")
X, Y = sp.symbols("x y")


def equal(left, right):
    return sp.simplify(sp.together(sp.expand(left - right))) == 0


def cases():
    """Small/medium CI suite; the paper notebook handles timeout-frontier sweeps."""
    out = []
    for n in range(5, 9):
        out.append((f"wheel_{n}", nx.MultiGraph(nx.wheel_graph(n))))
    for n in range(3, 6):
        out.append((f"ladder_{n}", nx.MultiGraph(nx.circular_ladder_graph(n))))
    for n in (6, 8):
        g = nx.random_regular_graph(3, n, seed=1000 + 10 * n)
        if nx.is_connected(g) and not list(nx.bridges(g)):
            out.append((f"random3_{n}", nx.MultiGraph(g)))
    out.append(("K33", nx.MultiGraph(nx.complete_bipartite_graph(3, 3))))
    out.append(("K4", nx.MultiGraph(nx.complete_graph(4))))
    return out


def timed(fn, repeats=2):
    vals = []
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        vals.append(time.perf_counter() - t0)
    return statistics.median(vals), value


def main():
    rows = []
    for name, graph in cases():
        edge_count = graph.number_of_edges()
        vertex_count = graph.number_of_nodes()

        old_t, old = timed(lambda: YamadaRecursiveEvaluator(A).compute(graph))
        fast_t, fast = timed(lambda: FastYamadaEvaluator().compute(graph, A))
        compact_t, compact = timed(lambda: CompactYamadaEvaluator().compute(graph, A))
        if not equal(old, fast) or not equal(old, compact):
            raise AssertionError(f"direct mismatch for {name}")

        oldn_t, oldh = timed(lambda: NegamiRecursiveEvaluator(X, Y).compute(graph))
        oldn = sp.expand(oldh.xreplace({X: -1, Y: -A - 2 - A**-1}))
        fastn_t, fastn = timed(
            lambda: FastNegamiSpecializedEvaluator().compute(graph, A)
        )
        compactn_t, compactn = timed(
            lambda: CompactNegamiSpecializedEvaluator().compute(graph, A)
        )
        if not equal(oldn, fastn) or not equal(oldn, compactn):
            raise AssertionError(f"Negami mismatch for {name}")

        row = dict(
            case=name,
            V=vertex_count,
            E=edge_count,
            old_direct_s=old_t,
            fast_direct_s=fast_t,
            compact_direct_s=compact_t,
            laurent_direct_speedup=old_t / fast_t,
            compact_direct_speedup=old_t / compact_t,
            old_negami_s=oldn_t,
            fast_negami_s=fastn_t,
            compact_negami_s=compactn_t,
            laurent_negami_speedup=oldn_t / fastn_t,
            compact_negami_speedup=oldn_t / compactn_t,
        )
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
