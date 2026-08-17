from __future__ import annotations

import random

import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.recursive import YamadaRecursiveEvaluator

A = sp.Symbol("A")


def _equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _random_multigraph(seed: int) -> nx.MultiGraph:
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n))
    # Cover loops, parallel edges, bridges, disconnected pieces, and denser cores.
    for _ in range(rng.randint(0, 9)):
        u = rng.randrange(n)
        v = rng.randrange(n)
        graph.add_edge(u, v)
    return graph


def test_compact_kernels_match_retained_sympy_reference_on_random_multigraphs():
    reference = YamadaRecursiveEvaluator(A)
    direct = CompactYamadaEvaluator()
    negami = CompactNegamiSpecializedEvaluator()

    for seed in range(120):
        graph = _random_multigraph(10000 + seed)
        compact = CompactGraph.from_networkx(graph)
        expected = reference.compute(graph)
        _equal(direct.compute(compact, A), expected)
        _equal(negami.compute(compact, A), expected)


def test_compact_kernels_agree_after_all_single_edge_mutations():
    direct = CompactYamadaEvaluator()
    negami = CompactNegamiSpecializedEvaluator()

    for seed in range(40):
        graph = _random_multigraph(20000 + seed)
        compact = CompactGraph.from_networkx(graph)
        candidates = [compact]
        loop = compact.first_loop()
        if loop is not None:
            candidates.append(compact.delete_loop(loop))
        edge = compact.first_nonloop()
        if edge is not None:
            candidates.extend((compact.delete_edge(*edge), compact.contract_edge(*edge)))

        for candidate in candidates:
            _equal(direct.compute(candidate, A), negami.compute(candidate, A))
