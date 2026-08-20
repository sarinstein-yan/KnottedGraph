from __future__ import annotations

import random

import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
    PythonCompactYamadaEvaluator,
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
    python_fallback = PythonCompactYamadaEvaluator()

    for seed in range(160):
        graph = _random_multigraph(10000 + seed)
        compact = CompactGraph.from_networkx(graph)
        expected = reference.compute(graph)
        _equal(direct.compute(compact, A), expected)
        _equal(negami.compute(compact, A), expected)
        _equal(python_fallback.compute(compact, A), expected)


def test_structural_reductions_match_reference_on_targeted_multigraphs():
    reference = YamadaRecursiveEvaluator(A)
    direct = CompactYamadaEvaluator()
    python_fallback = PythonCompactYamadaEvaluator()

    cases: list[nx.MultiGraph] = []

    # Many loops: exercises exact (-sigma)^k batching.
    bouquet = nx.MultiGraph()
    bouquet.add_node(0)
    for _ in range(7):
        bouquet.add_edge(0, 0)
    cases.append(bouquet)

    # A heavily subdivided bridgeless core: exercises homeomorphism reduction.
    subdivided = nx.MultiGraph(nx.complete_graph(4))
    next_node = 4
    for u, v in list(subdivided.edges()):
        subdivided.remove_edge(u, v)
        previous = u
        for _ in range(3):
            subdivided.add_edge(previous, next_node)
            previous = next_node
            next_node += 1
        subdivided.add_edge(previous, v)
    cases.append(subdivided)

    # Parallel class embedded in a nontrivial core.
    parallel = nx.MultiGraph(nx.complete_graph(4))
    for _ in range(6):
        parallel.add_edge(0, 1)
    cases.append(parallel)

    # Mixed loops, subdivisions and parallel edges.
    mixed = nx.MultiGraph()
    mixed.add_nodes_from(range(6))
    mixed.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 2),
            (0, 5),
            (5, 3),
        ]
    )
    mixed.add_edge(0, 1)
    mixed.add_edge(0, 1)
    mixed.add_edge(4, 4)
    mixed.add_edge(4, 4)
    cases.append(mixed)

    for graph in cases:
        compact = CompactGraph.from_networkx(graph)
        expected = reference.compute(graph)
        _equal(direct.compute(compact, A), expected)
        _equal(python_fallback.compute(compact, A), expected)


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
