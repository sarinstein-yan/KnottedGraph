import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.fast import (
    FastNegamiSpecializedEvaluator,
    FastYamadaEvaluator,
    normalize_yamada,
    to_sympy,
)
from knotted_graph.invariants.yamada.polynomial import compute_yamada_from_states
from knotted_graph.invariants.yamada.recursive import (
    NegamiRecursiveEvaluator,
    YamadaRecursiveEvaluator,
)


def _equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _graphs():
    out = []

    for n in range(1, 7):
        g = nx.MultiGraph()
        g.add_node(0)
        for _ in range(n):
            g.add_edge(0, 0)
        out.append(g)

    for n in range(3, 9):
        out.append(nx.MultiGraph(nx.cycle_graph(n)))

    for n in range(2, 9):
        g = nx.MultiGraph()
        g.add_nodes_from([0, 1])
        for _ in range(n):
            g.add_edge(0, 1)
        out.append(g)

    out.extend(
        [
            nx.MultiGraph(nx.complete_graph(4)),
            nx.MultiGraph(nx.wheel_graph(6)),
            nx.MultiGraph(nx.circular_ladder_graph(4)),
            nx.MultiGraph(nx.complete_bipartite_graph(3, 3)),
        ]
    )

    # Deterministic random bridgeless examples.
    for seed in range(5):
        g = nx.random_regular_graph(3, 8, seed=seed)
        if nx.is_connected(g) and not list(nx.bridges(g)):
            out.append(nx.MultiGraph(g))

    return out


def test_fast_direct_kernel_matches_reference_sympy_evaluator():
    A = sp.Symbol("A")

    for graph in _graphs():
        reference = YamadaRecursiveEvaluator(A).compute(graph)
        fast = FastYamadaEvaluator().compute(graph, A)
        _equal(fast, reference)


def test_fast_specialized_negami_matches_reference_specialization():
    A = sp.Symbol("A")
    x, y = sp.symbols("x y")

    for graph in _graphs():
        reference_h = NegamiRecursiveEvaluator(x, y).compute(graph)
        reference = sp.expand(
            reference_h.xreplace({x: -1, y: -A - 2 - A**-1})
        )
        fast = FastNegamiSpecializedEvaluator().compute(graph, A)
        _equal(fast, reference)


def test_fast_normalization_matches_existing_public_convention():
    A = sp.Symbol("A")

    for graph in _graphs():
        raw = FastYamadaEvaluator().compute_laurent(graph)
        normalized_fast = to_sympy(normalize_yamada(raw), A)

        reference_raw = YamadaRecursiveEvaluator(A).compute(graph)
        terms = sp.expand(sp.cancel(reference_raw)).as_ordered_terms()
        lowest = min(term.as_coeff_exponent(A)[1] for term in terms)
        reference = sp.expand(
            sp.cancel(reference_raw * (-A) ** (-lowest))
        )
        _equal(normalized_fast, reference)


def test_public_state_api_fast_backends_remain_equivalent():
    A = sp.Symbol("A")

    for graph in _graphs():
        direct = compute_yamada_from_states(
            [graph], [0], A, normalize=False, n_jobs=1, method="recursive"
        )
        negami = compute_yamada_from_states(
            [graph], [0], A, normalize=False, n_jobs=1, method="negami"
        )
        _equal(direct, negami)
