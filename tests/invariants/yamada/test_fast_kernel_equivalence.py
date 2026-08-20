import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    PythonCompactNegamiSpecializedEvaluator,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.fast import (
    FastNegamiSpecializedEvaluator,
    FastYamadaEvaluator,
    normalize_yamada,
    to_sympy,
)
from knotted_graph.invariants.yamada.polynomial import compute_yamada_from_states


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
    for seed in range(5):
        g = nx.random_regular_graph(3, 8, seed=seed)
        if nx.is_connected(g) and not list(nx.bridges(g)):
            out.append(nx.MultiGraph(g))
    return out


def test_dispatched_direct_kernel_matches_exact_python_compact():
    A = sp.Symbol("A")
    python_exact = PythonCompactYamadaEvaluator()
    for graph in _graphs():
        _equal(FastYamadaEvaluator().compute(graph, A), python_exact.compute(graph, A))


def test_dispatched_specialized_negami_matches_exact_python_compact():
    A = sp.Symbol("A")
    python_exact = PythonCompactNegamiSpecializedEvaluator()
    for graph in _graphs():
        _equal(
            FastNegamiSpecializedEvaluator().compute(graph, A),
            python_exact.compute(graph, A),
        )


def test_fast_normalization_is_exact_and_idempotent_at_polynomial_level():
    A = sp.Symbol("A")
    for graph in _graphs():
        raw = FastYamadaEvaluator().compute_laurent(graph)
        normalized = normalize_yamada(raw)
        expression = to_sympy(normalized, A)
        if expression != 0:
            terms = sp.expand(expression).as_ordered_terms()
            assert min(term.as_coeff_exponent(A)[1] for term in terms) == 0


def test_public_state_methods_remain_equivalent():
    A = sp.Symbol("A")
    for graph in _graphs():
        direct = compute_yamada_from_states(
            [graph], [0], A, normalize=False, n_jobs=1, method="recursive"
        )
        negami = compute_yamada_from_states(
            [graph], [0], A, normalize=False, n_jobs=1, method="negami"
        )
        _equal(direct, negami)
