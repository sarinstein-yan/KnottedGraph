import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada import (
    compute_graph_yamada_polynomial,
    laurent_y_to_sigma_polynomial,
)
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.polynomial import compute_yamada_from_states


def _assert_expr_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _bouquet(loop_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("v")
    for _ in range(loop_count):
        graph.add_edge("v", "v")
    return graph


def _cycle(length: int) -> nx.MultiGraph:
    if length == 1:
        graph = nx.MultiGraph()
        graph.add_node(0)
        graph.add_edge(0, 0)
        return graph
    if length == 2:
        graph = nx.MultiGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1)
        graph.add_edge(0, 1)
        return graph
    return nx.MultiGraph(nx.cycle_graph(length))


def _theta(edge_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_nodes_from(["u", "v"])
    for _ in range(edge_count):
        graph.add_edge("u", "v")
    return graph


def _tree(edge_count: int) -> nx.MultiGraph:
    return nx.MultiGraph(nx.path_graph(edge_count + 1))


def _wedge(left: nx.MultiGraph, right: nx.MultiGraph) -> nx.MultiGraph:
    left = nx.convert_node_labels_to_integers(left, first_label=0)
    right = nx.convert_node_labels_to_integers(
        right, first_label=left.number_of_nodes() - 1
    )
    return nx.MultiGraph(nx.compose(left, right))


def test_published_crossing_free_closed_forms():
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    _assert_expr_equal(compute_graph_yamada_polynomial(nx.MultiGraph(), A), 1)
    isolated = nx.MultiGraph()
    isolated.add_node("v")
    _assert_expr_equal(compute_graph_yamada_polynomial(isolated, A), -1)
    for edge_count in range(1, 7):
        _assert_expr_equal(compute_graph_yamada_polynomial(_tree(edge_count), A), 0)
    for length in range(1, 8):
        _assert_expr_equal(compute_graph_yamada_polynomial(_cycle(length), A), sigma)
    for loop_count in range(1, 7):
        _assert_expr_equal(
            compute_graph_yamada_polynomial(_bouquet(loop_count), A),
            (-1) ** (loop_count - 1) * sigma**loop_count,
        )
    for edge_count in range(1, 9):
        _assert_expr_equal(
            compute_graph_yamada_polynomial(_theta(edge_count), A),
            (sigma + (-sigma) ** edge_count) / (sigma + 1),
        )


def test_isthmus_one_point_union_and_published_k4():
    A = sp.Symbol("A")
    composite = nx.disjoint_union(_cycle(3), _cycle(4))
    composite.add_edge(0, 3)
    _assert_expr_equal(compute_graph_yamada_polynomial(composite, A), 0)

    left = _theta(3)
    right = _bouquet(2)
    _assert_expr_equal(
        compute_graph_yamada_polynomial(_wedge(left, right), A),
        -compute_graph_yamada_polynomial(left, A)
        * compute_graph_yamada_polynomial(right, A),
    )

    k4 = nx.MultiGraph(nx.complete_graph(4))
    _assert_expr_equal(
        compute_graph_yamada_polynomial(k4, A),
        A**3 + 2 * A + 2 * A**-1 + A**-3,
    )


def test_dispatched_backend_matches_arbitrary_precision_python_compact():
    A = sp.Symbol("A")
    python_exact = PythonCompactYamadaEvaluator()
    for graph in (
        _bouquet(2),
        _cycle(5),
        _theta(3),
        _theta(5),
        nx.MultiGraph(nx.complete_graph(4)),
    ):
        _assert_expr_equal(
            compute_graph_yamada_polynomial(graph, A),
            python_exact.compute(graph, A),
        )


def test_current_projection_methods_agree_for_crossing_free_states():
    A = sp.Symbol("A")
    for graph in (
        _bouquet(1),
        _bouquet(2),
        _cycle(3),
        _theta(2),
        _theta(3),
        nx.MultiGraph(nx.complete_graph(4)),
    ):
        negami = compute_yamada_from_states(
            [graph], [0], A, normalize=False, n_jobs=1, method="negami"
        )
        direct = compute_yamada_from_states(
            [graph], [0], A, normalize=False, n_jobs=1, method="recursive"
        )
        _assert_expr_equal(negami, direct)


def test_theta_results_are_canonical_laurent_and_convert_to_sigma():
    A = sp.Symbol("A")
    sigma_symbol = sp.Symbol("sigma")
    for edge_count in range(2, 9):
        result = compute_graph_yamada_polynomial(_theta(edge_count), A)
        shifted = sp.cancel(result * A ** (edge_count - 1))
        numerator, denominator = sp.fraction(shifted)
        assert A not in denominator.free_symbols
        sp.Poly(sp.expand(numerator / denominator), A)
        converted = laurent_y_to_sigma_polynomial(result, A, sigma_symbol).as_expr()
        expected = sp.cancel(
            (sigma_symbol + (-sigma_symbol) ** edge_count) / (sigma_symbol + 1)
        )
        _assert_expr_equal(converted, expected)
