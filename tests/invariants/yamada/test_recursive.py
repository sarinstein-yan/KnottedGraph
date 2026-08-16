import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import (
    compute_negami,
    compute_yamada_from_states,
)
from knotted_graph.invariants.yamada.recursive import (
    compute_negami_recursive,
    compute_yamada_polynomial_recursive,
    contract_multigraph_edge,
    delete_multigraph_edge,
    has_isthmus_multigraph,
    is_cycle_multigraph,
    multigraph_key,
    pick_nonloop_edge,
    theta_edge_count,
)


def _assert_expr_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _bouquet(loop_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("v")
    for _ in range(loop_count):
        graph.add_edge("v", "v")
    return graph


def _cycle(length: int) -> nx.MultiGraph:
    if length < 1:
        raise ValueError("length must be >= 1")

    graph = nx.MultiGraph()
    if length == 1:
        graph.add_node(0)
        graph.add_edge(0, 0)
        return graph
    if length == 2:
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


def _k4() -> nx.MultiGraph:
    return nx.MultiGraph(nx.complete_graph(4))


def _wedge(left: nx.MultiGraph, right: nx.MultiGraph) -> nx.MultiGraph:
    """Return a one-point union without relying on graph labels."""
    left = nx.convert_node_labels_to_integers(left, first_label=0)
    right = nx.convert_node_labels_to_integers(
        right,
        first_label=left.number_of_nodes() - 1,
    )
    graph = nx.compose(left, right)
    return nx.MultiGraph(graph)


def test_recursive_literature_closed_forms():
    """Li et al. (2018), Lemma 2.3 and standard Yamada identities."""
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1

    _assert_expr_equal(compute_yamada_polynomial_recursive(nx.MultiGraph(), A), 1)

    isolated = nx.MultiGraph()
    isolated.add_node("v")
    _assert_expr_equal(compute_yamada_polynomial_recursive(isolated, A), -1)

    for edge_count in range(1, 7):
        _assert_expr_equal(
            compute_yamada_polynomial_recursive(_tree(edge_count), A),
            0,
        )

    for length in range(1, 8):
        _assert_expr_equal(
            compute_yamada_polynomial_recursive(_cycle(length), A),
            sigma,
        )

    for loop_count in range(1, 7):
        _assert_expr_equal(
            compute_yamada_polynomial_recursive(_bouquet(loop_count), A),
            (-1) ** (loop_count - 1) * sigma**loop_count,
        )

    for edge_count in range(1, 9):
        _assert_expr_equal(
            compute_yamada_polynomial_recursive(_theta(edge_count), A),
            (sigma + (-sigma) ** edge_count) / (sigma + 1),
        )


def test_bridge_shortcut_is_multigraph_safe():
    single = nx.MultiGraph()
    single.add_edge("u", "v")
    assert has_isthmus_multigraph(single)

    parallel = nx.MultiGraph()
    parallel.add_edge("u", "v")
    parallel.add_edge("u", "v")
    assert not has_isthmus_multigraph(parallel)

    # A bridge remains fatal even when both sides contain nontrivial cycles.
    composite = nx.disjoint_union(_cycle(3), _cycle(4))
    composite.add_edge(0, 3)
    assert has_isthmus_multigraph(composite)

    A = sp.Symbol("A")
    _assert_expr_equal(
        compute_yamada_polynomial_recursive(composite, A),
        0,
    )


def test_cycle_and_theta_recognizers():
    for length in range(1, 8):
        assert is_cycle_multigraph(_cycle(length))

    not_cycle = _cycle(3)
    not_cycle.add_edge(0, 1)
    assert not is_cycle_multigraph(not_cycle)

    for edge_count in range(1, 8):
        assert theta_edge_count(_theta(edge_count)) == edge_count

    with_loop = _theta(3)
    with_loop.add_edge("u", "u")
    assert theta_edge_count(with_loop) is None


def test_one_point_union_factorization():
    """H(G1 . G2) = -H(G1)H(G2)."""
    A = sp.Symbol("A")
    left = _theta(3)
    right = _bouquet(2)
    wedge = _wedge(left, right)

    expected = -(
        compute_yamada_polynomial_recursive(left, A)
        * compute_yamada_polynomial_recursive(right, A)
    )
    _assert_expr_equal(
        compute_yamada_polynomial_recursive(wedge, A),
        expected,
    )


def test_published_planar_k4_value():
    """Dobrynin--Vesnin (1996), Table 1: planar K4 = G^1_4."""
    A = sp.Symbol("A")
    expected = A**3 + 2 * A + 2 * A**-1 + A**-3

    _assert_expr_equal(
        compute_yamada_polynomial_recursive(_k4(), A),
        expected,
    )


def test_recursive_negami_matches_defining_subset_sum():
    """The new recursion must agree with the retained independent definition."""
    x, y = sp.symbols("x y")

    graphs = [
        _bouquet(1),
        _bouquet(2),
        _cycle(3),
        _theta(3),
        _k4(),
        _tree(2),
    ]

    for graph in graphs:
        _assert_expr_equal(
            compute_negami_recursive(graph, x, y),
            compute_negami(graph, x, y),
        )


def test_negami_specialization_matches_direct_yamada_recursion():
    """H(G)=h(G)(-1,-A-2-A^-1), as in Yamada's definition."""
    A = sp.Symbol("A")
    x, y = sp.symbols("x y")

    graphs = [
        _bouquet(2),
        _cycle(5),
        _theta(3),
        _theta(5),
        _k4(),
    ]

    for graph in graphs:
        h_value = compute_negami_recursive(graph, x, y)
        specialized = h_value.xreplace(
            {
                x: sp.Integer(-1),
                y: -A - 2 - A**-1,
            }
        )
        _assert_expr_equal(
            specialized,
            compute_yamada_polynomial_recursive(graph, A),
        )


def test_multigraph_key_is_relabel_invariant_and_preserves_multiplicity():
    left = nx.MultiGraph()
    left.add_edge("a", "b")
    left.add_edge("a", "b")
    left.add_edge("a", "a")

    right = nx.MultiGraph()
    right.add_edge(10, 20)
    right.add_edge(20, 10)
    right.add_edge(20, 20)

    assert multigraph_key(left) == multigraph_key(right)


def test_delete_and_contract_preserve_multigraph_edge_occurrences():
    graph = _theta(3)
    edge = pick_nonloop_edge(graph)

    deleted = delete_multigraph_edge(graph, edge)
    contracted = contract_multigraph_edge(graph, edge)

    assert deleted.number_of_edges() == 2
    assert contracted.number_of_edges() == 2
    assert all(u == v for u, v in contracted.edges())


def test_recursive_backend_matches_negami_backend_for_crossing_free_states():
    A = sp.Symbol("A")
    graphs = [
        _bouquet(1),
        _bouquet(2),
        _cycle(3),
        _theta(2),
        _theta(3),
        _k4(),
    ]

    for graph in graphs:
        negami = compute_yamada_from_states(
            [graph],
            [0],
            A,
            normalize=False,
            n_jobs=1,
            method="negami",
        )
        recursive = compute_yamada_from_states(
            [graph],
            [0],
            A,
            normalize=False,
            n_jobs=1,
            method="recursive",
        )
        _assert_expr_equal(negami, recursive)


def test_yamada_state_evaluator_rejects_unknown_backend():
    A = sp.Symbol("A")
    graph = _bouquet(1)

    try:
        compute_yamada_from_states([graph], [0], A, method="unknown")
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("unknown backend should raise ValueError")
