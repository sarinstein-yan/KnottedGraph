import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import compute_yamada_from_states
from knotted_graph.invariants.yamada.recursive import (
    compute_yamada_polynomial_recursive,
    contract_multigraph_edge,
    delete_multigraph_edge,
    multigraph_key,
    pick_nonloop_edge,
)


def _assert_expr_equal(left, right):
    assert sp.simplify(sp.expand(left - right)) == 0


def _bouquet(loop_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("v")
    for _ in range(loop_count):
        graph.add_edge("v", "v")
    return graph


def _theta(edge_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_nodes_from(["u", "v"])
    for _ in range(edge_count):
        graph.add_edge("u", "v")
    return graph


def test_recursive_base_cases_and_closed_forms():
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1

    _assert_expr_equal(compute_yamada_polynomial_recursive(nx.MultiGraph(), A), 1)

    for loop_count in [1, 2, 3]:
        _assert_expr_equal(
            compute_yamada_polynomial_recursive(_bouquet(loop_count), A),
            -((-sigma) ** loop_count),
        )

    cycle = nx.MultiGraph(nx.cycle_graph(3))
    _assert_expr_equal(compute_yamada_polynomial_recursive(cycle, A), sigma)

    for edge_count in [2, 3, 4]:
        _assert_expr_equal(
            compute_yamada_polynomial_recursive(_theta(edge_count), A),
            (sigma + (-sigma) ** edge_count) / (sigma + 1),
        )


def test_recursive_disjoint_union_and_bridge_zero():
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1

    disconnected = nx.disjoint_union(_bouquet(1), _bouquet(2))
    _assert_expr_equal(
        compute_yamada_polynomial_recursive(disconnected, A),
        sigma * (-(sigma**2)),
    )

    bridge = nx.MultiGraph()
    bridge.add_edge("u", "v")
    _assert_expr_equal(compute_yamada_polynomial_recursive(bridge, A), 0)


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
        nx.MultiGraph(nx.cycle_graph(3)),
        _theta(2),
        _theta(3),
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
