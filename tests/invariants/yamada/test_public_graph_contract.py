import networkx as nx
import pytest
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import compute_negami
from knotted_graph.invariants.yamada.recursive import (
    compute_negami_recursive,
    compute_yamada_polynomial_recursive,
)


def test_public_abstract_yamada_functions_accept_simple_graphs():
    graph = nx.cycle_graph(4)
    A = sp.Symbol("A")
    x, y = sp.symbols("x y")

    yamada = compute_yamada_polynomial_recursive(graph, A)
    direct_negami = compute_negami(graph, x, y)
    recursive_negami = compute_negami_recursive(graph, x, y)

    assert sp.simplify(yamada - (A + 1 + A**-1)) == 0
    assert sp.simplify(direct_negami - recursive_negami) == 0


@pytest.mark.parametrize(
    "compute",
    [
        lambda graph: compute_yamada_polynomial_recursive(graph, sp.Symbol("A")),
        lambda graph: compute_negami(graph, *sp.symbols("x y")),
        lambda graph: compute_negami_recursive(graph, *sp.symbols("x y")),
    ],
)
def test_public_abstract_yamada_functions_reject_directed_graphs(compute):
    with pytest.raises(TypeError, match="must be undirected"):
        compute(nx.DiGraph([(0, 1)]))


@pytest.mark.parametrize(
    "compute",
    [
        lambda graph: compute_yamada_polynomial_recursive(graph, sp.Symbol("A")),
        lambda graph: compute_negami(graph, *sp.symbols("x y")),
        lambda graph: compute_negami_recursive(graph, *sp.symbols("x y")),
    ],
)
def test_public_abstract_yamada_functions_reject_non_graphs(compute):
    with pytest.raises(TypeError, match="networkx.Graph or networkx.MultiGraph"):
        compute([(0, 1)])
