import networkx as nx
import pytest
import sympy as sp

from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial


def test_public_abstract_yamada_functions_accept_simple_graphs():
    graph = nx.cycle_graph(4)
    variable = sp.Symbol("Y")

    simple_result = compute_graph_yamada_polynomial(graph, variable)
    multigraph_result = compute_graph_yamada_polynomial(
        nx.MultiGraph(graph),
        variable,
    )

    assert sp.simplify(simple_result - multigraph_result) == 0


def test_public_abstract_yamada_functions_reject_directed_graphs():
    with pytest.raises(TypeError, match="must be undirected"):
        compute_graph_yamada_polynomial(nx.DiGraph([(0, 1)]), sp.Symbol("Y"))


def test_public_abstract_yamada_functions_reject_non_graphs():
    with pytest.raises(TypeError, match="networkx.Graph or networkx.MultiGraph"):
        compute_graph_yamada_polynomial([(0, 1)], sp.Symbol("Y"))
