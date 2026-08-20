import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.recursive import (
    NegamiRecursiveEvaluator,
    YamadaRecursiveEvaluator,
)


def _equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _suite():
    graphs = []

    for loops in range(1, 6):
        g = nx.MultiGraph()
        g.add_node(0)
        for _ in range(loops):
            g.add_edge(0, 0)
        graphs.append(g)

    for n in range(1, 8):
        if n == 1:
            g = nx.MultiGraph()
            g.add_node(0)
            g.add_edge(0, 0)
        elif n == 2:
            g = nx.MultiGraph()
            g.add_nodes_from([0, 1])
            g.add_edge(0, 1)
            g.add_edge(0, 1)
        else:
            g = nx.MultiGraph(nx.cycle_graph(n))
        graphs.append(g)

    for edges in range(2, 8):
        g = nx.MultiGraph()
        g.add_nodes_from([0, 1])
        for _ in range(edges):
            g.add_edge(0, 1)
        graphs.append(g)

    graphs.extend(
        [
            nx.MultiGraph(nx.complete_graph(4)),
            nx.MultiGraph(nx.wheel_graph(6)),
            nx.MultiGraph(nx.circular_ladder_graph(4)),
            nx.MultiGraph(nx.complete_bipartite_graph(3, 3)),
            nx.MultiGraph(nx.path_graph(6)),
        ]
    )

    # Articulation / one-point union case.
    left = nx.convert_node_labels_to_integers(nx.complete_graph(4))
    right = nx.convert_node_labels_to_integers(nx.cycle_graph(4), first_label=3)
    graphs.append(nx.MultiGraph(nx.compose(left, right)))

    for n in (6, 8, 10):
        for seed in range(4):
            g = nx.random_regular_graph(3, n, seed=100 * n + seed)
            if nx.is_connected(g):
                graphs.append(nx.MultiGraph(g))

    return graphs


def test_compact_delete_contract_preserve_reference_values():
    A = sp.Symbol("A")

    for graph in _suite():
        reference = YamadaRecursiveEvaluator(A).compute(graph)
        compact = CompactYamadaEvaluator().compute(graph, A)
        _equal(compact, reference)


def test_compact_negami_specialization_matches_reference():
    A = sp.Symbol("A")
    x, y = sp.symbols("x y")

    for graph in _suite():
        reference_h = NegamiRecursiveEvaluator(x, y).compute(graph)
        reference = sp.expand(
            reference_h.xreplace({x: -1, y: -A - 2 - A**-1})
        )
        compact = CompactNegamiSpecializedEvaluator().compute(graph, A)
        _equal(compact, reference)


def test_compact_graph_edge_count_and_contraction_multiedges():
    graph = nx.MultiGraph()
    graph.add_edge("u", "v")
    graph.add_edge("u", "v")
    graph.add_edge("u", "v")

    compact = CompactGraph.from_networkx(graph)
    assert compact.edge_count == 3

    contracted = compact.contract_edge(0, 1)
    assert contracted.n == 1
    assert contracted.edge_count == 2
    assert contracted.rows == ((2,),)


def test_compact_bridge_detection_is_multigraph_safe():
    single = nx.MultiGraph()
    single.add_edge(0, 1)
    assert CompactGraph.from_networkx(single).has_bridge()

    parallel = nx.MultiGraph()
    parallel.add_edge(0, 1)
    parallel.add_edge(0, 1)
    assert not CompactGraph.from_networkx(parallel).has_bridge()
