import matplotlib
import sympy as sp

from knotted_graph.applications.mathematical import (
    GRAPH_FAMILY_CATALOG,
    build_graph_case,
    graph_family_names,
    graph_summary,
    plot_structured_multigraph,
)
from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial


matplotlib.use("Agg")


def test_catalog_sample_graphs_build_with_layout_positions():
    assert "theta" in graph_family_names()

    for family_name, spec in GRAPH_FAMILY_CATALOG.items():
        graph, pos = build_graph_case(family_name, *spec.sample_args)

        assert graph.number_of_nodes() > 0
        assert set(graph.nodes()) == set(pos)
        assert all(len(coords) == 2 for coords in pos.values())


def test_structured_multigraph_plotter_handles_loops_and_parallel_edges():
    graph, pos = build_graph_case("bouquet", 4)

    ax = plot_structured_multigraph(
        graph,
        pos,
        family_name="bouquet",
        family_args=(4,),
        show=False,
    )

    assert ax.get_aspect() == 1.0
    assert graph_summary(graph) == {
        "nodes": 1,
        "edges": 4,
        "loops": 4,
        "max_multiplicity": 4,
    }


def test_representative_catalog_yamada_values_are_computable():
    Y = sp.Symbol("Y")

    for family_name, args in [
        ("bouquet", (3,)),
        ("theta", (4,)),
        ("complete_graph", (4,)),
        ("cylinder", (2, 3)),
    ]:
        graph, _pos = build_graph_case(family_name, *args)
        value = compute_graph_yamada_polynomial(graph, Y)

        assert value != 0
