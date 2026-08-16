import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import PDCode


def _assert_equal(left, right):
    assert sp.simplify(
        sp.together(
            sp.expand(left - right)
        )
    ) == 0


def _bow_tie_self_loop():
    graph = nx.MultiGraph()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    graph.add_node(0, pos=points[0])
    graph.add_edge(0, 0, pts=points)
    return graph


def _crossing_free_self_loop():
    graph = nx.MultiGraph()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    graph.add_node(0, pos=points[0])
    graph.add_edge(0, 0, pts=points)
    return graph


def test_nugatory_duplicate_arc_crossing_keeps_normalized_yamada():
    A = sp.Symbol("A")

    curled = PDCode(_bow_tie_self_loop())
    curled_pd = curled.compute(rotation_angles=(0.0, 0.0, 0.0))

    assert len(curled.crossings) == 1
    crossing = next(iter(curled.crossings.values()))
    incident_ids = [arc_id for arc_id, _ in crossing.incident_arcs]
    assert len(incident_ids) == 4
    assert len(set(incident_ids)) == 3

    assert "X[" not in curled_pd
    assert crossing.pd_code == ""

    simple = PDCode(_crossing_free_self_loop())
    simple.compute(rotation_angles=(0.0, 0.0, 0.0))

    for method in ("recursive", "negami"):
        curled_value = curled.compute_yamada(
            A,
            normalize=True,
            n_jobs=1,
            method=method,
        )
        simple_value = simple.compute_yamada(
            A,
            normalize=True,
            n_jobs=1,
            method=method,
        )
        _assert_equal(curled_value, simple_value)


def test_duplicate_arc_crossing_backends_agree_before_normalization():
    A = sp.Symbol("A")

    processor = PDCode(_bow_tie_self_loop())
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))

    recursive = processor.compute_yamada(
        A,
        normalize=False,
        n_jobs=1,
        method="recursive",
    )
    negami = processor.compute_yamada(
        A,
        normalize=False,
        n_jobs=1,
        method="negami",
    )

    _assert_equal(recursive, negami)
