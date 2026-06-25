import math
import sys
import types
from pathlib import Path

import networkx as nx
import sympy as sp


try:
    from knotted_graph.projection.geom import Arc, Crossing
    from knotted_graph.invariants.yamada.polynomial import Yamada, compute_negami
except ModuleNotFoundError as exc:
    if exc.name != "poly2graph":
        raise
    src = Path(__file__).resolve().parents[1] / "src"
    kg_pkg = types.ModuleType("knotted_graph")
    kg_pkg.__path__ = [str(src / "knotted_graph")]
    sys.modules["knotted_graph"] = kg_pkg
    projection_pkg = types.ModuleType("knotted_graph.projection")
    projection_pkg.__path__ = [str(src / "knotted_graph" / "projection")]
    sys.modules["knotted_graph.projection"] = projection_pkg
    invariants_pkg = types.ModuleType("knotted_graph.invariants")
    invariants_pkg.__path__ = [str(src / "knotted_graph" / "invariants")]
    sys.modules["knotted_graph.invariants"] = invariants_pkg
    yamada_pkg = types.ModuleType("knotted_graph.invariants.yamada")
    yamada_pkg.__path__ = [str(src / "knotted_graph" / "invariants" / "yamada")]
    sys.modules["knotted_graph.invariants.yamada"] = yamada_pkg

    from knotted_graph.projection.geom import Arc, Crossing
    from knotted_graph.invariants.yamada.polynomial import Yamada, compute_negami


class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Line:
    def __init__(self, coords):
        self.coords = coords


def _crossing(crossing_id, x, y, incident_arcs):
    return Crossing(id=crossing_id, point=_Point(x, y), incident_arcs=incident_arcs)


def _arc(coords, start_id=0, end_id=1):
    return Arc(
        edge_key=("u", "v", Arc._id_counter),
        line=_Line(coords),
        start_type="x",
        start_id=start_id,
        end_type="x",
        end_id=end_id,
    )


def test_negami_uses_inverse_removed_edge_exponent():
    x, y = sp.symbols("x y")
    graph = nx.MultiGraph()
    graph.add_edge("a", "b")

    assert sp.simplify(compute_negami(graph, x, y)) == 0


def test_resolved_crossings_are_not_reintroduced_by_later_resolutions():
    Arc.reset_counter()
    arcs = [
        _arc([(0, 0, 2), (1, 0, 2), (9, 0, 2), (10, 0, 2)]),
        _arc([(0, 0, 0), (0, 1, 0), (10, 1, 0), (10, 0, 0)]),
        _arc([(0, 0, 2), (-1, 0, 2), (11, 0, 2), (10, 0, 2)]),
        _arc([(0, 0, 0), (0, -1, 0), (10, -1, 0), (10, 0, 0)]),
    ]
    crossing0 = _crossing(
        0,
        0,
        0,
        [(arcs[0].id, 0), (arcs[1].id, math.pi / 2), (arcs[2].id, math.pi), (arcs[3].id, -math.pi / 2)],
    )
    crossing1 = _crossing(
        1,
        10,
        0,
        [(arcs[0].id, math.pi), (arcs[1].id, math.pi / 2), (arcs[2].id, 0), (arcs[3].id, -math.pi / 2)],
    )

    state_graphs, _ = Yamada(vertices=[], crossings=[crossing0, crossing1], arcs=arcs)._build_state_graphs()
    fully_resolved_graph = state_graphs[0]

    assert not any(node[0] == "x" for node in fully_resolved_graph.nodes if isinstance(node, tuple))


def test_self_crossing_duplicate_arc_ids_are_resolved_by_ports():
    Arc.reset_counter()
    arcs = [
        _arc([(0, 0, 2), (1, 0, 2), (-1, 0, 2), (0, 0, 2)], start_id=0, end_id=0),
        _arc([(0, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)], start_id=0, end_id=0),
    ]
    crossing = _crossing(
        0,
        0,
        0,
        [(arcs[0].id, 0), (arcs[1].id, math.pi / 2), (arcs[0].id, math.pi), (arcs[1].id, -math.pi / 2)],
    )

    state_graphs, _ = Yamada(vertices=[], crossings=[crossing], arcs=arcs)._build_state_graphs()

    assert state_graphs[0].number_of_edges() > 0
