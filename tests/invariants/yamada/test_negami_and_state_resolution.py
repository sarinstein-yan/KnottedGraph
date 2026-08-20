import math

import sympy as sp

from knotted_graph.invariants.yamada.compact import CompactYamadaEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection.geom import Arc, Crossing


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


def _prepared(calculator):
    return PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )


def test_resolved_crossings_build_valid_compact_states():
    Arc.reset_counter()
    arcs = [
        _arc([(0, 0, 2), (1, 0, 2), (9, 0, 2), (10, 0, 2)]),
        _arc([(0, 0, 0), (0, 1, 0), (10, 1, 0), (10, 0, 0)]),
        _arc([(0, 0, 2), (-1, 0, 2), (11, 0, 2), (10, 0, 2)]),
        _arc([(0, 0, 0), (0, -1, 0), (10, -1, 0), (10, 0, 0)]),
    ]
    crossing0 = _crossing(0, 0, 0, [(arcs[0].id,0),(arcs[1].id,math.pi/2),(arcs[2].id,math.pi),(arcs[3].id,-math.pi/2)])
    crossing1 = _crossing(1, 10, 0, [(arcs[0].id,math.pi),(arcs[1].id,math.pi/2),(arcs[2].id,0),(arcs[3].id,-math.pi/2)])
    calculator = Yamada(vertices=[], crossings=[crossing0, crossing1], arcs=arcs)
    prepared = _prepared(calculator)
    state = prepared.build((0, 0))
    assert state.n >= 0
    CompactYamadaEvaluator().compute(state, sp.Symbol("A"))


def test_self_crossing_duplicate_arc_ids_are_resolved_by_compact_ports():
    Arc.reset_counter()
    arcs = [
        _arc([(0,0,2),(1,0,2),(-1,0,2),(0,0,2)], start_id=0, end_id=0),
        _arc([(0,0,0),(0,1,0),(0,-1,0),(0,0,0)], start_id=0, end_id=0),
    ]
    crossing = _crossing(0, 0, 0, [(arcs[0].id,0),(arcs[1].id,math.pi/2),(arcs[0].id,math.pi),(arcs[1].id,-math.pi/2)])
    calculator = Yamada(vertices=[], crossings=[crossing], arcs=arcs)
    prepared = _prepared(calculator)
    states = [prepared.build((resolution,)) for resolution in (0, 1, 2)]
    assert all(state.edge_count > 0 for state in states)
