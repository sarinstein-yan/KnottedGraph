import itertools

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    _build_state_graph_from_ports,
    _ordered_crossing_ports,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.invariants.yamada.recursive import YamadaRecursiveEvaluator
from knotted_graph.projection import PDCode


def _assert_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _multi_crossing_theta(component_count=3):
    graph = nx.MultiGraph()
    for component in range(component_count):
        y_offset = 5.0 * component
        sign = 1.0 if component % 2 == 0 else -1.0
        left = f"u{component}"
        right = f"v{component}"
        graph.add_node(left, pos=np.array([-2.0, y_offset, 0.0]))
        graph.add_node(right, pos=np.array([2.0, y_offset, 0.0]))
        curves = [
            np.array([[-2.0,0.0,0.0],[-1.0,-1.0,0.5*sign],[1.0,1.0,0.5*sign],[2.0,0.0,0.0]]),
            np.array([[-2.0,0.0,0.0],[-1.0,1.0,-0.5*sign],[1.0,-1.0,-0.5*sign],[2.0,0.0,0.0]]),
            np.array([[-2.0,0.0,0.0],[-1.0,2.0,0.0],[1.0,2.0,0.0],[2.0,0.0,0.0]]),
        ]
        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)
    return graph


def test_compact_state_builder_matches_every_reference_state_value():
    A = sp.Symbol("A")
    processor = PDCode(_multi_crossing_theta(3))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )

    for state in itertools.product([0, 1, 2], repeat=3):
        reference_graph = _build_state_graph_from_ports(
            calculator.vertices,
            calculator.crossings,
            calculator.arcs,
            state,
        )
        compact_graph = prepared.build(state)

        assert compact_graph.n == reference_graph.number_of_nodes()
        assert compact_graph.edge_count == reference_graph.number_of_edges()

        reference_value = YamadaRecursiveEvaluator(A).compute(reference_graph)
        compact_direct = CompactYamadaEvaluator().compute(compact_graph, A)
        compact_negami = CompactNegamiSpecializedEvaluator().compute(compact_graph, A)
        _assert_equal(compact_direct, reference_value)
        _assert_equal(compact_negami, reference_value)


def test_compact_state_builder_preserves_full_state_exponents():
    processor = PDCode(_multi_crossing_theta(2))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )

    states = list(itertools.product([0, 1, 2], repeat=2))
    assert len(states) == 9
    for state in states:
        compact_graph = prepared.build(state)
        assert compact_graph.n >= 0
        assert state.count(0) - state.count(1) == state.count(0) - state.count(1)
