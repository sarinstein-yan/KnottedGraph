import itertools

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
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
            np.array(
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, -1.0, 0.5 * sign],
                    [1.0, 1.0, 0.5 * sign],
                    [2.0, 0.0, 0.0],
                ]
            ),
            np.array(
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 1.0, -0.5 * sign],
                    [1.0, -1.0, -0.5 * sign],
                    [2.0, 0.0, 0.0],
                ]
            ),
            np.array(
                [[-2.0, 0.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]]
            ),
        ]
        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)
    return graph


def _prepared(component_count):
    processor = PDCode(_multi_crossing_theta(component_count))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)
    return PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )


def test_every_compact_state_agrees_across_current_exact_kernels():
    A = sp.Symbol("A")
    prepared = _prepared(3)
    direct = CompactYamadaEvaluator()
    negami = CompactNegamiSpecializedEvaluator()
    python_exact = PythonCompactYamadaEvaluator()
    for state in itertools.product([0, 1, 2], repeat=3):
        compact_graph = prepared.build(state)
        expected = python_exact.compute(compact_graph, A)
        _assert_equal(direct.compute(compact_graph, A), expected)
        _assert_equal(negami.compute(compact_graph, A), expected)


def test_compact_state_builder_covers_full_state_space_and_exponents():
    prepared = _prepared(2)
    states = list(itertools.product([0, 1, 2], repeat=2))
    assert len(states) == 9
    exponents = []
    for state in states:
        compact_graph = prepared.build(state)
        assert compact_graph.n >= 0
        exponents.append(state.count(0) - state.count(1))
    assert sorted(exponents) == [-2, -1, -1, 0, 0, 0, 1, 1, 2]
