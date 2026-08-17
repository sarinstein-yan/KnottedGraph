from __future__ import annotations

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    _evaluate_fast_state,
    _make_fast_evaluator,
    _sum_laurent_states,
)
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def _multi_crossing_theta(component_count: int = 3) -> nx.MultiGraph:
    """Self-contained nondegenerate fixture used by the regression suite."""
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
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 2.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
        ]

        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)

    return graph


def _global_state_sum(calculator: Yamada, method: str, normalize: bool):
    evaluator = _make_fast_evaluator(method)
    evaluated = (
        _evaluate_fast_state(evaluator, graph, exponent)
        for graph, exponent in calculator._iter_compact_states()
    )
    return _sum_laurent_states(evaluated, A, normalize)


def _assert_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def test_factorized_state_sum_matches_full_cartesian_state_sum():
    graph = _multi_crossing_theta(4)
    processor = PDCode(graph)
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)

    assert len(calculator._diagram_blocks()) == 4
    for method in ("negami", "recursive"):
        for normalize in (False, True):
            expected = _global_state_sum(calculator, method, normalize)
            actual = calculator.compute(A, normalize=normalize, n_jobs=1, method=method)
            _assert_equal(actual, expected)


def test_connected_diagram_is_not_split_into_independent_blocks():
    graph = _multi_crossing_theta(1)
    processor = PDCode(graph)
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)

    assert len(calculator._diagram_blocks()) == 1
    for method in ("negami", "recursive"):
        expected = _global_state_sum(calculator, method, False)
        actual = calculator.compute(A, normalize=False, n_jobs=1, method=method)
        _assert_equal(actual, expected)
