from __future__ import annotations

import sympy as sp

from dev.benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    _evaluate_fast_state,
    _make_fast_evaluator,
    _sum_laurent_states,
)
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


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
    graph = multi_crossing_theta(4)
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
    # A single member of the same family has one projection crossing and is a
    # connected spatial graph. This guards the conservative crossing-terminal
    # union used by the factorizer.
    graph = multi_crossing_theta(1)
    processor = PDCode(graph)
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)

    assert len(calculator._diagram_blocks()) == 1
    for method in ("negami", "recursive"):
        expected = _global_state_sum(calculator, method, False)
        actual = calculator.compute(A, normalize=False, n_jobs=1, method=method)
        _assert_equal(actual, expected)
