from __future__ import annotations

import itertools
import json
import statistics
import time

import sympy as sp

from benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    _evaluate_fast_state,
    _make_fast_evaluator,
    _sum_laurent_states,
)
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def timed(fn, repeats=3):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def compact_serial(calculator: Yamada, method="negami"):
    evaluator = _make_fast_evaluator(method)
    evaluated = (
        _evaluate_fast_state(evaluator, graph, exponent)
        for graph, exponent in calculator._iter_compact_states()
    )
    return _sum_laurent_states(evaluated, A, True)


def public(calculator: Yamada, n_jobs):
    return calculator.compute(A, normalize=True, n_jobs=n_jobs, method="negami")


def main():
    for crossings in range(2, 8):
        graph = multi_crossing_theta(crossings)
        processor = PDCode(graph)
        processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        calculator = Yamada.from_PDCode(processor)

        compact_time, compact_value = timed(lambda: compact_serial(calculator))
        default_time, default_value = timed(lambda: public(calculator, -1), repeats=1)
        one_time, one_value = timed(lambda: public(calculator, 1))

        if sp.expand(compact_value - default_value) != 0:
            raise AssertionError("default output differs from compact serial")
        if sp.expand(compact_value - one_value) != 0:
            raise AssertionError("n_jobs=1 output differs from compact serial")

        print(json.dumps({
            "crossings": crossings,
            "states": 3**crossings,
            "compact_serial_s": compact_time,
            "public_default_s": default_time,
            "public_n_jobs_1_s": one_time,
            "default_over_compact": default_time / compact_time,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
