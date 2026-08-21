from __future__ import annotations

import statistics
import time

from benchmark_yamada_frontier_ordering_probe import random_crossing_diagram
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_frontier import (
    compute_diagram_frontier_laurent,
    plan_diagram_frontier,
)
from knotted_graph.invariants.yamada.frontier_ordering import plan_frontier_to_target
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator


def median_run(fn, repeats=3):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), values, answer


def benchmark(crossings, offset):
    seed = 220000 + 1000 * crossings + offset
    prepared = random_crossing_diagram(seed, crossings)
    cheap = plan_diagram_frontier(prepared)
    improved = plan_frontier_to_target(prepared, 10)
    print(
        f"CASE crossings={crossings} offset={offset} seed={seed} "
        f"CHEAP_PEAK={cheap['peak_ports']} IMPROVED_PEAK={improved['peak_ports']} "
        f"CHEAP_BOUNDARY={cheap['max_boundary_ports']} "
        f"IMPROVED_BOUNDARY={improved['max_boundary_ports']}"
    )

    native_time, native_times, native_value = median_run(
        lambda: NativeCompactEvaluator(PythonCompactYamadaEvaluator).compute_prepared_bulk_laurent(prepared),
        repeats=2,
    )
    stats = {}
    frontier_time, frontier_times, frontier_value = median_run(
        lambda: compute_diagram_frontier_laurent(
            prepared,
            factor_order=improved["factor_order"],
            stats=stats,
        ),
        repeats=3,
    )
    assert frontier_value == native_value
    print(
        f"NATIVE_EXHAUSTIVE={native_time:.9f}s FRONTIER={frontier_time:.9f}s "
        f"SPEEDUP={native_time/frontier_time:.6f}x EXACT=PASS "
        f"MAX_STATES={stats['max_states']} TRANSITIONS={stats['transitions']}"
    )
    print(f"  native_times={native_times}")
    print(f"  frontier_times={frontier_times}")


def main():
    # These are deterministic generic diagrams found by the ordering probe where
    # the cheap plan is rejected (>10) but multi-start reaches the production target.
    for case in ((10, 32), (12, 68), (16, 41)):
        benchmark(*case)


if __name__ == "__main__":
    main()
