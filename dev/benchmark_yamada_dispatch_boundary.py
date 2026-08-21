from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from benchmark_topoly_random_cubic_ensemble import (  # noqa: E402
    DEFAULT_SEED,
    load_committed_ensemble,
    prepare_sample,
)
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator  # noqa: E402
from knotted_graph.invariants.yamada.diagram_locality import compute_locality_laurent  # noqa: E402
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator  # noqa: E402
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports  # noqa: E402
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder  # noqa: E402


def _prepared_v20_sample0():
    sample, abstract = load_committed_ensemble(20, 1, base_seed=DEFAULT_SEED)[0]
    _embedded, processor, _pdcode, _attempt = prepare_sample(
        sample,
        abstract,
        DEFAULT_SEED,
    )
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )
    prepared, _ = prepared.reduce_reidemeister_ii()
    return prepared


def _median(fn, repeats=5):
    values = []
    answer = None
    for _ in range(repeats):
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        start = time.perf_counter()
        answer = fn(evaluator)
        values.append(time.perf_counter() - start)
    return statistics.median(values), values, answer


def main():
    prepared = _prepared_v20_sample0()
    c = len(prepared.crossing_ids)
    if c != 7:
        raise AssertionError(f"frozen V=20 sample 0 expected seven crossings, got {c}")

    bulk_median, bulk_times, bulk = _median(
        lambda evaluator: evaluator.compute_prepared_bulk_laurent(prepared)
    )
    structural_median, structural_times, structural = _median(
        lambda evaluator: compute_locality_laurent(prepared, evaluator)
    )
    if structural != bulk:
        raise AssertionError("structural and exhaustive dispatch candidates disagree")

    speedup = bulk_median / structural_median
    print(f"crossings={c}")
    print(f"bulk_times={bulk_times}")
    print(f"structural_times={structural_times}")
    print(f"bulk_median_s={bulk_median:.9f}")
    print(f"structural_median_s={structural_median:.9f}")
    print(f"bulk_over_structural={speedup:.6f}")


if __name__ == "__main__":
    main()
