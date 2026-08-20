"""Probe whether deterministic prepared-state renumbering makes iso memo unnecessary.

A labeled-key hit is stronger than graph isomorphism and therefore always safe.
This benchmark compares exact tuple-key memoization with the native isomorphism
memo after queue R1, first-success inversion and local RII reduction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "dev") not in sys.path:
    sys.path.insert(0, str(ROOT / "dev"))

import benchmark_bounded_inversion_candidate as bounded
import benchmark_local_rii_candidate as local
import benchmark_r1_queue_candidate as q
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.skein_hybrid import (
    _best_resolution,
    _skein_delta,
    diagram_key,
    resolve_crossing,
)

BULK = 4


def candidate(prepared, evaluator):
    memo = {}
    stats = dict(
        calls=0,
        memo_hits=0,
        memo_entries=0,
        r1_moves=0,
        r1_rebuilds=0,
        inversions=0,
        inversion_crossing_scans=0,
        local_rii_pair_checks=0,
        resolutions=0,
        bulk=0,
        max_bulk=0,
    )

    def rec(state):
        stats["calls"] += 1
        state, exponent, moves, rebuilds = local.normalize_r1_only(state)
        stats["r1_moves"] += moves
        stats["r1_rebuilds"] += rebuilds

        key = diagram_key(state)
        if key in memo:
            stats["memo_hits"] += 1
            return shift(memo[key], exponent)

        crossing_count = len(state.crossing_ids)
        if crossing_count <= BULK:
            stats["bulk"] += 1
            stats["max_bulk"] = max(stats["max_bulk"], crossing_count)
            value = evaluator.compute_prepared_bulk_laurent(state)
        else:
            inversion, scans, checks = local.first_local_inversion(state)
            stats["inversion_crossing_scans"] += scans
            stats["local_rii_pair_checks"] += checks
            if inversion is not None:
                _, crossing, inverted = inversion
                stats["inversions"] += 1
                positive = rec(resolve_crossing(state, crossing, 0))
                negative = rec(resolve_crossing(state, crossing, 1))
                value = add(rec(inverted), _skein_delta(positive, negative))
            else:
                resolution = _best_resolution(state)
                if resolution is not None:
                    _, _, children = resolution
                    stats["resolutions"] += 1
                    value = add(
                        add(shift(rec(children[0]), 1), shift(rec(children[1]), -1)),
                        rec(children[2]),
                    )
                else:
                    stats["bulk"] += 1
                    stats["max_bulk"] = max(stats["max_bulk"], crossing_count)
                    value = evaluator.compute_prepared_bulk_laurent(state)
        memo[key] = value
        stats["memo_entries"] = len(memo)
        return shift(value, exponent)

    return rec(prepared), stats


def timed_labeled(prepared, expected):
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    start = time.perf_counter()
    result, stats = candidate(prepared, evaluator)
    elapsed = time.perf_counter() - start
    assert result == expected
    return elapsed, stats


def run(n: int, mirror: bool, compare_iso: bool):
    prepared = q.prepare(n, mirror=mirror)
    expected = bounded.expected_terms(n, mirror)

    iso_s = None
    iso_stats = None
    if compare_iso:
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        start = time.perf_counter()
        iso_result, iso_stats = local.candidate(prepared, evaluator)
        iso_s = time.perf_counter() - start
        assert iso_result == expected

    labeled_s, labeled_stats = timed_labeled(prepared, expected)
    row = {
        "n": n,
        "mirror": mirror,
        "iso_s": iso_s,
        "labeled_s": labeled_s,
        "speedup_over_iso": iso_s / labeled_s if iso_s is not None else None,
        "iso_memo_hits": iso_stats["memo_hits"] if iso_stats else None,
        "labeled_memo_hits": labeled_stats["memo_hits"],
        "labeled_stats": labeled_stats,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="19,31,51,81")
    parser.add_argument("--mirror-n", type=int, default=19)
    parser.add_argument("--high-n", default="121,161")
    args = parser.parse_args()
    for n in [int(x) for x in args.n_values.split(",") if x.strip()]:
        run(n, False, True)
    run(args.mirror_n, True, True)
    for n in [int(x) for x in args.high_n.split(",") if x.strip()]:
        run(n, False, False)


if __name__ == "__main__":
    main()
