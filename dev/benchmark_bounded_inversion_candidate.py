"""Probe exact bounded inversion search on top of queue-based R1 closure.

The Yamada skein identity is exact at every crossing.  Searching all crossings
for the inversion that maximizes subsequent Reidemeister-II cancellation is
therefore only a performance heuristic.  This benchmark compares the current
exhaustive inversion search with deterministic bounded-success variants.
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

import benchmark_r1_queue_candidate as q
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_structural import _IsomorphicMemo
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.skein_hybrid import (
    _best_inversion,
    _best_resolution,
    _skein_delta,
    invert_crossing,
    resolve_crossing,
)

BULK = 4


def bounded_inversion(prepared, success_limit: int):
    """Return best among the first ``success_limit`` RII-producing inversions.

    Every returned candidate is mathematically exact; the limit changes only
    the crossing-selection heuristic.  ``success_limit=1`` is first-success.
    """
    if success_limit < 1:
        raise ValueError("success_limit must be >= 1")
    best = None
    scans = 0
    successes = 0
    for crossing_index in range(len(prepared.crossing_ids)):
        scans += 1
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if not moves:
            continue
        successes += 1
        if best is None or moves > best[0]:
            best = moves, crossing_index, reduced
        if successes >= success_limit:
            break
    return best, scans, successes


def candidate(prepared, evaluator, *, success_limit: int | None):
    memo = _IsomorphicMemo()
    stats = dict(
        calls=0,
        r1_moves=0,
        r1_rebuilds=0,
        rii_moves=0,
        memo_hits=0,
        inversions=0,
        inversion_scans=0,
        inversion_successes_examined=0,
        resolutions=0,
        bulk=0,
        max_bulk=0,
    )

    def rec(state):
        stats["calls"] += 1
        state, exponent, r1, rebuilds, rii = q.normalize_queue(state)
        stats["r1_moves"] += r1
        stats["r1_rebuilds"] += rebuilds
        stats["rii_moves"] += rii

        hit, value, key, index = memo.get(state)
        if hit:
            stats["memo_hits"] += 1
            return shift(value, exponent)

        crossing_count = len(state.crossing_ids)
        if crossing_count <= BULK:
            stats["bulk"] += 1
            stats["max_bulk"] = max(stats["max_bulk"], crossing_count)
            value = evaluator.compute_prepared_bulk_laurent(state)
        else:
            if success_limit is None:
                inversion = _best_inversion(state)
                # Exhaustive search has inspected every crossing.
                stats["inversion_scans"] += crossing_count
                stats["inversion_successes_examined"] += crossing_count
            else:
                inversion, scans, successes = bounded_inversion(state, success_limit)
                stats["inversion_scans"] += scans
                stats["inversion_successes_examined"] += successes

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
        memo.put(key, index, state, value)
        return shift(value, exponent)

    return rec(prepared), stats


def expected_terms(n: int, mirror: bool):
    published = q.dv.published_theta_terms(n)
    if mirror:
        return tuple(sorted((-power, coeff) for power, coeff in published.items()))
    return tuple(sorted(published.items()))


def timed_variant(prepared, expected, success_limit):
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    start = time.perf_counter()
    result, stats = candidate(prepared, evaluator, success_limit=success_limit)
    elapsed = time.perf_counter() - start
    assert result == expected
    return elapsed, stats


def run(n: int, mirror: bool, variants: tuple[int | None, ...], production: bool):
    prepared = q.prepare(n, mirror=mirror)
    q.verify_queue_matches_sequential(prepared)
    expected = expected_terms(n, mirror)

    production_s = None
    if production:
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        start = time.perf_counter()
        result = evaluator.compute_prepared_laurent(prepared)
        production_s = time.perf_counter() - start
        assert result == expected

    rows = []
    for limit in variants:
        elapsed, stats = timed_variant(prepared, expected, limit)
        row = {
            "n": n,
            "mirror": mirror,
            "variant": "full" if limit is None else f"first-{limit}-success",
            "production_s": production_s,
            "candidate_s": elapsed,
            "production_over_candidate": (
                production_s / elapsed if production_s is not None else None
            ),
            "stats": stats,
            "correctness": "PASS",
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="19,31,51,81")
    parser.add_argument("--mirror-n", type=int, default=19)
    parser.add_argument("--high-n", default="121,161")
    args = parser.parse_args()

    variants = (None, 4, 1)
    for n in [int(x) for x in args.n_values.split(",") if x.strip()]:
        run(n, False, variants, production=True)
    run(args.mirror_n, True, variants, production=True)

    # At larger n, compare only bounded variants; the exhaustive scan is already
    # known to dominate and would needlessly consume CI time.
    for n in [int(x) for x in args.high_n.split(",") if x.strip()]:
        run(n, False, (4, 1), production=False)


if __name__ == "__main__":
    main()
