"""Second-round exact optimization probe: fixed-point R1/RII normalization.

This benchmark deliberately bypasses every theorem fast path.  The published
Dobrynin--Vesnin formula is expanded only after timing as an independent oracle.
The candidate differs from production only by normalizing a partial prepared
diagram to an R1/RII fixed point before memoization/branch selection, thereby
removing repeated Python recursion frames and repeated isomorphism construction.
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

import benchmark_dobrynin_vesnin_theta_family as dv
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_structural import _IsomorphicMemo
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import (
    _best_inversion,
    _best_resolution,
    _skein_delta,
    find_reidemeister_i,
    resolve_crossing,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder

BULK = 4


def normalize_regular_isotopy(prepared):
    """Return exact R1/RII fixed point, total Laurent shift and move counts."""
    current = prepared
    exponent = 0
    r1_moves = 0
    rii_moves = 0
    while True:
        current, moves = current.reduce_reidemeister_ii()
        rii_moves += moves
        curl = find_reidemeister_i(current)
        if curl is None:
            return current, exponent, r1_moves, rii_moves
        current, delta = curl
        exponent += delta
        r1_moves += 1


def candidate(prepared, evaluator):
    memo = _IsomorphicMemo()
    stats = dict(calls=0, r1_moves=0, rii_moves=0, memo_hits=0,
                 inversions=0, resolutions=0, bulk=0, max_bulk=0)

    def rec(q):
        stats["calls"] += 1
        q, exponent, r1, rii = normalize_regular_isotopy(q)
        stats["r1_moves"] += r1
        stats["rii_moves"] += rii

        hit, value, key, index = memo.get(q)
        if hit:
            stats["memo_hits"] += 1
            return shift(value, exponent)

        crossing_count = len(q.crossing_ids)
        if crossing_count <= BULK:
            stats["bulk"] += 1
            stats["max_bulk"] = max(stats["max_bulk"], crossing_count)
            value = evaluator.compute_prepared_bulk_laurent(q)
        else:
            inversion = _best_inversion(q)
            if inversion is not None:
                _, crossing, inverted = inversion
                stats["inversions"] += 1
                positive = rec(resolve_crossing(q, crossing, 0))
                negative = rec(resolve_crossing(q, crossing, 1))
                value = add(rec(inverted), _skein_delta(positive, negative))
            else:
                resolution = _best_resolution(q)
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
                    value = evaluator.compute_prepared_bulk_laurent(q)
        memo.put(key, index, q, value)
        return shift(value, exponent)

    return rec(prepared), stats


def prepare(n):
    _, processor, _ = dv.prepare_theta_family(n)
    y = Yamada.from_PDCode(processor)
    return PreparedCompactStateBuilder.prepare(
        y.vertices, y.crossings, y.arcs, _ordered_crossing_ports
    )


def run(n):
    prepared = prepare(n)
    expected = tuple(sorted(dv.published_theta_terms(n).items()))

    baseline_eval = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    t0 = time.perf_counter()
    baseline = baseline_eval.compute_prepared_laurent(prepared)
    baseline_s = time.perf_counter() - t0
    assert baseline == expected

    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    t0 = time.perf_counter()
    result, stats = candidate(prepared, evaluator)
    candidate_s = time.perf_counter() - t0
    assert result == expected
    assert result == baseline
    print(json.dumps({
        "n": n,
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        "speedup": baseline_s / candidate_s,
        "stats": stats,
        "correctness": "PASS",
    }, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="19,31,51,81")
    args = parser.parse_args()
    for n in [int(x) for x in args.n_values.split(",") if x.strip()]:
        run(n)


if __name__ == "__main__":
    main()
