"""Probe locality-preserving RII reduction with queue R1 and first-success skein.

Once a state is RII-free, inverting one crossing can create a new RII bigon only
if that bigon contains the changed crossing.  Its possible partner crossings are
therefore among the physical-arc neighbors of the changed crossing.  This probe
uses the exact existing RII predicate/removal semantics but searches that local
neighborhood instead of all crossing pairs.  Entry-time global RII scans are
omitted; omission of an optimization cannot affect the exact skein polynomial.
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
import benchmark_r1_queue_candidate as q
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_structural import _IsomorphicMemo
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.skein_hybrid import (
    _best_resolution,
    _skein_delta,
    invert_crossing,
    resolve_crossing,
)

BULK = 4


def rii_pair(prepared, first: int, second: int):
    """Exact per-pair predicate factored from PreparedCompactStateBuilder."""
    if first == second or first < 0 or second < 0:
        return None
    first_ports = prepared.ordered_ports[first]
    second_ports = prepared.ordered_ports[second]
    second_position = {port: i for i, port in enumerate(second_ports)}
    shared = []
    for first_position, first_port in enumerate(first_ports):
        partner = prepared.arc_partner[first_port]
        if partner in second_position:
            shared.append(
                (first_position, second_position[partner], first_port, partner)
            )
    if len(shared) != 2:
        return None

    first_positions = [entry[0] for entry in shared]
    second_positions = [entry[1] for entry in shared]
    if (first_positions[0] - first_positions[1]) % 4 not in (1, 3):
        return None
    if (second_positions[0] - second_positions[1]) % 4 not in (1, 3):
        return None
    if any((a % 2) != (b % 2) for a, b, _, _ in shared):
        return None

    removed = set(first_ports) | set(second_ports)
    splices = []
    for first_position, second_position_index, _, _ in shared:
        first_external = first_ports[(first_position + 2) % 4]
        second_external = second_ports[(second_position_index + 2) % 4]
        remote_first = prepared.arc_partner[first_external]
        remote_second = prepared.arc_partner[second_external]
        if (
            remote_first in removed
            or remote_second in removed
            or remote_first == remote_second
        ):
            return None
        splices.append((remote_first, remote_second))
    if len({port for pair in splices for port in pair}) != 4:
        return None
    return tuple(splices)


def local_rii_after_inversion(inverted, crossing_index: int):
    """Remove one newly created RII pair containing ``crossing_index``."""
    neighbors = set()
    for port in inverted.ordered_ports[crossing_index]:
        remote = inverted.arc_partner[port]
        neighbor = inverted.crossing_for_port[remote]
        if neighbor >= 0 and neighbor != crossing_index:
            neighbors.add(neighbor)

    pair_checks = 0
    for neighbor in sorted(neighbors):
        pair_checks += 1
        splices = rii_pair(inverted, crossing_index, neighbor)
        if splices is not None:
            reduced = inverted._remove_reidemeister_ii_pair(  # noqa: SLF001
                crossing_index, neighbor, splices
            )
            return reduced, pair_checks
    return None, pair_checks


def first_local_inversion(prepared):
    crossing_scans = 0
    pair_checks = 0
    for crossing_index in range(len(prepared.crossing_ids)):
        crossing_scans += 1
        inverted = invert_crossing(prepared, crossing_index)
        reduced, checks = local_rii_after_inversion(inverted, crossing_index)
        pair_checks += checks
        if reduced is not None:
            return (1, crossing_index, reduced), crossing_scans, pair_checks
    return None, crossing_scans, pair_checks


def normalize_r1_only(prepared):
    reduced, exponent, moves = q.reduce_r1_queue(prepared)
    return reduced, exponent, moves, int(bool(moves))


def candidate(prepared, evaluator):
    memo = _IsomorphicMemo()
    stats = dict(
        calls=0,
        r1_moves=0,
        r1_rebuilds=0,
        memo_hits=0,
        inversions=0,
        inversion_crossing_scans=0,
        local_rii_pair_checks=0,
        resolutions=0,
        bulk=0,
        max_bulk=0,
    )

    def rec(state):
        stats["calls"] += 1
        state, exponent, moves, rebuilds = normalize_r1_only(state)
        stats["r1_moves"] += moves
        stats["r1_rebuilds"] += rebuilds

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
            inversion, crossing_scans, pair_checks = first_local_inversion(state)
            stats["inversion_crossing_scans"] += crossing_scans
            stats["local_rii_pair_checks"] += pair_checks
            if inversion is not None:
                _, crossing, inverted = inversion
                stats["inversions"] += 1
                positive = rec(resolve_crossing(state, crossing, 0))
                negative = rec(resolve_crossing(state, crossing, 1))
                value = add(rec(inverted), _skein_delta(positive, negative))
            else:
                # Falling back to the existing resolution lookahead is exact.
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


def timed_local(prepared, expected):
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    start = time.perf_counter()
    result, stats = candidate(prepared, evaluator)
    elapsed = time.perf_counter() - start
    assert result == expected
    return elapsed, stats


def run(n: int, mirror: bool, compare_first_success: bool):
    prepared = q.prepare(n, mirror=mirror)
    q.verify_queue_matches_sequential(prepared)
    expected = bounded.expected_terms(n, mirror)

    first_s = None
    if compare_first_success:
        evaluator = NativeCompactYamadaEvaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        start = time.perf_counter()
        result, _ = bounded.candidate(prepared, evaluator, success_limit=1)
        first_s = time.perf_counter() - start
        assert result == expected

    local_s, stats = timed_local(prepared, expected)
    row = {
        "n": n,
        "mirror": mirror,
        "first_success_global_rii_s": first_s,
        "local_rii_s": local_s,
        "speedup_over_first_success": first_s / local_s if first_s is not None else None,
        "stats": stats,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="19,31,51,81")
    parser.add_argument("--mirror-n", type=int, default=19)
    parser.add_argument("--high-n", default="121,161,241,321")
    args = parser.parse_args()

    for n in [int(x) for x in args.n_values.split(",") if x.strip()]:
        run(n, False, True)
    run(args.mirror_n, True, True)
    for n in [int(x) for x in args.high_n.split(",") if x.strip()]:
        run(n, False, False)


if __name__ == "__main__":
    main()
