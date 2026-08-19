"""Benchmark a generic exact global crossing-recursion candidate.

The Dobrynin--Vesnin formula is used only after evaluation as a correctness
oracle.  It is never called by ``full_recursive_laurent``.
"""

from __future__ import annotations

import json
import time

import benchmark_topoly_essential_torus_scaling as torus
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import (
    _skein_delta,
    invert_crossing,
    resolve_crossing,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def diagram_key(prepared):
    return (
        len(prepared.vertex_ids),
        prepared.ordered_ports,
        prepared.arc_partner,
        prepared.fixed_terminal_index,
        prepared.crossing_for_port,
    )


def first_reducing_inversion(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if moves:
            return crossing_index, reduced
    return None


def first_resolvable_crossing(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        try:
            children = tuple(
                resolve_crossing(prepared, crossing_index, spin)
                for spin in (0, 1, 2)
            )
        except ValueError:
            continue
        return crossing_index, children
    return None


def full_recursive_laurent(prepared, evaluator, stats=None):
    """Exact Yamada state recursion with one global partial-diagram memo.

    This is a generic algorithm.  It uses only exact RII simplification, the
    Yamada crossing-inversion identity when that exposes RII, and otherwise the
    defining three-state expansion R(D)=A R(D+) + A^-1 R(D-) + R(Dv).
    """
    memo = {}
    if stats is None:
        stats = {}
    stats.update(calls=0, memo_hits=0, rii_moves=0, inversions=0, resolutions=0)

    def rec(current):
        stats["calls"] += 1
        current, moves = current.reduce_reidemeister_ii()
        stats["rii_moves"] += moves
        key = diagram_key(current)
        cached = memo.get(key)
        if cached is not None:
            stats["memo_hits"] += 1
            return cached

        crossing_count = len(current.crossing_ids)
        if crossing_count == 0:
            value = evaluator.compute_prepared_bulk_laurent(current)
            memo[key] = value
            return value

        inversion = first_reducing_inversion(current)
        if inversion is not None:
            crossing_index, inverted_reduced = inversion
            stats["inversions"] += 1
            positive = rec(resolve_crossing(current, crossing_index, 0))
            negative = rec(resolve_crossing(current, crossing_index, 1))
            value = add(rec(inverted_reduced), _skein_delta(positive, negative))
            memo[key] = value
            return value

        resolved = first_resolvable_crossing(current)
        if resolved is None:
            # Rare self-adjacent/degenerate configurations keep the exact old
            # evaluator as a correctness-preserving escape hatch.
            value = evaluator.compute_prepared_bulk_laurent(current)
            memo[key] = value
            return value

        _crossing_index, (plus, minus, vertex) = resolved
        stats["resolutions"] += 1
        value = add(add(shift(rec(plus), 1), shift(rec(minus), -1)), rec(vertex))
        memo[key] = value
        return value

    value = rec(prepared)
    stats["memo_size"] = len(memo)
    return value


def prepared_theta(n):
    _graph, processor, _pdcode = torus.prepare_essential_torus(n)
    yamada = Yamada.from_PDCode(processor)
    return PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )


def main():
    for n in (9, 11, 13, 15, 17):
        prepared = prepared_theta(n)
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        expected = tuple(sorted(torus.independent_theta_terms(n).items()))
        stats = {}
        started = time.perf_counter()
        actual = full_recursive_laurent(prepared, evaluator, stats=stats)
        elapsed = time.perf_counter() - started
        if actual != expected:
            raise AssertionError(
                f"generic full recursion disagrees with external theorem oracle at n={n}"
            )
        print(
            json.dumps(
                {
                    "candidate": "global_exact_crossing_recursion",
                    "n": n,
                    "seconds": elapsed,
                    "stats": stats,
                    "correctness": "PASS",
                },
                separators=(",", ":"),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
