"""Benchmark exact generic crossing recursion with native isomorphism memoization.

The Dobrynin--Vesnin formula is used only after evaluation as an external
correctness oracle. It is never used by the candidate evaluator.
"""

from __future__ import annotations

import json
import time

import benchmark_topoly_essential_torus_scaling as torus
from knotted_graph.invariants.yamada import _yamada_iso
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


def direct_reducing_inversion(prepared):
    """Find an exact RII pair exposed by flipping one crossing in O(c) adjacency work.

    The ordinary RII certificate requires the two shared arcs to preserve
    over/under parity at both crossings. Flipping one crossing toggles the parity
    of all four of its ports, so a geometrically valid bigon whose two shared
    arcs both *reverse* parity is exactly one crossing inversion away from the
    same conservative RII certificate. No trial inversions are needed.
    """
    arc_partner = prepared.arc_partner
    crossing_for_port = prepared.crossing_for_port
    ordered_ports = prepared.ordered_ports

    for first, first_ports in enumerate(ordered_ports):
        by_second = {}
        for first_position, first_port in enumerate(first_ports):
            partner = arc_partner[first_port]
            second = crossing_for_port[partner]
            if second < 0 or second >= first:
                continue
            by_second.setdefault(second, []).append((first_position, partner))

        for second, links in by_second.items():
            if len(links) != 2:
                continue
            second_ports = ordered_ports[second]
            second_position = {port: index for index, port in enumerate(second_ports)}
            shared = []
            for first_position, partner in links:
                position = second_position.get(partner)
                if position is None:
                    break
                shared.append((first_position, position))
            if len(shared) != 2:
                continue
            if (shared[0][0] - shared[1][0]) % 4 not in (1, 3):
                continue
            if (shared[0][1] - shared[1][1]) % 4 not in (1, 3):
                continue
            # Both shared arcs must reverse parity. A one-quarter rotation used
            # by invert_crossing toggles the first crossing parity on both arcs.
            if any((left % 2) == (right % 2) for left, right in shared):
                continue

            removed = set(first_ports) | set(second_ports)
            splices = []
            valid = True
            for first_position, second_position_index in shared:
                first_external = first_ports[(first_position + 2) % 4]
                second_external = second_ports[(second_position_index + 2) % 4]
                remote_first = arc_partner[first_external]
                remote_second = arc_partner[second_external]
                if (
                    remote_first in removed
                    or remote_second in removed
                    or remote_first == remote_second
                ):
                    valid = False
                    break
                splices.append((remote_first, remote_second))
            if not valid or len({port for pair in splices for port in pair}) != 4:
                continue

            inverted = invert_crossing(prepared, first)
            # _remove_reidemeister_ii_pair is the same exact splice operation
            # used by the production conservative RII reducer. Rotation does not
            # change the physical port opposite any shared port, so the splices
            # computed above remain the correct ones after inversion.
            reduced = inverted._remove_reidemeister_ii_pair(
                first,
                second,
                tuple(splices),
            )
            return first, reduced
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


def _native_index(prepared):
    return _yamada_iso.PreparedDiagramIndex(
        len(prepared.vertex_ids),
        [list(ports) for ports in prepared.ordered_ports],
        list(prepared.arc_partner),
        list(prepared.fixed_terminal_index),
        list(prepared.crossing_for_port),
    )


class ExactNativeIsoMemo:
    """Native exact-isomorphism memo; fingerprints are bucket filters only."""

    def __init__(self):
        self.buckets = {}
        self.size = 0
        self.hits = 0
        self.comparisons = 0
        self.index_seconds = 0.0
        self.iso_seconds = 0.0

    def get(self, prepared):
        started = time.perf_counter()
        index = _native_index(prepared)
        self.index_seconds += time.perf_counter() - started
        bucket_key = (
            len(prepared.crossing_ids),
            index.node_count,
            index.fingerprint,
        )
        for other, value in self.buckets.get(bucket_key, ()):
            self.comparisons += 1
            started = time.perf_counter()
            equivalent = index.isomorphic(other)
            self.iso_seconds += time.perf_counter() - started
            if equivalent:
                self.hits += 1
                return True, value, bucket_key, index
        return False, None, bucket_key, index

    def put(self, bucket_key, index, value):
        self.buckets.setdefault(bucket_key, []).append((index, value))
        self.size += 1


def full_recursive_laurent(prepared, evaluator, stats=None):
    """Exact generic Yamada recursion with global exact diagram-isomorphism memo."""
    memo = ExactNativeIsoMemo()
    if stats is None:
        stats = {}
    stats.update(calls=0, memo_hits=0, rii_moves=0, inversions=0, resolutions=0)

    def rec(current):
        stats["calls"] += 1
        current, moves = current.reduce_reidemeister_ii()
        stats["rii_moves"] += moves
        hit, cached, bucket_key, index = memo.get(current)
        if hit:
            stats["memo_hits"] += 1
            return cached

        if not current.crossing_ids:
            value = evaluator.compute_prepared_bulk_laurent(current)
        else:
            inversion = direct_reducing_inversion(current)
            if inversion is not None:
                crossing_index, inverted_reduced = inversion
                stats["inversions"] += 1
                positive = rec(resolve_crossing(current, crossing_index, 0))
                negative = rec(resolve_crossing(current, crossing_index, 1))
                value = add(rec(inverted_reduced), _skein_delta(positive, negative))
            else:
                resolved = first_resolvable_crossing(current)
                if resolved is None:
                    value = evaluator.compute_prepared_bulk_laurent(current)
                else:
                    _index, (plus, minus, vertex) = resolved
                    stats["resolutions"] += 1
                    value = add(
                        add(shift(rec(plus), 1), shift(rec(minus), -1)),
                        rec(vertex),
                    )
        memo.put(bucket_key, index, value)
        return value

    value = rec(prepared)
    stats.update(
        memo_size=memo.size,
        iso_hits=memo.hits,
        iso_comparisons=memo.comparisons,
        index_seconds=memo.index_seconds,
        iso_seconds=memo.iso_seconds,
        buckets=len(memo.buckets),
    )
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
    for n in (9, 11, 13, 15, 17, 19):
        prepared = prepared_theta(n)
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        expected = tuple(sorted(torus.independent_theta_terms(n).items()))
        stats = {}
        started = time.perf_counter()
        actual = full_recursive_laurent(prepared, evaluator, stats=stats)
        elapsed = time.perf_counter() - started
        if actual != expected:
            raise AssertionError(
                f"native-isomorphism generic recursion disagrees with external theorem oracle at n={n}"
            )
        print(json.dumps({
            "candidate": "direct_RII_native_iso_global_recursion",
            "n": n,
            "seconds": elapsed,
            "stats": stats,
            "correctness": "PASS",
        }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
