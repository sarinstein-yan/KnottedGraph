"""Sweep exact generic crossing-recursion policies and native bulk cutoffs.

The Dobrynin--Vesnin formula is used only after evaluation as an external
correctness oracle. It is never used by a candidate evaluator.
"""
from __future__ import annotations
import json, time
import benchmark_topoly_essential_torus_scaling as torus
from knotted_graph.invariants.yamada import _yamada_iso
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import _skein_delta, invert_crossing, resolve_crossing
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def trial_reducing_inversion(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if moves:
            return crossing_index, reduced
    return None


def direct_reducing_inversion(prepared):
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
                if remote_first in removed or remote_second in removed or remote_first == remote_second:
                    valid = False
                    break
                splices.append((remote_first, remote_second))
            if valid and len({port for pair in splices for port in pair}) == 4:
                inverted = invert_crossing(prepared, first)
                return first, inverted._remove_reidemeister_ii_pair(first, second, tuple(splices))
    return None


def first_resolvable_crossing(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        try:
            return crossing_index, tuple(resolve_crossing(prepared, crossing_index, spin) for spin in (0, 1, 2))
        except ValueError:
            continue
    return None


def native_index(prepared):
    return _yamada_iso.PreparedDiagramIndex(
        len(prepared.vertex_ids), [list(x) for x in prepared.ordered_ports],
        list(prepared.arc_partner), list(prepared.fixed_terminal_index), list(prepared.crossing_for_port)
    )


class ExactNativeIsoMemo:
    def __init__(self):
        self.buckets = {}; self.size = self.hits = self.comparisons = 0
    def get(self, prepared):
        index = native_index(prepared)
        key = (len(prepared.crossing_ids), index.node_count, index.fingerprint)
        for other, value in self.buckets.get(key, ()):
            self.comparisons += 1
            if index.isomorphic(other):
                self.hits += 1
                return True, value, key, index
        return False, None, key, index
    def put(self, key, index, value):
        self.buckets.setdefault(key, []).append((index, value)); self.size += 1


def evaluate(prepared, evaluator, *, cutoff, policy, stats):
    memo = ExactNativeIsoMemo()
    inversion_fn = trial_reducing_inversion if policy == "trial" else direct_reducing_inversion
    stats.update(calls=0, hits=0, bulk=0, inversions=0, resolutions=0)
    def rec(current):
        stats["calls"] += 1
        current, _moves = current.reduce_reidemeister_ii()
        hit, cached, key, index = memo.get(current)
        if hit:
            stats["hits"] += 1
            return cached
        c = len(current.crossing_ids)
        if c <= cutoff:
            stats["bulk"] += 1
            value = evaluator.compute_prepared_bulk_laurent(current)
        else:
            inversion = inversion_fn(current)
            if inversion is not None:
                crossing_index, reduced = inversion
                stats["inversions"] += 1
                plus = rec(resolve_crossing(current, crossing_index, 0))
                minus = rec(resolve_crossing(current, crossing_index, 1))
                value = add(rec(reduced), _skein_delta(plus, minus))
            else:
                resolved = first_resolvable_crossing(current)
                if resolved is None:
                    stats["bulk"] += 1
                    value = evaluator.compute_prepared_bulk_laurent(current)
                else:
                    _crossing_index, (plus, minus, vertex) = resolved
                    stats["resolutions"] += 1
                    value = add(add(shift(rec(plus), 1), shift(rec(minus), -1)), rec(vertex))
        memo.put(key, index, value)
        return value
    result = rec(prepared)
    stats.update(memo_size=memo.size, iso_hits=memo.hits, iso_comparisons=memo.comparisons)
    return result


def prepared_theta(n):
    _graph, processor, _pd = torus.prepare_essential_torus(n)
    y = Yamada.from_PDCode(processor)
    return PreparedCompactStateBuilder.prepare(y.vertices, y.crossings, y.arcs, _ordered_crossing_ports)


def main():
    # Broad sweep at n=17 locates the native-bulk crossover. Then only the best
    # plausible cutoffs are paid for at n=15 and n=19.
    cases = []
    for policy in ("trial", "direct"):
        for cutoff in (0, 2, 3, 4, 5, 6, 7):
            cases.append((17, policy, cutoff))
    for n in (15, 19):
        for policy in ("trial", "direct"):
            for cutoff in (3, 4, 5, 6):
                cases.append((n, policy, cutoff))
    for n, policy, cutoff in cases:
        prepared = prepared_theta(n)
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        expected = tuple(sorted(torus.independent_theta_terms(n).items()))
        stats = {}
        start = time.perf_counter()
        actual = evaluate(prepared, evaluator, cutoff=cutoff, policy=policy, stats=stats)
        elapsed = time.perf_counter() - start
        if actual != expected:
            raise AssertionError((n, policy, cutoff, actual, expected))
        print(json.dumps({"n":n,"policy":policy,"cutoff":cutoff,"seconds":elapsed,"stats":stats,"correctness":"PASS"}, separators=(",", ":")), flush=True)

if __name__ == "__main__":
    main()
