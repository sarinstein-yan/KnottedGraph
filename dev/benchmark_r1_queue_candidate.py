"""Probe a queue-based exact Reidemeister-I closure.

Unlike the batch-visible prototype, this candidate follows newly exposed curl
chains without rebuilding the immutable prepared diagram after every move.  A
min-heap preserves the exact lowest-index removal order of find_reidemeister_i,
while only crossings adjacent to a changed physical arc are rechecked.
"""
from __future__ import annotations

import argparse
import heapq
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
    _resolution_tables,
    _skein_delta,
    diagram_key,
    reduce_reidemeister_i_chain,
    resolve_crossing,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder

BULK = 4


def _r1_move(prepared, crossing_index, partner, active):
    if not active[crossing_index]:
        return None
    ports = prepared.ordered_ports[crossing_index]
    position = {port: i for i, port in enumerate(ports)}
    self_pairs = []
    for i, port in enumerate(ports):
        j = position.get(partner[port])
        if j is not None and i < j:
            self_pairs.append((i, j))
    if len(self_pairs) != 1:
        return None
    pattern = self_pairs[0]
    if pattern in ((0, 1), (2, 3)):
        exponent = -2
        external_positions = (2, 3) if pattern == (0, 1) else (0, 1)
    elif pattern in ((0, 3), (1, 2)):
        exponent = 2
        external_positions = (1, 2) if pattern == (0, 3) else (0, 3)
    else:
        return None

    left = ports[external_positions[0]]
    right = ports[external_positions[1]]
    remote_left = partner[left]
    remote_right = partner[right]
    port_set = set(ports)
    if remote_left in port_set or remote_right in port_set or remote_left == remote_right:
        raise RuntimeError("certified R1 curl has malformed external partners")
    return exponent, remote_left, remote_right


def reduce_r1_queue(prepared):
    """Match sequential R1 reduction exactly, but rebuild the diagram once."""
    crossing_count = len(prepared.crossing_ids)
    if not crossing_count:
        return prepared, 0, 0

    partner = list(prepared.arc_partner)
    active = [True] * crossing_count
    heap = []
    queued = set()

    def maybe_push(crossing):
        if crossing < 0 or crossing >= crossing_count or not active[crossing]:
            return
        if crossing in queued:
            return
        if _r1_move(prepared, crossing, partner, active) is not None:
            heapq.heappush(heap, crossing)
            queued.add(crossing)

    for crossing in range(crossing_count):
        maybe_push(crossing)

    exponent = 0
    moves = 0
    while heap:
        crossing = heapq.heappop(heap)
        queued.discard(crossing)
        move = _r1_move(prepared, crossing, partner, active)
        if move is None:
            continue
        delta, remote_left, remote_right = move
        active[crossing] = False
        exponent += delta
        moves += 1

        # This is exactly the non-self smoothing performed by _smooth_crossing
        # after the curl's self-paired ports disappear.
        partner[remote_left] = remote_right
        partner[remote_right] = remote_left
        for port in prepared.ordered_ports[crossing]:
            partner[port] = -1

        maybe_push(prepared.crossing_for_port[remote_left])
        maybe_push(prepared.crossing_for_port[remote_right])

    if not moves:
        return prepared, 0, 0

    removed_crossings = {i for i, keep in enumerate(active) if not keep}
    removed_ports = {
        port for crossing in removed_crossings for port in prepared.ordered_ports[crossing]
    }
    active_ports = [
        port for port in range(len(partner)) if port not in removed_ports
    ]
    old_to_new = {old: new for new, old in enumerate(active_ports)}
    new_partner = []
    for old in active_ports:
        other = partner[old]
        if other not in old_to_new:
            raise RuntimeError("queue R1 closure left an edge on a removed port")
        new_partner.append(old_to_new[other])

    surviving = [i for i in range(crossing_count) if active[i]]
    crossing_remap = {old: new for new, old in enumerate(surviving)}
    new_crossing_for = []
    for old in active_ports:
        crossing = prepared.crossing_for_port[old]
        if crossing < 0:
            new_crossing_for.append(-1)
        elif crossing in removed_crossings:
            raise RuntimeError("removed crossing port survived queue R1 closure")
        else:
            new_crossing_for.append(crossing_remap[crossing])

    new_ordered = tuple(
        tuple(old_to_new[port] for port in prepared.ordered_ports[i])
        for i in surviving
    )
    plus, minus = _resolution_tables(new_ordered, len(new_partner))
    reduced = PreparedCompactStateBuilder(
        vertex_ids=prepared.vertex_ids,
        crossing_ids=tuple(prepared.crossing_ids[i] for i in surviving),
        ordered_ports=new_ordered,
        arc_partner=tuple(new_partner),
        fixed_terminal_index=tuple(
            prepared.fixed_terminal_index[old] for old in active_ports
        ),
        crossing_for_port=tuple(new_crossing_for),
        plus_partner=plus,
        minus_partner=minus,
    )
    return reduced, exponent, moves


def normalize_queue(prepared):
    current = prepared
    exponent = 0
    r1_moves = 0
    r1_rebuilds = 0
    rii_moves = 0
    while True:
        current, rii = current.reduce_reidemeister_ii()
        rii_moves += rii
        reduced, delta, r1 = reduce_r1_queue(current)
        if not r1:
            return current, exponent, r1_moves, r1_rebuilds, rii_moves
        current = reduced
        exponent += delta
        r1_moves += r1
        r1_rebuilds += 1


def candidate(prepared, evaluator):
    memo = _IsomorphicMemo()
    stats = dict(calls=0, r1_moves=0, r1_rebuilds=0, rii_moves=0,
                 memo_hits=0, inversions=0, resolutions=0, bulk=0, max_bulk=0)

    def rec(q):
        stats["calls"] += 1
        q, exponent, r1, rebuilds, rii = normalize_queue(q)
        stats["r1_moves"] += r1
        stats["r1_rebuilds"] += rebuilds
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


def prepare(n, mirror=False):
    graph, processor, _ = dv.prepare_theta_family(n)
    if mirror:
        for _node, data in graph.nodes(data=True):
            data["pos"] = data["pos"].copy()
            data["pos"][2] *= -1.0
        for _u, _v, _key, data in graph.edges(keys=True, data=True):
            data["pts"] = data["pts"].copy()
            data["pts"][:, 2] *= -1.0
        from knotted_graph.projection import PDCode
        processor = PDCode(graph)
        processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        assert len(processor.crossings) == n
    y = Yamada.from_PDCode(processor)
    return PreparedCompactStateBuilder.prepare(
        y.vertices, y.crossings, y.arcs, _ordered_crossing_ports
    )


def verify_queue_matches_sequential(prepared):
    queue_reduced, queue_shift, queue_moves = reduce_r1_queue(prepared)
    sequential_reduced, sequential_shift, sequential_moves = reduce_reidemeister_i_chain(prepared)
    assert queue_shift == sequential_shift
    assert queue_moves == sequential_moves
    assert diagram_key(queue_reduced) == diagram_key(sequential_reduced)


def run(n, mirror=False, compare_baseline=True):
    prepared = prepare(n, mirror=mirror)
    verify_queue_matches_sequential(prepared)
    published = dv.published_theta_terms(n)
    expected = tuple(sorted(
        ((-power, coeff) for power, coeff in published.items()) if mirror
        else published.items()
    ))

    baseline_s = None
    if compare_baseline:
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
    if compare_baseline:
        assert result == baseline
    print(json.dumps({
        "n": n,
        "mirror": mirror,
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        "speedup": baseline_s / candidate_s if baseline_s is not None else None,
        "stats": stats,
        "correctness": "PASS",
    }, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="19,31,51,81")
    parser.add_argument("--mirror-n", type=int, default=19)
    parser.add_argument("--candidate-only", default="121,161")
    args = parser.parse_args()
    for n in [int(x) for x in args.n_values.split(",") if x.strip()]:
        run(n)
    run(args.mirror_n, mirror=True)
    for n in [int(x) for x in args.candidate_only.split(",") if x.strip()]:
        run(n, compare_baseline=False)


if __name__ == "__main__":
    main()
