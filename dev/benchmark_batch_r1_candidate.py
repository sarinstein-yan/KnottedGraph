"""Second-round breakthrough probe: simultaneous R1 + exact inversion pruning.

The candidate applies all currently visible Reidemeister-I curls in one rebuild
and avoids trying crossing inversions that cannot possibly create an RII pair.
Both optimizations are family-agnostic.  The published Dobrynin--Vesnin formula
is used only after timing as an external oracle.
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
    _best_resolution,
    _skein_delta,
    invert_crossing,
    resolve_crossing,
)
from knotted_graph.invariants.yamada.state_compact import (
    PreparedCompactStateBuilder,
    _MINUS_PAIRS,
    _PLUS_PAIRS,
)

BULK = 4


def _resolution_tables(ordered_ports, port_count):
    plus = [-1] * port_count
    minus = [-1] * port_count
    for ports in ordered_ports:
        for a, b in _PLUS_PAIRS:
            pa, pb = ports[a], ports[b]
            plus[pa] = pb
            plus[pb] = pa
        for a, b in _MINUS_PAIRS:
            pa, pb = ports[a], ports[b]
            minus[pa] = pb
            minus[pb] = pa
    return tuple(plus), tuple(minus)


def visible_r1_moves(prepared):
    moves = []
    partner = prepared.arc_partner
    for crossing_index, ports in enumerate(prepared.ordered_ports):
        position = {port: i for i, port in enumerate(ports)}
        self_pairs = []
        for i, port in enumerate(ports):
            j = position.get(partner[port])
            if j is not None and i < j:
                self_pairs.append((i, j))
        if len(self_pairs) != 1:
            continue
        pattern = self_pairs[0]
        if pattern in ((0, 1), (2, 3)):
            moves.append((crossing_index, _PLUS_PAIRS, -2))
        elif pattern in ((0, 3), (1, 2)):
            moves.append((crossing_index, _MINUS_PAIRS, 2))
    return moves


def apply_r1_batch(prepared, moves):
    if not moves:
        return prepared, 0, 0

    removed_crossings = {crossing for crossing, _pairs, _shift in moves}
    removed_ports = {
        port
        for crossing in removed_crossings
        for port in prepared.ordered_ports[crossing]
    }
    adjacency: dict[int, set[int]] = {port: set() for port in removed_ports}
    partner = prepared.arc_partner
    for port in removed_ports:
        other = partner[port]
        adjacency.setdefault(port, set()).add(other)
        adjacency.setdefault(other, set()).add(port)

    total_shift = 0
    for crossing, pairs, exponent in moves:
        ports = prepared.ordered_ports[crossing]
        total_shift += exponent
        for a, b in pairs:
            left, right = ports[a], ports[b]
            adjacency[left].add(right)
            adjacency[right].add(left)

    seen = set()
    splices = []
    closed_loops = 0
    for start in tuple(adjacency):
        if start in seen:
            continue
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        external = sorted(node for node in component if node not in removed_ports)
        if len(external) == 2:
            splices.append((external[0], external[1]))
        elif len(external) == 0:
            closed_loops += 1
        else:
            raise ValueError("R1 batch connectivity is not locally reducible")

    active_ports = [p for p in range(len(partner)) if p not in removed_ports]
    old_to_new = {old: new for new, old in enumerate(active_ports)}
    updated_partner = list(partner)
    for left, right in splices:
        updated_partner[left] = right
        updated_partner[right] = left
    new_partner = []
    for old in active_ports:
        other = updated_partner[old]
        if other not in old_to_new:
            raise ValueError("R1 batch left an arc attached to a removed port")
        new_partner.append(old_to_new[other])

    surviving = [
        i for i in range(len(prepared.crossing_ids)) if i not in removed_crossings
    ]
    crossing_remap = {old: new for new, old in enumerate(surviving)}
    new_crossing_for = []
    for old in active_ports:
        crossing = prepared.crossing_for_port[old]
        if crossing < 0:
            new_crossing_for.append(-1)
        elif crossing in removed_crossings:
            raise ValueError("removed crossing port survived R1 batch")
        else:
            new_crossing_for.append(crossing_remap[crossing])

    new_ordered = tuple(
        tuple(old_to_new[port] for port in prepared.ordered_ports[i])
        for i in surviving
    )
    new_fixed = [prepared.fixed_terminal_index[old] for old in active_ports]
    new_vertex_ids = list(prepared.vertex_ids)
    next_id = max((*prepared.vertex_ids, *prepared.crossing_ids), default=-1) + 1
    for loop_index in range(closed_loops):
        vertex_index = len(new_vertex_ids)
        new_vertex_ids.append(next_id + loop_index)
        left = len(new_partner)
        right = left + 1
        new_partner.extend((right, left))
        new_fixed.extend((vertex_index, vertex_index))
        new_crossing_for.extend((-1, -1))

    plus, minus = _resolution_tables(new_ordered, len(new_partner))
    return PreparedCompactStateBuilder(
        vertex_ids=tuple(new_vertex_ids),
        crossing_ids=tuple(prepared.crossing_ids[i] for i in surviving),
        ordered_ports=new_ordered,
        arc_partner=tuple(new_partner),
        fixed_terminal_index=tuple(new_fixed),
        crossing_for_port=tuple(new_crossing_for),
        plus_partner=plus,
        minus_partner=minus,
    ), total_shift, len(moves)


def normalize_batch(prepared):
    current = prepared
    exponent = 0
    r1_moves = 0
    r1_batches = 0
    rii_moves = 0
    while True:
        current, moves = current.reduce_reidemeister_ii()
        rii_moves += moves
        visible = visible_r1_moves(current)
        if not visible:
            return current, exponent, r1_moves, r1_batches, rii_moves
        try:
            current, delta, count = apply_r1_batch(current, visible)
        except ValueError:
            current, delta, count = apply_r1_batch(current, visible[:1])
        exponent += delta
        r1_moves += count
        r1_batches += 1


def inversion_candidate_indices(prepared):
    """Return exactly the crossings whose inversion can create an immediate RII.

    Inverting one crossing changes no relation between any other pair. Therefore
    any newly available RII bigon must contain that crossing. Before inversion,
    such a pair is joined by exactly two physical arcs at adjacent cyclic ports
    at both crossings with opposite over/under parity at each joined endpoint.
    """
    crossing_count = len(prepared.ordered_ports)
    port_position = [-1] * len(prepared.arc_partner)
    for crossing, ports in enumerate(prepared.ordered_ports):
        for position, port in enumerate(ports):
            port_position[port] = position

    candidates = set()
    for first in range(crossing_count):
        by_second: dict[int, list[tuple[int, int]]] = {}
        for first_position, port in enumerate(prepared.ordered_ports[first]):
            partner = prepared.arc_partner[port]
            second = prepared.crossing_for_port[partner]
            if second < 0 or second == first:
                continue
            by_second.setdefault(second, []).append(
                (first_position, port_position[partner])
            )
        for second, shared in by_second.items():
            if len(shared) != 2:
                continue
            first_positions = (shared[0][0], shared[1][0])
            second_positions = (shared[0][1], shared[1][1])
            if (first_positions[0] - first_positions[1]) % 4 not in (1, 3):
                continue
            if (second_positions[0] - second_positions[1]) % 4 not in (1, 3):
                continue
            if any((a % 2) == (b % 2) for a, b in shared):
                continue
            candidates.add(first)
            candidates.add(second)
    return tuple(sorted(candidates))


def best_inversion_filtered(prepared, stats):
    candidates = inversion_candidate_indices(prepared)
    stats["inversion_scans"] += len(prepared.crossing_ids)
    stats["inversion_candidates"] += len(candidates)
    best = None
    for crossing_index in candidates:
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if moves and (best is None or moves > best[0]):
            best = moves, crossing_index, reduced
    return best


def candidate(prepared, evaluator):
    memo = _IsomorphicMemo()
    stats = dict(calls=0, r1_moves=0, r1_batches=0, rii_moves=0,
                 memo_hits=0, inversions=0, inversion_scans=0,
                 inversion_candidates=0, resolutions=0, bulk=0, max_bulk=0)

    def rec(q):
        stats["calls"] += 1
        q, exponent, r1, batches, rii = normalize_batch(q)
        stats["r1_moves"] += r1
        stats["r1_batches"] += batches
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
            inversion = best_inversion_filtered(q, stats)
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


def run(n, mirror=False, compare_baseline=True):
    prepared = prepare(n, mirror=mirror)
    published = dv.published_theta_terms(n)
    expected = tuple(sorted(
        ((-power, coeff) if mirror else (power, coeff))
        for power, coeff in published.items()
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
