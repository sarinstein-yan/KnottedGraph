"""Locality-enhanced exact structural Yamada evaluation.

This module preserves the established generic structural evaluator's algebraic
identities while replacing global-search bottlenecks with local closure:

* Reidemeister-II closure uses a mutable physical-arc matching and rebuilds the
  immutable prepared state only once after a maximal chain.
* Inversion lookahead probes a virtual rotation of only four ports.  A complete
  inverted prepared diagram is materialized only for the first successful
  candidate, avoiding one O(c) table rebuild for every failed crossing probe.
* Resolution lookahead checks only pairs among the surviving physical neighbors
  of the resolved crossing, because no remote crossing adjacency changed.

No family-specific formula or recognizer is used here.
"""

from __future__ import annotations

import heapq
import itertools

from .diagram_structural import (
    BULK_LEAF_MAX_CROSSINGS,
    _IsomorphicMemo,
    _reduce_r1_queue,
)
from .fast import add, shift
from .skein_hybrid import (
    _resolution_tables,
    _skein_delta,
    diagram_key,
    invert_crossing,
    resolve_crossing,
)
from .state_compact import PreparedCompactStateBuilder


def _rii_splices_for_orders(prepared, first_ports, second_ports):
    """Return exact RII splice data for two supplied cyclic port orders."""
    second_position = {port: index for index, port in enumerate(second_ports)}
    shared = []
    for first_position, first_port in enumerate(first_ports):
        remote = prepared.arc_partner[first_port]
        if remote in second_position:
            shared.append((first_position, second_position[remote]))
    if len(shared) != 2:
        return None

    first_positions = [entry[0] for entry in shared]
    second_positions = [entry[1] for entry in shared]
    if (first_positions[0] - first_positions[1]) % 4 not in (1, 3):
        return None
    if (second_positions[0] - second_positions[1]) % 4 not in (1, 3):
        return None
    if any((a % 2) != (b % 2) for a, b in shared):
        return None

    removed = set(first_ports) | set(second_ports)
    splices = []
    for first_position, second_position in shared:
        first_external = first_ports[(first_position + 2) % 4]
        second_external = second_ports[(second_position + 2) % 4]
        remote_first = prepared.arc_partner[first_external]
        remote_second = prepared.arc_partner[second_external]
        if remote_first in removed or remote_second in removed:
            return None
        if remote_first == remote_second:
            return None
        splices.append((remote_first, remote_second))
    if len({port for pair in splices for port in pair}) != 4:
        return None
    return tuple(splices)


def _rii_pair_mutable(prepared, first, second, partner, active):
    if first == second or first < 0 or second < 0:
        return None
    if not active[first] or not active[second]:
        return None

    first_ports = prepared.ordered_ports[first]
    second_ports = prepared.ordered_ports[second]
    second_position = {port: index for index, port in enumerate(second_ports)}
    shared = []
    for first_position, first_port in enumerate(first_ports):
        remote = partner[first_port]
        if remote in second_position:
            shared.append((first_position, second_position[remote]))
    if len(shared) != 2:
        return None

    first_positions = [entry[0] for entry in shared]
    second_positions = [entry[1] for entry in shared]
    if (first_positions[0] - first_positions[1]) % 4 not in (1, 3):
        return None
    if (second_positions[0] - second_positions[1]) % 4 not in (1, 3):
        return None
    if any((a % 2) != (b % 2) for a, b in shared):
        return None

    removed = set(first_ports) | set(second_ports)
    splices = []
    for first_position, second_position in shared:
        first_external = first_ports[(first_position + 2) % 4]
        second_external = second_ports[(second_position + 2) % 4]
        remote_first = partner[first_external]
        remote_second = partner[second_external]
        if remote_first in removed or remote_second in removed:
            return None
        if remote_first == remote_second:
            return None
        splices.append((remote_first, remote_second))
    if len({port for pair in splices for port in pair}) != 4:
        return None
    return tuple(splices)


def _rebuild(prepared, partner, active):
    removed_crossings = {index for index, keep in enumerate(active) if not keep}
    removed_ports = {
        port
        for crossing in removed_crossings
        for port in prepared.ordered_ports[crossing]
    }
    active_ports = [port for port in range(len(partner)) if port not in removed_ports]
    old_to_new = {old: new for new, old in enumerate(active_ports)}

    new_partner = []
    for old in active_ports:
        other = partner[old]
        if other not in old_to_new:
            raise RuntimeError("RII closure left an edge on a removed port")
        new_partner.append(old_to_new[other])

    surviving = [index for index, keep in enumerate(active) if keep]
    crossing_remap = {old: new for new, old in enumerate(surviving)}
    crossing_for = []
    for old in active_ports:
        crossing = prepared.crossing_for_port[old]
        if crossing < 0:
            crossing_for.append(-1)
        elif crossing in removed_crossings:
            raise RuntimeError("removed crossing port survived RII closure")
        else:
            crossing_for.append(crossing_remap[crossing])

    ordered = tuple(
        tuple(old_to_new[port] for port in prepared.ordered_ports[index])
        for index in surviving
    )
    plus, minus = _resolution_tables(ordered, len(new_partner))
    return PreparedCompactStateBuilder(
        vertex_ids=prepared.vertex_ids,
        crossing_ids=tuple(prepared.crossing_ids[index] for index in surviving),
        ordered_ports=ordered,
        arc_partner=tuple(new_partner),
        fixed_terminal_index=tuple(prepared.fixed_terminal_index[old] for old in active_ports),
        crossing_for_port=tuple(crossing_for),
        plus_partner=plus,
        minus_partner=minus,
    )


def _adjacent_pairs(prepared):
    for first, ports in enumerate(prepared.ordered_ports):
        neighbors = set()
        for port in ports:
            remote = prepared.arc_partner[port]
            second = prepared.crossing_for_port[remote]
            if 0 <= second < first:
                neighbors.add(second)
        for second in sorted(neighbors):
            yield first, second


def reduce_rii_queue(prepared, *, candidate_pairs=None):
    """Exact maximal RII closure with local rechecks and one final rebuild."""
    crossing_count = len(prepared.crossing_ids)
    if crossing_count < 2:
        return prepared, 0, 0

    partner = list(prepared.arc_partner)
    active = [True] * crossing_count
    heap = []
    queued = set()
    checks = 0

    def maybe_push(first, second):
        nonlocal checks
        if first == second or first < 0 or second < 0:
            return
        if first < second:
            first, second = second, first
        pair = first, second
        if pair in queued or not active[first] or not active[second]:
            return
        checks += 1
        if _rii_pair_mutable(prepared, first, second, partner, active) is not None:
            heapq.heappush(heap, pair)
            queued.add(pair)

    seeds = _adjacent_pairs(prepared) if candidate_pairs is None else candidate_pairs
    for pair in seeds:
        maybe_push(*pair)

    moves = 0
    while heap:
        first, second = heapq.heappop(heap)
        queued.discard((first, second))
        checks += 1
        splices = _rii_pair_mutable(prepared, first, second, partner, active)
        if splices is None:
            continue

        active[first] = False
        active[second] = False
        moves += 1
        for left, right in splices:
            partner[left] = right
            partner[right] = left
        for crossing in (first, second):
            for port in prepared.ordered_ports[crossing]:
                partner[port] = -1

        for left, right in splices:
            maybe_push(
                prepared.crossing_for_port[left],
                prepared.crossing_for_port[right],
            )

    if not moves:
        return prepared, 0, checks
    return _rebuild(prepared, partner, active), moves, checks


def _neighbor_indices(prepared, crossing_index):
    result = set()
    for port in prepared.ordered_ports[crossing_index]:
        remote = prepared.arc_partner[port]
        neighbor = prepared.crossing_for_port[remote]
        if neighbor >= 0 and neighbor != crossing_index:
            result.add(neighbor)
    return result


def _neighbor_ids(prepared, crossing_index):
    return {
        prepared.crossing_ids[index]
        for index in _neighbor_indices(prepared, crossing_index)
    }


def _first_virtual_inversion(prepared):
    """Probe inversions without rebuilding full resolution tables on misses.

    The physical arc matching is unchanged by inversion.  Only the cyclic order
    of the candidate crossing rotates, so the exact RII predicate can be tested
    against each physical neighbor using that virtual four-port order.  The
    immutable inverted state is constructed only after the first successful
    probe, and the same single RII pair as the retained implementation is then
    removed.
    """
    scans = 0
    checks = 0
    for crossing_index, ports in enumerate(prepared.ordered_ports):
        scans += 1
        virtual_ports = ports[1:] + ports[:1]
        for neighbor in sorted(_neighbor_indices(prepared, crossing_index)):
            checks += 1
            splices = _rii_splices_for_orders(
                prepared,
                virtual_ports,
                prepared.ordered_ports[neighbor],
            )
            if splices is None:
                continue
            inverted = invert_crossing(prepared, crossing_index)
            reduced = inverted._remove_reidemeister_ii_pair(  # noqa: SLF001
                crossing_index,
                neighbor,
                splices,
            )
            return (1, crossing_index, reduced), scans, checks
    return None, scans, checks


def _local_resolution_pairs(parent, crossing_index, child):
    child_index = {crossing_id: i for i, crossing_id in enumerate(child.crossing_ids)}
    local = sorted(
        child_index[crossing_id]
        for crossing_id in _neighbor_ids(parent, crossing_index)
        if crossing_id in child_index
    )
    return tuple((right, left) for left, right in itertools.combinations(local, 2))


def _first_local_resolution(prepared):
    scans = 0
    checks = 0
    for crossing_index in range(len(prepared.crossing_ids)):
        scans += 1
        children = []
        moves_total = 0
        for spin in (0, 1):
            child = resolve_crossing(prepared, crossing_index, spin)
            child, moves, child_checks = reduce_rii_queue(
                child,
                candidate_pairs=_local_resolution_pairs(prepared, crossing_index, child),
            )
            children.append(child)
            moves_total += moves
            checks += child_checks
        children.append(resolve_crossing(prepared, crossing_index, 2))
        if moves_total:
            return (moves_total, crossing_index, tuple(children)), scans, checks
    return None, scans, checks


def compute_locality_laurent(prepared, evaluator, *, memo=None, stats=None):
    """Exact structural evaluator with recursive RII canonicalization."""
    if memo is None:
        memo = _IsomorphicMemo()
    if stats is None:
        stats = {}
    for key in (
        "calls", "memo_hits", "rii_moves", "rii_pair_checks", "r1_moves",
        "r1_rebuilds", "inversion_steps", "inversion_crossing_scans",
        "local_rii_pair_checks", "resolution_steps", "resolution_crossing_scans",
        "resolution_local_pair_checks", "bulk_fallbacks", "max_bulk_crossings",
        "max_crossings_seen",
    ):
        stats.setdefault(key, 0)

    def rec(state):
        stats["calls"] += 1
        stats["max_crossings_seen"] = max(stats["max_crossings_seen"], len(state.crossing_ids))

        exponent = 0
        while True:
            state, rii_moves, rii_checks = reduce_rii_queue(state)
            stats["rii_moves"] += rii_moves
            stats["rii_pair_checks"] += rii_checks
            state, r1_shift, r1_moves = _reduce_r1_queue(state)
            exponent += r1_shift
            stats["r1_moves"] += r1_moves
            if r1_moves:
                stats["r1_rebuilds"] += 1
            if not rii_moves and not r1_moves:
                break

        if isinstance(memo, _IsomorphicMemo):
            hit, cached, key, index = memo.get(state)
            if hit:
                stats["memo_hits"] += 1
                return shift(cached, exponent)
        else:
            key = diagram_key(state)
            index = None
            if key in memo:
                stats["memo_hits"] += 1
                return shift(memo[key], exponent)

        crossing_count = len(state.crossing_ids)
        if crossing_count <= BULK_LEAF_MAX_CROSSINGS:
            stats["bulk_fallbacks"] += 1
            stats["max_bulk_crossings"] = max(stats["max_bulk_crossings"], crossing_count)
            value = evaluator.compute_prepared_bulk_laurent(state)
        else:
            inversion, scans, checks = _first_virtual_inversion(state)
            stats["inversion_crossing_scans"] += scans
            stats["local_rii_pair_checks"] += checks
            if inversion is not None:
                _, crossing_index, inverted = inversion
                stats["inversion_steps"] += 1
                positive = rec(resolve_crossing(state, crossing_index, 0))
                negative = rec(resolve_crossing(state, crossing_index, 1))
                value = add(rec(inverted), _skein_delta(positive, negative))
            else:
                resolution, scans, checks = _first_local_resolution(state)
                stats["resolution_crossing_scans"] += scans
                stats["resolution_local_pair_checks"] += checks
                if resolution is not None:
                    _, _, children = resolution
                    stats["resolution_steps"] += 1
                    positive = rec(children[0])
                    negative = rec(children[1])
                    vertex = rec(children[2])
                    value = add(add(shift(positive, 1), shift(negative, -1)), vertex)
                else:
                    stats["bulk_fallbacks"] += 1
                    stats["max_bulk_crossings"] = max(stats["max_bulk_crossings"], crossing_count)
                    value = evaluator.compute_prepared_bulk_laurent(state)

        if isinstance(memo, _IsomorphicMemo):
            memo.put(key, index, state, value)
        else:
            memo[key] = value
        return shift(value, exponent)

    return rec(prepared)
