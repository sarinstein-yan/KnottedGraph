"""Generic exact structural recursion for high-crossing Yamada diagrams.

The production evaluator contains no family-specific closed form.  It combines
exact regular-isotopy reductions with the Yamada skein identity, exact prepared-
diagram isomorphism memoization, and the retained native exhaustive kernel for
small irreducible residual states.

The current high-crossing path uses three locality optimizations that preserve
exactness:

* maximal Reidemeister-I curl chains are removed with a mutable port-matching
  queue and only one immutable diagram rebuild;
* skein inversion uses the first crossing that exposes an exact RII bigon rather
  than scanning every crossing to maximize a purely heuristic score; and
* after one crossing is inverted, only its physical neighbors are inspected for
  the newly created RII pair, because an unchanged remote pair cannot become RII.

The isomorphism fingerprint is only a bucket filter. Cache reuse occurs solely
after the native helper proves a full color/adjacency-preserving bijection, so
hash collisions cannot change a polynomial.
"""

from __future__ import annotations

import heapq

from .fast import add, shift
from .skein_hybrid import (
    _best_resolution,
    _resolution_tables,
    _skein_delta,
    diagram_key,
    invert_crossing,
    resolve_crossing,
)
from .state_compact import PreparedCompactStateBuilder

BULK_LEAF_MAX_CROSSINGS = 4

_STAT_KEYS = (
    "calls",
    "memo_hits",
    "rii_moves",
    "r1_moves",
    "r1_rebuilds",
    "inversion_steps",
    "inversion_crossing_scans",
    "local_rii_pair_checks",
    "resolution_steps",
    "bulk_fallbacks",
    "max_bulk_crossings",
    "max_crossings_seen",
)


class _IsomorphicMemo:
    """Exact prepared-diagram memo modulo relabeling.

    The compiled index is optional. On builds without it we fall back to the
    complete labeled prepared-diagram key, preserving correctness and
    portability.
    """

    def __init__(self):
        try:
            from . import _yamada_iso
        except Exception:  # pragma: no cover - compiler/platform fallback
            _yamada_iso = None
        self._native_module = _yamada_iso
        self._buckets = {}
        self._labeled = {}
        self.hits = 0

    @staticmethod
    def _native_index(module, prepared):
        return module.PreparedDiagramIndex(
            len(prepared.vertex_ids),
            [list(ports) for ports in prepared.ordered_ports],
            list(prepared.arc_partner),
            list(prepared.fixed_terminal_index),
            list(prepared.crossing_for_port),
        )

    def get(self, prepared):
        if self._native_module is None:
            key = diagram_key(prepared)
            if key in self._labeled:
                self.hits += 1
                return True, self._labeled[key], key, None
            return False, None, key, None

        index = self._native_index(self._native_module, prepared)
        key = (len(prepared.crossing_ids), index.node_count, index.fingerprint)
        for old_index, value in self._buckets.get(key, ()):
            if index.isomorphic(old_index):
                self.hits += 1
                return True, value, key, index
        return False, None, key, index

    def put(self, key, index, prepared, value):
        if self._native_module is None:
            self._labeled[key] = value
        else:
            self._buckets.setdefault(key, []).append((index, value))

    def __len__(self):
        if self._native_module is None:
            return len(self._labeled)
        return sum(len(bucket) for bucket in self._buckets.values())


def _r1_move(prepared, crossing_index, partner, active):
    """Return one exact R1 splice in the mutable matching, or ``None``."""
    if not active[crossing_index]:
        return None
    ports = prepared.ordered_ports[crossing_index]
    position = {port: index for index, port in enumerate(ports)}
    self_pairs = []
    for index, port in enumerate(ports):
        partner_position = position.get(partner[port])
        if partner_position is not None and index < partner_position:
            self_pairs.append((index, partner_position))
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


def _reduce_r1_queue(prepared):
    """Remove the same maximal R1 chain as sequential reduction, rebuilding once.

    A min-heap preserves the previous lowest-crossing-index removal order. After
    a splice only the two crossings adjacent to the changed physical arc can
    acquire a new self-adjacent curl, so only those crossings are rechecked.
    """
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

        partner[remote_left] = remote_right
        partner[remote_right] = remote_left
        for port in prepared.ordered_ports[crossing]:
            partner[port] = -1

        maybe_push(prepared.crossing_for_port[remote_left])
        maybe_push(prepared.crossing_for_port[remote_right])

    if not moves:
        return prepared, 0, 0

    removed_crossings = {index for index, keep in enumerate(active) if not keep}
    removed_ports = {
        port
        for crossing in removed_crossings
        for port in prepared.ordered_ports[crossing]
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

    surviving = [index for index in range(crossing_count) if active[index]]
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
        tuple(old_to_new[port] for port in prepared.ordered_ports[index])
        for index in surviving
    )
    plus, minus = _resolution_tables(new_ordered, len(new_partner))
    reduced = PreparedCompactStateBuilder(
        vertex_ids=prepared.vertex_ids,
        crossing_ids=tuple(prepared.crossing_ids[index] for index in surviving),
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


def _rii_pair(prepared, first: int, second: int):
    """Return exact RII splice data for one crossing pair, or ``None``."""
    if first == second or first < 0 or second < 0:
        return None
    first_ports = prepared.ordered_ports[first]
    second_ports = prepared.ordered_ports[second]
    second_position = {port: index for index, port in enumerate(second_ports)}
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


def _local_rii_after_inversion(inverted, crossing_index: int):
    """Remove one newly created RII pair containing the inverted crossing."""
    neighbors = set()
    for port in inverted.ordered_ports[crossing_index]:
        remote = inverted.arc_partner[port]
        neighbor = inverted.crossing_for_port[remote]
        if neighbor >= 0 and neighbor != crossing_index:
            neighbors.add(neighbor)

    pair_checks = 0
    for neighbor in sorted(neighbors):
        pair_checks += 1
        splices = _rii_pair(inverted, crossing_index, neighbor)
        if splices is not None:
            reduced = inverted._remove_reidemeister_ii_pair(  # noqa: SLF001
                crossing_index, neighbor, splices
            )
            return reduced, pair_checks
    return None, pair_checks


def _first_local_inversion(prepared):
    """Return the first inversion that exposes an exact local RII cancellation.

    Choosing the maximum number of subsequent RII moves is only a search
    heuristic, not part of the skein identity. Taking the first successful
    inversion preserves the exact polynomial while avoiding a full best-candidate
    scan over the diagram.
    """
    crossing_scans = 0
    pair_checks = 0
    for crossing_index in range(len(prepared.crossing_ids)):
        crossing_scans += 1
        inverted = invert_crossing(prepared, crossing_index)
        reduced, checks = _local_rii_after_inversion(inverted, crossing_index)
        pair_checks += checks
        if reduced is not None:
            return (1, crossing_index, reduced), crossing_scans, pair_checks
    return None, crossing_scans, pair_checks


def _prepare_stats(stats):
    if stats is None:
        stats = {}
    for key in _STAT_KEYS:
        stats.setdefault(key, 0)
    return stats


def compute_structural_laurent(prepared, evaluator, *, memo=None, stats=None):
    """Return the exact Laurent Yamada value of ``prepared``.

    ``evaluator`` supplies the existing exact crossing-free graph recurrence and
    exhaustive prepared-state fallback. No theorem-family recognizer or
    precomputed polynomial participates in this calculation.
    """
    if memo is None:
        memo = _IsomorphicMemo()
    stats = _prepare_stats(stats)

    # Preserve the previous generic benefit of cancelling any RII bigons already
    # present in an arbitrary user diagram, but do it only once at the root.
    prepared, root_rii_moves = prepared.reduce_reidemeister_ii()
    stats["rii_moves"] += root_rii_moves

    def rec(state):
        stats["calls"] += 1
        stats["max_crossings_seen"] = max(
            stats["max_crossings_seen"], len(state.crossing_ids)
        )

        state, exponent, r1_moves = _reduce_r1_queue(state)
        stats["r1_moves"] += r1_moves
        if r1_moves:
            stats["r1_rebuilds"] += 1

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
            stats["max_bulk_crossings"] = max(
                stats["max_bulk_crossings"], crossing_count
            )
            value = evaluator.compute_prepared_bulk_laurent(state)
        else:
            inversion, scans, checks = _first_local_inversion(state)
            stats["inversion_crossing_scans"] += scans
            stats["local_rii_pair_checks"] += checks
            if inversion is not None:
                _, crossing_index, inverted_reduced = inversion
                stats["inversion_steps"] += 1
                positive_value = rec(resolve_crossing(state, crossing_index, 0))
                negative_value = rec(resolve_crossing(state, crossing_index, 1))
                inverted_value = rec(inverted_reduced)
                value = add(
                    inverted_value,
                    _skein_delta(positive_value, negative_value),
                )
            else:
                # Exact fallback for diagrams without an RII-producing inversion.
                # The resolution lookahead is an optimization only; if it finds no
                # profitable child, the retained native exhaustive kernel is used.
                resolution = _best_resolution(state)
                if resolution is not None:
                    _, _, children = resolution
                    stats["resolution_steps"] += 1
                    positive_value = rec(children[0])
                    negative_value = rec(children[1])
                    vertex_value = rec(children[2])
                    value = add(
                        add(shift(positive_value, 1), shift(negative_value, -1)),
                        vertex_value,
                    )
                else:
                    stats["bulk_fallbacks"] += 1
                    stats["max_bulk_crossings"] = max(
                        stats["max_bulk_crossings"], crossing_count
                    )
                    value = evaluator.compute_prepared_bulk_laurent(state)

        if isinstance(memo, _IsomorphicMemo):
            memo.put(key, index, state, value)
        else:
            memo[key] = value
        return shift(value, exponent)

    return rec(prepared)
