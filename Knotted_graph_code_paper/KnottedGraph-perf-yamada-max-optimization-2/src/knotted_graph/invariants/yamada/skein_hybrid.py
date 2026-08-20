"""Exact local structural operations for Yamada crossing recursion.

This module contains no family-specific polynomial formula.  Every reduction is
an exact local identity of the Yamada state model and is regression-tested
against the retained exhaustive prepared-state evaluator.

In addition to ordinary crossing resolution and crossing inversion, smoothing
supports self-adjacent crossings.  Removing such a crossing may create a closed
terminal-free component; this is represented exactly as the same dummy
vertex/self-loop component used by :class:`PreparedCompactStateBuilder.build`.

The Reidemeister-I helper uses the regular-isotopy curl factors in the package's
crossing convention.  They were independently checked against exhaustive state
sums in both orientations and are also covered by committed regression tests.
"""

from __future__ import annotations

import itertools

from .fast import add, scale, shift
from .state_compact import PreparedCompactStateBuilder, _MINUS_PAIRS, _PLUS_PAIRS

_STAT_KEYS = (
    "calls",
    "memo_hits",
    "rii_moves",
    "r1_moves",
    "inversion_steps",
    "resolution_steps",
    "bulk_fallbacks",
)


def _resolution_tables(ordered_ports, port_count):
    plus_partner = [-1] * port_count
    minus_partner = [-1] * port_count
    for ports in ordered_ports:
        for a, b in _PLUS_PAIRS:
            pa, pb = ports[a], ports[b]
            plus_partner[pa] = pb
            plus_partner[pb] = pa
        for a, b in _MINUS_PAIRS:
            pa, pb = ports[a], ports[b]
            minus_partner[pa] = pb
            minus_partner[pb] = pa
    return tuple(plus_partner), tuple(minus_partner)


def _smooth_crossing(prepared, crossing_index, pairs):
    """Remove one crossing with an exact two-pair smoothing.

    Unlike the original fast smoother, this implementation also handles a
    physical arc joining two ports of the crossing itself.  The local graph made
    from the four removed ports, their physical-arc partners and the smoothing
    pairings is degree two.  Each connected component therefore has either two
    external endpoints, which are spliced, or no external endpoints, which is a
    detached circle.  Detached circles use the exact dummy-loop representation
    of the ordinary state builder.
    """
    crossing_ports = prepared.ordered_ports[crossing_index]
    removed_ports = set(crossing_ports)
    arc_partner = prepared.arc_partner

    adjacency: dict[int, set[int]] = {port: set() for port in crossing_ports}
    for port in crossing_ports:
        partner = arc_partner[port]
        adjacency.setdefault(port, set()).add(partner)
        adjacency.setdefault(partner, set()).add(port)
    for a, b in pairs:
        left = crossing_ports[a]
        right = crossing_ports[b]
        adjacency[left].add(right)
        adjacency[right].add(left)

    seen: set[int] = set()
    splices: list[tuple[int, int]] = []
    closed_loop_count = 0
    for start in tuple(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
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
        elif not external:
            closed_loop_count += 1
        else:  # pragma: no cover - malformed prepared table guard
            raise RuntimeError("malformed local smoothing component")

    active_ports = [
        port for port in range(len(arc_partner)) if port not in removed_ports
    ]
    old_to_new = {old: new for new, old in enumerate(active_ports)}
    updated_partner = list(arc_partner)
    for left, right in splices:
        updated_partner[left] = right
        updated_partner[right] = left

    new_arc_partner = []
    for old in active_ports:
        partner = updated_partner[old]
        if partner not in old_to_new:
            raise RuntimeError("smoothing left an edge attached to a removed port")
        new_arc_partner.append(old_to_new[partner])

    surviving_crossings = [
        index
        for index in range(len(prepared.crossing_ids))
        if index != crossing_index
    ]
    crossing_remap = {old: new for new, old in enumerate(surviving_crossings)}
    new_crossing_for_port = []
    for old in active_ports:
        crossing = prepared.crossing_for_port[old]
        if crossing < 0:
            new_crossing_for_port.append(-1)
        elif crossing == crossing_index:  # pragma: no cover - table invariant
            raise RuntimeError("removed crossing port survived smoothing")
        else:
            new_crossing_for_port.append(crossing_remap[crossing])

    new_ordered_ports = tuple(
        tuple(old_to_new[port] for port in prepared.ordered_ports[index])
        for index in surviving_crossings
    )
    new_fixed_terminal = [prepared.fixed_terminal_index[old] for old in active_ports]
    new_vertex_ids = list(prepared.vertex_ids)

    # Match PreparedCompactStateBuilder.build() exactly for every detached circle:
    # one fresh graph vertex carrying a single self-loop (two paired ports).
    next_id = max((*prepared.vertex_ids, *prepared.crossing_ids), default=-1) + 1
    for loop_index in range(closed_loop_count):
        vertex_index = len(new_vertex_ids)
        new_vertex_ids.append(next_id + loop_index)
        left = len(new_arc_partner)
        right = left + 1
        new_arc_partner.extend((right, left))
        new_fixed_terminal.extend((vertex_index, vertex_index))
        new_crossing_for_port.extend((-1, -1))

    plus_partner, minus_partner = _resolution_tables(
        new_ordered_ports, len(new_arc_partner)
    )
    return PreparedCompactStateBuilder(
        vertex_ids=tuple(new_vertex_ids),
        crossing_ids=tuple(
            prepared.crossing_ids[index] for index in surviving_crossings
        ),
        ordered_ports=new_ordered_ports,
        arc_partner=tuple(new_arc_partner),
        fixed_terminal_index=tuple(new_fixed_terminal),
        crossing_for_port=tuple(new_crossing_for_port),
        plus_partner=plus_partner,
        minus_partner=minus_partner,
    )


def resolve_crossing(prepared, crossing_index: int, spin: int):
    """Resolve one crossing while leaving every other crossing unresolved."""
    crossing_count = len(prepared.crossing_ids)
    if not 0 <= crossing_index < crossing_count:
        raise IndexError(crossing_index)
    if spin == 0:
        return _smooth_crossing(prepared, crossing_index, _PLUS_PAIRS)
    if spin == 1:
        return _smooth_crossing(prepared, crossing_index, _MINUS_PAIRS)
    if spin != 2:
        raise ValueError("invalid spin configuration")

    crossing_ports = set(prepared.ordered_ports[crossing_index])
    surviving_crossings = [
        index for index in range(crossing_count) if index != crossing_index
    ]
    crossing_remap = {old: new for new, old in enumerate(surviving_crossings)}

    new_vertex_index = len(prepared.vertex_ids)
    synthetic_id = max((*prepared.vertex_ids, *prepared.crossing_ids), default=-1) + 1
    fixed_terminal = list(prepared.fixed_terminal_index)
    crossing_for_port = list(prepared.crossing_for_port)
    for port in crossing_ports:
        fixed_terminal[port] = new_vertex_index
        crossing_for_port[port] = -1
    for port, old_crossing in enumerate(prepared.crossing_for_port):
        if old_crossing < 0 or old_crossing == crossing_index:
            continue
        crossing_for_port[port] = crossing_remap[old_crossing]

    new_ordered_ports = tuple(
        prepared.ordered_ports[index] for index in surviving_crossings
    )
    plus_partner, minus_partner = _resolution_tables(
        new_ordered_ports, len(prepared.arc_partner)
    )
    return PreparedCompactStateBuilder(
        vertex_ids=prepared.vertex_ids + (synthetic_id,),
        crossing_ids=tuple(
            prepared.crossing_ids[index] for index in surviving_crossings
        ),
        ordered_ports=new_ordered_ports,
        arc_partner=prepared.arc_partner,
        fixed_terminal_index=tuple(fixed_terminal),
        crossing_for_port=tuple(crossing_for_port),
        plus_partner=plus_partner,
        minus_partner=minus_partner,
    )


def invert_crossing(prepared, crossing_index: int):
    """Swap over/under information at one crossing."""
    crossing_count = len(prepared.crossing_ids)
    if not 0 <= crossing_index < crossing_count:
        raise IndexError(crossing_index)
    ordered_ports = list(prepared.ordered_ports)
    ports = ordered_ports[crossing_index]
    ordered_ports[crossing_index] = ports[1:] + ports[:1]
    ordered_ports = tuple(ordered_ports)
    plus_partner, minus_partner = _resolution_tables(
        ordered_ports, len(prepared.arc_partner)
    )
    return PreparedCompactStateBuilder(
        vertex_ids=prepared.vertex_ids,
        crossing_ids=prepared.crossing_ids,
        ordered_ports=ordered_ports,
        arc_partner=prepared.arc_partner,
        fixed_terminal_index=prepared.fixed_terminal_index,
        crossing_for_port=prepared.crossing_for_port,
        plus_partner=plus_partner,
        minus_partner=minus_partner,
    )


def find_reidemeister_i(prepared):
    """Return ``(reduced, exponent)`` for one exact curl, or ``None``.

    With crossing ports ordered cyclically and positions 0/2 over, the package's
    exact regular-isotopy convention is

    * self-pair (0,1) or (2,3): plus smoothing and factor ``A**-2``;
    * self-pair (0,3) or (1,2): minus smoothing and factor ``A**+2``.

    The rule is local and context independent.  Production tests compare both
    orientations directly with the retained exhaustive ``3**c`` evaluator.
    """
    for crossing_index, ports in enumerate(prepared.ordered_ports):
        position = {port: index for index, port in enumerate(ports)}
        self_pairs = []
        for index, port in enumerate(ports):
            partner_position = position.get(prepared.arc_partner[port])
            if partner_position is not None and index < partner_position:
                self_pairs.append((index, partner_position))
        if len(self_pairs) != 1:
            continue
        pattern = self_pairs[0]
        if pattern in ((0, 1), (2, 3)):
            reduced = _smooth_crossing(prepared, crossing_index, _PLUS_PAIRS)
            exponent = -2
        elif pattern in ((0, 3), (1, 2)):
            reduced = _smooth_crossing(prepared, crossing_index, _MINUS_PAIRS)
            exponent = 2
        else:
            continue
        # A Reidemeister-I removal reconnects the strand; it must not manufacture
        # a detached circle.  Keep this assertion as a structural guard.
        if len(reduced.vertex_ids) != len(prepared.vertex_ids):
            raise RuntimeError("R1 removal unexpectedly created a detached loop")
        return reduced, exponent
    return None


def reduce_reidemeister_i_chain(prepared):
    """Remove a maximal exact R1 chain and return total Laurent shift."""
    current = prepared
    exponent = 0
    moves = 0
    while True:
        found = find_reidemeister_i(current)
        if found is None:
            return current, exponent, moves
        current, delta = found
        exponent += delta
        moves += 1


def diagram_key(prepared):
    """Exact labeled memo key for a partially resolved diagram."""
    return (
        len(prepared.vertex_ids),
        prepared.ordered_ports,
        prepared.arc_partner,
        prepared.fixed_terminal_index,
        prepared.crossing_for_port,
    )


def _iter_states(prepared):
    crossing_count = len(prepared.crossing_ids)
    for config in itertools.product((0, 1, 2), repeat=crossing_count):
        yield prepared.build(config), config.count(0) - config.count(1)


def bulk_laurent(prepared, evaluator):
    """Evaluate with the existing exact compact/native bulk state sum."""
    states = _iter_states(prepared)
    if hasattr(evaluator, "compute_many_laurent"):
        return evaluator.compute_many_laurent(states)
    total = ()
    for graph, exponent in states:
        total = add(total, shift(evaluator.compute_laurent(graph), exponent))
    return total


def _skein_delta(positive, negative):
    """Return ``(A-A^-1) * (positive-negative)`` exactly."""
    difference = add(positive, scale(negative, -1))
    return add(shift(difference, 1), scale(shift(difference, -1), -1))


def _best_inversion(prepared):
    best = None
    for crossing_index in range(len(prepared.crossing_ids)):
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if moves and (best is None or moves > best[0]):
            best = moves, crossing_index, reduced
    return best


def _best_resolution(prepared):
    """Find a crossing whose ordinary resolutions expose additional RII moves."""
    best = None
    crossing_count = len(prepared.crossing_ids)
    for crossing_index in range(crossing_count):
        children = []
        extra_moves = 0
        for spin in (0, 1, 2):
            child = resolve_crossing(prepared, crossing_index, spin)
            reduced, moves = child.reduce_reidemeister_ii()
            children.append(reduced)
            extra_moves += moves
        if extra_moves and (best is None or extra_moves > best[0]):
            best = extra_moves, crossing_index, tuple(children)
    return best


def _prepare_stats(stats):
    if stats is None:
        stats = {}
    for key in _STAT_KEYS:
        stats.setdefault(key, 0)
    return stats


def compute_hybrid_laurent(prepared, evaluator, *, memo=None, stats=None):
    """Evaluate a prepared diagram with exact guarded structural recursion."""
    if memo is None:
        memo = {}
    stats = _prepare_stats(stats)
    stats["calls"] += 1

    prepared, moves = prepared.reduce_reidemeister_ii()
    stats["rii_moves"] += moves
    prepared, r1_shift, r1_moves = reduce_reidemeister_i_chain(prepared)
    stats["r1_moves"] += r1_moves
    if r1_moves:
        value = compute_hybrid_laurent(prepared, evaluator, memo=memo, stats=stats)
        return shift(value, r1_shift)

    key = diagram_key(prepared)
    cached = memo.get(key)
    if cached is not None:
        stats["memo_hits"] += 1
        return cached

    crossing_count = len(prepared.crossing_ids)
    if crossing_count == 0:
        value = evaluator.compute_laurent(prepared.build(()))
        memo[key] = value
        return value

    inversion = _best_inversion(prepared)
    if inversion is not None:
        _, crossing_index, inverted_reduced = inversion
        stats["inversion_steps"] += 1
        positive_value = compute_hybrid_laurent(
            resolve_crossing(prepared, crossing_index, 0),
            evaluator,
            memo=memo,
            stats=stats,
        )
        negative_value = compute_hybrid_laurent(
            resolve_crossing(prepared, crossing_index, 1),
            evaluator,
            memo=memo,
            stats=stats,
        )
        inverted_value = compute_hybrid_laurent(
            inverted_reduced, evaluator, memo=memo, stats=stats
        )
        value = add(inverted_value, _skein_delta(positive_value, negative_value))
        memo[key] = value
        return value

    resolution = _best_resolution(prepared)
    if resolution is not None:
        _, _, children = resolution
        stats["resolution_steps"] += 1
        positive_value = compute_hybrid_laurent(
            children[0], evaluator, memo=memo, stats=stats
        )
        negative_value = compute_hybrid_laurent(
            children[1], evaluator, memo=memo, stats=stats
        )
        vertex_value = compute_hybrid_laurent(
            children[2], evaluator, memo=memo, stats=stats
        )
        value = add(
            add(shift(positive_value, 1), shift(negative_value, -1)),
            vertex_value,
        )
        memo[key] = value
        return value

    stats["bulk_fallbacks"] += 1
    value = bulk_laurent(prepared, evaluator)
    memo[key] = value
    return value
