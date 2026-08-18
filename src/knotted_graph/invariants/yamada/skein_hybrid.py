"""Experimental exact crossing-recursion candidate for Yamada evaluation.

This module is intentionally not wired into the public evaluator yet. It is
used by regression tests and benchmarks to validate a hybrid strategy before
production dispatch changes: RII preprocessing, inversion look-ahead, exact
skein recursion, memoization, and fallback to the current native bulk state
sum when no structural reduction is exposed.
"""

from __future__ import annotations

import itertools

from .fast import add, scale, shift
from .state_compact import PreparedCompactStateBuilder, _MINUS_PAIRS, _PLUS_PAIRS

_STAT_KEYS = (
    "calls",
    "memo_hits",
    "rii_moves",
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
    crossing_ports = prepared.ordered_ports[crossing_index]
    removed_ports = set(crossing_ports)
    partner = list(prepared.arc_partner)

    for a, b in pairs:
        left_port = crossing_ports[a]
        right_port = crossing_ports[b]
        remote_left = partner[left_port]
        remote_right = partner[right_port]
        if remote_left in removed_ports or remote_right in removed_ports:
            raise ValueError("self-adjacent crossing is not supported by fast smoothing")
        if remote_left == remote_right:
            raise ValueError("degenerate smoothing would identify one endpoint with itself")
        partner[remote_left] = remote_right
        partner[remote_right] = remote_left

    active_ports = [
        port for port in range(len(prepared.arc_partner)) if port not in removed_ports
    ]
    old_to_new = {old: new for new, old in enumerate(active_ports)}
    surviving_crossings = [
        index
        for index in range(len(prepared.crossing_ids))
        if index != crossing_index
    ]
    crossing_remap = {old: new for new, old in enumerate(surviving_crossings)}

    new_arc_partner = tuple(old_to_new[partner[old]] for old in active_ports)
    new_crossing_for_port = []
    for old in active_ports:
        old_crossing = prepared.crossing_for_port[old]
        if old_crossing < 0:
            new_crossing_for_port.append(-1)
        elif old_crossing == crossing_index:
            raise RuntimeError("smoothing retained a removed crossing port")
        else:
            new_crossing_for_port.append(crossing_remap[old_crossing])

    new_ordered_ports = tuple(
        tuple(old_to_new[port] for port in prepared.ordered_ports[index])
        for index in surviving_crossings
    )
    plus_partner, minus_partner = _resolution_tables(
        new_ordered_ports, len(active_ports)
    )
    return PreparedCompactStateBuilder(
        vertex_ids=prepared.vertex_ids,
        crossing_ids=tuple(
            prepared.crossing_ids[index] for index in surviving_crossings
        ),
        ordered_ports=new_ordered_ports,
        arc_partner=new_arc_partner,
        fixed_terminal_index=tuple(
            prepared.fixed_terminal_index[old] for old in active_ports
        ),
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
        try:
            for spin in (0, 1, 2):
                child = resolve_crossing(prepared, crossing_index, spin)
                reduced, moves = child.reduce_reidemeister_ii()
                children.append(reduced)
                extra_moves += moves
        except ValueError:
            continue
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
    """Evaluate a prepared diagram with exact guarded structural recursion.

    The candidate recurses only when an inversion or an ordinary resolution
    exposes at least one additional RII cancellation. Otherwise it immediately
    falls back to the current native bulk evaluator. This keeps the generic
    irreducible case on the already optimized production path.
    """
    if memo is None:
        memo = {}
    stats = _prepare_stats(stats)
    stats["calls"] += 1

    prepared, moves = prepared.reduce_reidemeister_ii()
    stats["rii_moves"] += moves
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
        try:
            positive = resolve_crossing(prepared, crossing_index, 0)
            negative = resolve_crossing(prepared, crossing_index, 1)
        except ValueError:
            inversion = None
        else:
            stats["inversion_steps"] += 1
            positive_value = compute_hybrid_laurent(
                positive, evaluator, memo=memo, stats=stats
            )
            negative_value = compute_hybrid_laurent(
                negative, evaluator, memo=memo, stats=stats
            )
            inverted_value = compute_hybrid_laurent(
                inverted_reduced, evaluator, memo=memo, stats=stats
            )
            value = add(
                inverted_value,
                _skein_delta(positive_value, negative_value),
            )
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
