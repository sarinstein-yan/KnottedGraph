"""Experimental exact memoized resolution DAG for Yamada diagrams.

Unlike the production structural dispatcher, this engine does not fall back to
the exhaustive 3**c prepared-state sum merely because a crossing has no
immediate RII-producing resolution. It always resolves one crossing and caches
subproblems modulo the existing exact prepared-diagram isomorphism index. The
result is algebraically the same Yamada state sum, represented as a DAG rather
than an explicit ternary tree.

This module is experimental until the hard benchmark/validation gate promotes
it. It contains no family recognizers or closed-form Yamada formulas.
"""

from __future__ import annotations

from .diagram_locality import reduce_rii_queue
from .diagram_structural import _IsomorphicMemo, _reduce_r1_queue
from .fast import add, shift
from .skein_hybrid import resolve_crossing


def _crossing_neighbor_count(prepared, crossing_index: int) -> int:
    neighbors = set()
    for port in prepared.ordered_ports[crossing_index]:
        remote = prepared.arc_partner[port]
        other = prepared.crossing_for_port[remote]
        if other >= 0 and other != crossing_index:
            neighbors.add(other)
    return len(neighbors)


def _choose_crossing(prepared) -> int:
    """Choose a generic elimination crossing.

    Prefer the smallest crossing-frontier degree, analogous to min-degree
    elimination in graph/treewidth algorithms. Stable index tie-breaking makes
    the recursion deterministic. No graph-family information is used.
    """
    return min(
        range(len(prepared.crossing_ids)),
        key=lambda index: (_crossing_neighbor_count(prepared, index), index),
    )


def compute_memo_resolution_laurent(
    prepared,
    evaluator,
    *,
    bulk_leaf_crossings: int = 2,
    memo=None,
    stats=None,
):
    """Compute exact Yamada Laurent polynomial through an isomorphism-memo DAG."""
    if memo is None:
        memo = _IsomorphicMemo()
    if stats is None:
        stats = {}
    for key in (
        "calls",
        "memo_hits",
        "rii_moves",
        "r1_moves",
        "resolution_nodes",
        "bulk_leaves",
        "max_crossings_seen",
        "memo_size",
    ):
        stats.setdefault(key, 0)

    def rec(state):
        stats["calls"] += 1
        stats["max_crossings_seen"] = max(
            stats["max_crossings_seen"], len(state.crossing_ids)
        )

        exponent = 0
        while True:
            state, rii_moves, _checks = reduce_rii_queue(state)
            stats["rii_moves"] += rii_moves
            state, r1_shift, r1_moves = _reduce_r1_queue(state)
            exponent += r1_shift
            stats["r1_moves"] += r1_moves
            if not rii_moves and not r1_moves:
                break

        hit, cached, key, index = memo.get(state)
        if hit:
            stats["memo_hits"] += 1
            return shift(cached, exponent)

        crossing_count = len(state.crossing_ids)
        if crossing_count <= bulk_leaf_crossings:
            stats["bulk_leaves"] += 1
            value = evaluator.compute_prepared_bulk_laurent(state)
        else:
            stats["resolution_nodes"] += 1
            crossing_index = _choose_crossing(state)
            positive = rec(resolve_crossing(state, crossing_index, 0))
            negative = rec(resolve_crossing(state, crossing_index, 1))
            vertex = rec(resolve_crossing(state, crossing_index, 2))
            value = add(add(shift(positive, 1), shift(negative, -1)), vertex)

        memo.put(key, index, state, value)
        stats["memo_size"] = len(memo)
        return shift(value, exponent)

    return rec(prepared)
