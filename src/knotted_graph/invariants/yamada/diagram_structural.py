"""Exact structural recursion for high-crossing Yamada diagrams.

The legacy prepared-state backend is exceptionally fast per resolved state, but
its worst-case diagram cost remains proportional to the full three-state space
``3**c`` after Reidemeister-II preprocessing.  This module attacks that outer
combinatorial layer without changing the Yamada invariant or its conventions.

It uses only exact identities already regression-tested in ``skein_hybrid``:

* regular-isotopy Reidemeister-II cancellation;
* crossing inversion plus the Yamada skein relation;
* ordinary three-way crossing resolution;
* memoization of exactly equal labelled partial diagrams.

When no structural reduction is exposed, evaluation falls back to the existing
native prepared-state sum.  The fallback is deliberately explicit so this
module can never recursively dispatch back into itself.
"""

from __future__ import annotations

from .fast import add, shift
from .skein_hybrid import (
    _best_inversion,
    _best_resolution,
    _skein_delta,
    diagram_key,
    resolve_crossing,
)

# Full three-child resolution look-ahead is useful for moderate diagrams but
# becomes unnecessary overhead on large diagrams when inversion already exposes
# a Reidemeister-II reduction.  The value is a performance policy only; it does
# not affect exactness because the alternative is the exact legacy bulk path.
RESOLUTION_LOOKAHEAD_MAX_CROSSINGS = 14

_STAT_KEYS = (
    "calls",
    "memo_hits",
    "rii_moves",
    "inversion_steps",
    "resolution_steps",
    "bulk_fallbacks",
    "max_crossings_seen",
)


def _prepare_stats(stats):
    if stats is None:
        stats = {}
    for key in _STAT_KEYS:
        stats.setdefault(key, 0)
    return stats


def compute_structural_laurent(prepared, evaluator, *, memo=None, stats=None):
    """Return the exact Laurent Yamada value of ``prepared``.

    ``evaluator`` must provide ``compute_laurent`` for crossing-free compact
    graphs and ``compute_prepared_bulk_laurent`` for the exact legacy fallback.
    Every recursive branch strictly reduces the number of unresolved crossings
    either directly or after an RII cancellation, so recursion terminates.
    """
    if memo is None:
        memo = {}
    stats = _prepare_stats(stats)
    stats["calls"] += 1

    prepared, moves = prepared.reduce_reidemeister_ii()
    stats["rii_moves"] += moves
    crossing_count = len(prepared.crossing_ids)
    stats["max_crossings_seen"] = max(stats["max_crossings_seen"], crossing_count)

    key = diagram_key(prepared)
    cached = memo.get(key)
    if cached is not None:
        stats["memo_hits"] += 1
        return cached

    if crossing_count == 0:
        value = evaluator.compute_laurent(prepared.build(()))
        memo[key] = value
        return value

    # The most valuable exact move on repeated braid/twist structure is to
    # invert one crossing when doing so creates an RII pair.  The skein identity
    # then replaces the original diagram by one diagram with two fewer crossings
    # plus the two one-crossing resolutions.  All three children are exact.
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
            positive_value = compute_structural_laurent(
                positive, evaluator, memo=memo, stats=stats
            )
            negative_value = compute_structural_laurent(
                negative, evaluator, memo=memo, stats=stats
            )
            inverted_value = compute_structural_laurent(
                inverted_reduced, evaluator, memo=memo, stats=stats
            )
            value = add(
                inverted_value,
                _skein_delta(positive_value, negative_value),
            )
            memo[key] = value
            return value

    # A direct three-state resolution is worthwhile only when at least one child
    # immediately exposes additional RII cancellation.  For large generic
    # diagrams we avoid an O(c^2) look-ahead scan and preserve the optimized
    # native bulk evaluator as the guarded exact fallback.
    if crossing_count <= RESOLUTION_LOOKAHEAD_MAX_CROSSINGS:
        resolution = _best_resolution(prepared)
        if resolution is not None:
            _, _, children = resolution
            stats["resolution_steps"] += 1
            positive_value = compute_structural_laurent(
                children[0], evaluator, memo=memo, stats=stats
            )
            negative_value = compute_structural_laurent(
                children[1], evaluator, memo=memo, stats=stats
            )
            vertex_value = compute_structural_laurent(
                children[2], evaluator, memo=memo, stats=stats
            )
            value = add(
                add(shift(positive_value, 1), shift(negative_value, -1)),
                vertex_value,
            )
            memo[key] = value
            return value

    stats["bulk_fallbacks"] += 1
    value = evaluator.compute_prepared_bulk_laurent(prepared)
    memo[key] = value
    return value
