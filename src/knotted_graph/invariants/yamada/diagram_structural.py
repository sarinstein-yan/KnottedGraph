"""Generic exact structural recursion for high-crossing Yamada diagrams.

The evaluator deliberately contains no family-specific closed form.  It reduces
crossing complexity with exact regular-isotopy RII/R1 moves and the Yamada skein
identity, quotients repeated partial diagrams by exact combinatorial
isomorphism when the optional native helper is available, and leaves only small
residual state sums to the retained exhaustive native kernel.

The isomorphism fingerprint is only a bucket filter.  Cache reuse occurs solely
after the C++ helper proves a full color/adjacency-preserving bijection, so hash
collisions cannot change a polynomial.
"""

from __future__ import annotations

from .fast import add, shift
from .skein_hybrid import (
    _best_inversion,
    _best_resolution,
    _skein_delta,
    diagram_key,
    find_reidemeister_i,
    resolve_crossing,
)

BULK_LEAF_MAX_CROSSINGS = 4

_STAT_KEYS = (
    "calls",
    "memo_hits",
    "rii_moves",
    "r1_moves",
    "inversion_steps",
    "resolution_steps",
    "bulk_fallbacks",
    "max_bulk_crossings",
    "max_crossings_seen",
)


class _IsomorphicMemo:
    """Exact prepared-diagram memo modulo relabeling.

    The compiled index is optional.  On builds without it we fall back to the
    exact labeled key, preserving correctness and portability.
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


def _prepare_stats(stats):
    if stats is None:
        stats = {}
    for key in _STAT_KEYS:
        stats.setdefault(key, 0)
    return stats


def compute_structural_laurent(prepared, evaluator, *, memo=None, stats=None):
    """Return the exact Laurent Yamada value of ``prepared``.

    ``evaluator`` supplies the existing exact crossing-free graph recurrence and
    exhaustive prepared-state fallback.  The outer recursion itself never uses
    a precomputed polynomial or a theorem-family recognizer.
    """
    if memo is None:
        memo = _IsomorphicMemo()
    stats = _prepare_stats(stats)
    stats["calls"] += 1

    prepared, moves = prepared.reduce_reidemeister_ii()
    stats["rii_moves"] += moves
    crossing_count = len(prepared.crossing_ids)
    stats["max_crossings_seen"] = max(stats["max_crossings_seen"], crossing_count)

    # R1 is a multiplicative local identity.  Apply it before memoization so the
    # cache stores the simpler diagram and its value independent of curl count.
    curl = find_reidemeister_i(prepared)
    if curl is not None:
        reduced, exponent = curl
        stats["r1_moves"] += 1
        return shift(
            compute_structural_laurent(reduced, evaluator, memo=memo, stats=stats),
            exponent,
        )

    if isinstance(memo, _IsomorphicMemo):
        hit, cached, key, index = memo.get(prepared)
        if hit:
            stats["memo_hits"] += 1
            return cached
    else:
        key = diagram_key(prepared)
        index = None
        if key in memo:
            stats["memo_hits"] += 1
            return memo[key]

    if crossing_count <= BULK_LEAF_MAX_CROSSINGS:
        stats["bulk_fallbacks"] += 1
        stats["max_bulk_crossings"] = max(
            stats["max_bulk_crossings"], crossing_count
        )
        value = evaluator.compute_prepared_bulk_laurent(prepared)
    else:
        inversion = _best_inversion(prepared)
        if inversion is not None:
            _, crossing_index, inverted_reduced = inversion
            stats["inversion_steps"] += 1
            positive_value = compute_structural_laurent(
                resolve_crossing(prepared, crossing_index, 0),
                evaluator,
                memo=memo,
                stats=stats,
            )
            negative_value = compute_structural_laurent(
                resolve_crossing(prepared, crossing_index, 1),
                evaluator,
                memo=memo,
                stats=stats,
            )
            inverted_value = compute_structural_laurent(
                inverted_reduced, evaluator, memo=memo, stats=stats
            )
            value = add(
                inverted_value,
                _skein_delta(positive_value, negative_value),
            )
        else:
            # Direct three-state recursion is accepted only when it exposes
            # further exact RII cancellation.  Generic irreducible diagrams stay
            # on the optimized exhaustive native kernel rather than suffering an
            # uncontrolled Python recursion-tree expansion.
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
            else:
                stats["bulk_fallbacks"] += 1
                stats["max_bulk_crossings"] = max(
                    stats["max_bulk_crossings"], crossing_count
                )
                value = evaluator.compute_prepared_bulk_laurent(prepared)

    if isinstance(memo, _IsomorphicMemo):
        memo.put(key, index, prepared, value)
    else:
        memo[key] = value
    return value
