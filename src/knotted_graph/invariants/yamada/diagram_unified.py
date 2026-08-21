"""Single exact rewrite-and-contract evaluator for prepared Yamada diagrams.

This module deliberately has no empirical solver dispatch. Every state follows
the same algorithm:

1. apply exact RII and R1 reductions;
2. hash-cons the reduced prepared diagram modulo relabeling;
3. if an exact local skein rewrite creates a topological cancellation, recurse
   on that identity and reduce its children;
4. otherwise contract the residual diagram by the polynomial-valued
   connectivity frontier.

The terminal contraction is compiled when available and falls back to the same
exact Python connectivity DP on platforms without a compiler or on int64
coefficient overflow. No crossing-count, width, benchmark-family or measured
runtime threshold participates in the mathematical path.
"""

from __future__ import annotations

from .diagram_frontier import compute_diagram_frontier_laurent, plan_diagram_frontier
from .diagram_structural import (
    _IsomorphicMemo,
    _first_local_inversion,
    _reduce_r1_queue,
)
from .fast import add, shift
from .skein_hybrid import _best_resolution, _skein_delta, diagram_key, resolve_crossing

try:
    from . import _yamada_frontier_dense as _native_frontier
except Exception:  # pragma: no cover - compiler/platform fallback
    try:
        from . import _yamada_frontier as _native_frontier
    except Exception:  # pragma: no cover - compiler/platform fallback
        _native_frontier = None


def native_frontier_available() -> bool:
    return _native_frontier is not None


def _as_laurent(value) -> tuple[tuple[int, int], ...]:
    return tuple((int(power), int(coefficient)) for power, coefficient in value)


def contract_frontier_laurent(prepared, *, stats=None):
    """Contract one residual prepared diagram by polynomial connectivity states."""
    if stats is None:
        stats = {}
    plan = plan_diagram_frontier(prepared)
    stats["contractions"] = stats.get("contractions", 0) + 1
    stats["max_contract_peak_ports"] = max(
        int(stats.get("max_contract_peak_ports", 0)), int(plan["peak_ports"])
    )

    if _native_frontier is not None:
        try:
            value = _native_frontier.compute_prepared_frontier(
                len(prepared.vertex_ids),
                len(prepared.crossing_ids),
                list(prepared.arc_partner),
                list(prepared.fixed_terminal_index),
                list(prepared.crossing_for_port),
                list(prepared.plus_partner),
                list(prepared.minus_partner),
                list(plan["factor_order"]),
            )
        except OverflowError:
            stats["native_frontier_overflows"] = stats.get(
                "native_frontier_overflows", 0
            ) + 1
        else:
            stats["native_frontier_calls"] = stats.get("native_frontier_calls", 0) + 1
            return _as_laurent(value)

    stats["python_frontier_calls"] = stats.get("python_frontier_calls", 0) + 1
    frontier_stats = {}
    value = compute_diagram_frontier_laurent(
        prepared,
        factor_order=plan["factor_order"],
        stats=frontier_stats,
    )
    stats["max_python_frontier_states"] = max(
        int(stats.get("max_python_frontier_states", 0)),
        int(frontier_stats.get("max_states", 0)),
    )
    return value


def compute_unified_laurent(prepared, *, memo=None, stats=None):
    """Evaluate exactly with one reduction + connectivity-contraction algorithm."""
    if memo is None:
        memo = _IsomorphicMemo()
    if stats is None:
        stats = {}
    for key in (
        "calls",
        "memo_hits",
        "rii_moves",
        "r1_moves",
        "r1_rebuilds",
        "inversion_steps",
        "inversion_crossing_scans",
        "local_rii_pair_checks",
        "resolution_steps",
        "contractions",
        "max_crossings_seen",
    ):
        stats.setdefault(key, 0)

    def rec(state):
        stats["calls"] += 1
        stats["max_crossings_seen"] = max(
            int(stats["max_crossings_seen"]), len(state.crossing_ids)
        )

        state, rii_moves = state.reduce_reidemeister_ii()
        stats["rii_moves"] += rii_moves
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

        inversion, scans, checks = _first_local_inversion(state)
        stats["inversion_crossing_scans"] += scans
        stats["local_rii_pair_checks"] += checks
        if inversion is not None:
            _, crossing_index, inverted_reduced = inversion
            stats["inversion_steps"] += 1
            positive_value = rec(resolve_crossing(state, crossing_index, 0))
            negative_value = rec(resolve_crossing(state, crossing_index, 1))
            inverted_value = rec(inverted_reduced)
            value = add(inverted_value, _skein_delta(positive_value, negative_value))
        else:
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
                value = contract_frontier_laurent(state, stats=stats)

        if isinstance(memo, _IsomorphicMemo):
            memo.put(key, index, state, value)
        else:
            memo[key] = value
        return shift(value, exponent)

    return rec(prepared)
