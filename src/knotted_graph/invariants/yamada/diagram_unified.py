"""Single exact factorized-connectivity evaluator for prepared Yamada diagrams.

Every prepared diagram follows the same mathematical path: factor high-arity
fixed-vertex equalities into low-arity equality constraints joined by identity
wires, then contract the resulting polynomial-valued connectivity frontier.
There is no empirical solver dispatch, crossing-count threshold, benchmark-family
special case, or structural skein-recursion branch in this evaluator.

The retained raw frontier routines below remain available as an independent exact
reference and portability fallback for diagnostics/tests.
"""

from __future__ import annotations

from .diagram_frontier import compute_diagram_frontier_laurent, plan_diagram_frontier
from .factorized_frontier import compute_factorized_frontier_laurent

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
    """Contract one prepared diagram with the retained raw connectivity DP."""
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
    """Evaluate exactly using the single generic factorized frontier algorithm."""
    # ``memo`` is retained for API compatibility with the research diagnostics;
    # this non-recursive evaluator needs no diagram-level memo table.
    del memo
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
        stats["contractions"] = stats.get("contractions", 0) + 1
        stats["max_crossings_seen"] = max(
            int(stats.get("max_crossings_seen", 0)), len(prepared.crossing_ids)
        )
    return compute_factorized_frontier_laurent(prepared)
