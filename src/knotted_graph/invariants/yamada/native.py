"""Optional native backend for exact Yamada evaluation.

The public API remains pure Python. When the compiled extension is available,
the compact deletion--contraction recurrence and state summation run in C++.
If the native int64 coefficient fast path overflows, the computation is rerun
with the arbitrary-precision Python Laurent kernel, preserving exactness.

Prepared diagrams use the generic exact structural engine for sufficiently large
crossing count. When that engine would otherwise fall back to a large exhaustive
3**c state sum, a conservative width-adaptive diagram-frontier backend may be
used instead. No theorem-family recognizer or precomputed polynomial is part of
production dispatch; published formulas are used only by external validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_NATIVE_IMPORT_ERROR: Exception | None = None
try:
    from . import _yamada_native
except Exception as exc:  # pragma: no cover - platform/build fallback
    _yamada_native = None
    _NATIVE_IMPORT_ERROR = exc

# Below this number of unresolved crossings the native exhaustive state sum is
# normally faster than Python-level structural reduction. This is a performance
# policy only; both branches are algebraically exact and regression-compared.
STRUCTURAL_DISPATCH_MIN_CROSSINGS = 8

# The direct diagram-frontier engine is deliberately only a replacement for a
# *large structural bulk fallback*. Existing R1/RII/inversion recursion remains
# first. These guards are conservative: the hard LLLV c=10/11 cases have planned
# peak 8, whereas the heterogeneous c=10 case that caused state growth has peak
# 12 and is rejected before frontier polynomial work begins.
FRONTIER_FALLBACK_MIN_CROSSINGS = 10
FRONTIER_MAX_PEAK_PORTS = 10
FRONTIER_MAX_STATES = 100_000


def native_available() -> bool:
    """Return whether the compiled Yamada backend is importable."""
    return _yamada_native is not None


def native_import_error() -> Exception | None:
    """Return the extension import error when the native backend is unavailable."""
    return _NATIVE_IMPORT_ERROR


def _rows(graph: Any) -> list[list[int]]:
    return [list(row) for row in graph.rows]


def _as_laurent(value) -> tuple[tuple[int, int], ...]:
    return tuple((int(power), int(coefficient)) for power, coefficient in value)


class _MemoSizeProxy:
    """Lightweight view for internal callers that inspect ``len(memo)``."""

    def __init__(self, evaluator: "NativeCompactEvaluator"):
        self._evaluator = evaluator

    def __len__(self) -> int:
        return self._evaluator.memo_size


class _StructuralBulkProxy:
    """Intercept only expensive structural bulk fallbacks with frontier DP."""

    def __init__(self, evaluator: "NativeCompactEvaluator", stats: dict):
        self._evaluator = evaluator
        self._stats = stats

    def compute_prepared_bulk_laurent(self, prepared):
        crossing_count = len(prepared.crossing_ids)
        if crossing_count >= FRONTIER_FALLBACK_MIN_CROSSINGS:
            from .diagram_frontier import (
                FrontierLimitExceeded,
                compute_diagram_frontier_laurent,
                plan_diagram_frontier,
            )

            plan = plan_diagram_frontier(prepared)
            self._stats["frontier_plans"] = self._stats.get("frontier_plans", 0) + 1
            self._stats["max_frontier_planned_peak"] = max(
                self._stats.get("max_frontier_planned_peak", 0),
                int(plan["peak_ports"]),
            )
            if plan["peak_ports"] <= FRONTIER_MAX_PEAK_PORTS:
                self._stats["frontier_attempts"] = self._stats.get("frontier_attempts", 0) + 1
                frontier_stats: dict = {}
                try:
                    value = compute_diagram_frontier_laurent(
                        prepared,
                        factor_order=plan["factor_order"],
                        max_peak_ports=FRONTIER_MAX_PEAK_PORTS,
                        max_states=FRONTIER_MAX_STATES,
                        stats=frontier_stats,
                    )
                except FrontierLimitExceeded:
                    self._stats["frontier_aborts"] = self._stats.get("frontier_aborts", 0) + 1
                else:
                    self._evaluator.frontier_calls += 1
                    self._stats["frontier_successes"] = self._stats.get("frontier_successes", 0) + 1
                    self._stats["max_frontier_states"] = max(
                        self._stats.get("max_frontier_states", 0),
                        int(frontier_stats.get("max_states", 0)),
                    )
                    self._stats["frontier_transitions"] = self._stats.get(
                        "frontier_transitions", 0
                    ) + int(frontier_stats.get("transitions", 0))
                    return value

        return self._evaluator.compute_prepared_bulk_laurent(prepared)


class NativeCompactEvaluator:
    """Native compact evaluator with exact arbitrary-precision Python fallback."""

    def __init__(self, fallback_factory):
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._native = _yamada_native.NativeEvaluator() if native_available() else None
        self.native_calls = 0
        self.fallback_calls = 0
        self.structural_calls = 0
        self.frontier_calls = 0
        self.last_structural_stats = None
        self.memo = _MemoSizeProxy(self)

    @property
    def backend(self) -> str:
        return "native" if self._native is not None else "python"

    @property
    def memo_size(self) -> int:
        native_size = int(self._native.memo_size) if self._native is not None else 0
        python_size = len(self._fallback.memo) if self._fallback is not None else 0
        return native_size + python_size

    def _python(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    def compute_laurent(self, graph):
        from .compact import CompactGraph

        compact = graph if isinstance(graph, CompactGraph) else CompactGraph.from_networkx(graph)
        if self._native is not None:
            try:
                self.native_calls += 1
                return _as_laurent(self._native.compute(_rows(compact)))
            except OverflowError:
                self.fallback_calls += 1
                return self._python().compute_laurent(compact)
        self.fallback_calls += 1
        return self._python().compute_laurent(compact)

    def compute_many_laurent(self, states: Iterable[tuple[Any, int]]):
        """Evaluate and sum a state stream in one native call when possible."""
        materialized = list(states)
        if self._native is not None:
            try:
                self.native_calls += 1
                return _as_laurent(
                    self._native.compute_many(
                        [_rows(graph) for graph, _ in materialized],
                        [int(exponent) for _, exponent in materialized],
                    )
                )
            except OverflowError:
                self.fallback_calls += 1

        from .fast import add, shift

        evaluator = self._python()
        total = ()
        for graph, exponent in materialized:
            total = add(total, shift(evaluator.compute_laurent(graph), int(exponent)))
        return total

    def compute_prepared_bulk_laurent(self, prepared):
        """Exact exhaustive prepared-state sum with no structural redispatch.

        This remains an independent regression oracle. Adaptive frontier dispatch
        is intentionally implemented by the structural proxy, not here.
        """
        if self._native is not None and hasattr(self._native, "compute_prepared"):
            try:
                self.native_calls += 1
                return _as_laurent(
                    self._native.compute_prepared(
                        len(prepared.vertex_ids),
                        len(prepared.crossing_ids),
                        list(prepared.arc_partner),
                        list(prepared.fixed_terminal_index),
                        list(prepared.crossing_for_port),
                        list(prepared.plus_partner),
                        list(prepared.minus_partner),
                    )
                )
            except OverflowError:
                self.fallback_calls += 1

        import itertools

        from .fast import add, shift

        evaluator = self._python()
        total = ()
        for config in itertools.product((0, 1, 2), repeat=len(prepared.crossing_ids)):
            total = add(
                total,
                shift(
                    evaluator.compute_laurent(prepared.build(config)),
                    config.count(0) - config.count(1),
                ),
            )
        return total

    def compute_prepared_laurent(self, prepared):
        """Evaluate a prepared diagram with exact size/width-aware dispatch."""
        crossing_count = len(prepared.crossing_ids)
        if crossing_count < STRUCTURAL_DISPATCH_MIN_CROSSINGS:
            return self.compute_prepared_bulk_laurent(prepared)

        from .diagram_locality import compute_locality_laurent

        stats: dict = {}
        proxy = _StructuralBulkProxy(self, stats)
        self.structural_calls += 1
        value = compute_locality_laurent(prepared, proxy, stats=stats)
        self.last_structural_stats = stats
        return value

    def compute(self, graph, variable):
        from .fast import to_sympy

        return to_sympy(self.compute_laurent(graph), variable)


def make_native_or_python_evaluator(fallback_factory):
    """Return the native wrapper when built, otherwise the Python evaluator."""
    if native_available():
        return NativeCompactEvaluator(fallback_factory)
    return fallback_factory()
