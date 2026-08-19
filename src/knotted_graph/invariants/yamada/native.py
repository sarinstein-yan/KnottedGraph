"""Optional native backend for exact Yamada evaluation.

The public API remains pure Python. When the compiled extension is available,
the compact deletion--contraction recurrence and state summation run in C++.
If the native int64 coefficient fast path overflows, the computation is rerun
with the existing arbitrary-precision Python Laurent kernel, preserving exactness.

Prepared diagrams first pass through conservative exact structural fast paths.
The certified Dobrynin--Vesnin Theta(n) family is evaluated by its published
closed form. Other high-crossing diagrams may use exact skein/RII recursion
before falling back to the legacy native ``3**c`` state sum. Small generic
diagrams stay on the legacy C++ path because its constant factors are excellent.
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
# normally faster than Python-level structural look-ahead. This is a dispatch
# policy only; both branches are algebraically exact and regression-compared.
STRUCTURAL_DISPATCH_MIN_CROSSINGS = 10


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
    """Compatibility object for internal callers that only inspect ``len(memo)``."""

    def __init__(self, evaluator: "NativeCompactEvaluator"):
        self._evaluator = evaluator

    def __len__(self) -> int:
        return self._evaluator.memo_size


class NativeCompactEvaluator:
    """Native compact evaluator with exact arbitrary-precision Python fallback."""

    def __init__(self, fallback_factory):
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._native = _yamada_native.NativeEvaluator() if native_available() else None
        self.native_calls = 0
        self.fallback_calls = 0
        self.theta_twist_calls = 0
        self.structural_calls = 0
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
                # Exactness takes priority over speed. Python integers are arbitrary precision.
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
        """Exact legacy prepared-state sum with no structural redispatch.

        This method exists both as the production fallback and as an independent
        regression oracle for the structural high-crossing paths.
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

        # Preserve exact arbitrary-precision behavior on non-native platforms or
        # int64 overflow by evaluating the identical prepared state definition.
        import itertools
        from .fast import add, shift

        evaluator = self._python()
        total = ()
        for config in itertools.product(
            (0, 1, 2), repeat=len(prepared.crossing_ids)
        ):
            total = add(
                total,
                shift(
                    evaluator.compute_laurent(prepared.build(config)),
                    config.count(0) - config.count(1),
                ),
            )
        return total

    def compute_prepared_laurent(self, prepared):
        """Evaluate a prepared diagram with exact size-aware structural dispatch."""
        # First try the strongest possible optimization: a mathematically
        # certified family with a published closed form. The recognizer returns
        # None for every diagram not proven to be the canonical odd-n theta
        # two-braid family, so there is no heuristic acceptance here.
        from .theta_twist_prepared import certified_prepared_theta_twist_laurent

        theta_value = certified_prepared_theta_twist_laurent(prepared)
        if theta_value is not None:
            self.theta_twist_calls += 1
            return theta_value

        crossing_count = len(prepared.crossing_ids)
        if crossing_count < STRUCTURAL_DISPATCH_MIN_CROSSINGS:
            return self.compute_prepared_bulk_laurent(prepared)

        from .diagram_structural import compute_structural_laurent

        stats = {}
        self.structural_calls += 1
        value = compute_structural_laurent(prepared, self, stats=stats)
        self.last_structural_stats = stats
        return value

    def compute(self, graph, variable):
        from .fast import to_sympy

        return to_sympy(self.compute_laurent(graph), variable)


def make_native_or_python_evaluator(fallback_factory):
    """Return the native wrapper when built, otherwise the existing Python evaluator."""
    if native_available():
        return NativeCompactEvaluator(fallback_factory)
    return fallback_factory()
