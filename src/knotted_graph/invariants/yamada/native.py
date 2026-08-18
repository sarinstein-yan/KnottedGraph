"""Optional native backend for exact Yamada evaluation.

The public API remains pure Python. When the compiled extension is available,
the compact deletion--contraction recurrence and state summation run in C++.
If the native int64 coefficient fast path overflows, the computation is rerun
with the existing arbitrary-precision Python Laurent kernel, preserving exactness.
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


class NativeCompactEvaluator:
    """Native compact evaluator with exact arbitrary-precision Python fallback."""

    def __init__(self, fallback_factory):
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._native = _yamada_native.NativeEvaluator() if native_available() else None
        self.native_calls = 0
        self.fallback_calls = 0

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

    def compute(self, graph, variable):
        from .fast import to_sympy

        return to_sympy(self.compute_laurent(graph), variable)


def make_native_or_python_evaluator(fallback_factory):
    """Return the native wrapper when built, otherwise the existing Python evaluator."""
    if native_available():
        return NativeCompactEvaluator(fallback_factory)
    return fallback_factory()
