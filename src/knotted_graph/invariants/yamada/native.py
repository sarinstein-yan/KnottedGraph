"""Optional native backend for exact Yamada evaluation.

The public API remains pure Python. When the compiled extension is available,
the compact deletion--contraction recurrence and state summation run in C++.
If the native int64 coefficient fast path overflows, the computation is rerun
with the existing arbitrary-precision Python Laurent kernel, preserving exactness.

Prepared diagrams always use the same generic exact structural engine for
sufficiently large crossing count, regardless of whether the optional native
extension is importable. The extension accelerates only the low-level graph and
small residual state kernels; it no longer controls algorithmic dispatch.
No theorem-family recognizer or precomputed polynomial is part of production
dispatch; published formulas are used only by external validation benchmarks.
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

# Below this number of unresolved crossings the exhaustive prepared-state sum is
# normally faster than Python-level structural reduction. This is a performance
# policy only; both branches are algebraically exact and regression-compared.
STRUCTURAL_DISPATCH_MIN_CROSSINGS = 8


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
    """Structural dispatcher with optional native low-level acceleration.

    The historical class name is retained for API/internal compatibility. When
    the extension is unavailable, ``backend`` is ``"python"`` and every exact
    operation transparently uses the arbitrary-precision Python kernels, while
    high-crossing diagrams still take the optimized structural recursion.
    """

    def __init__(self, fallback_factory):
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._native = _yamada_native.NativeEvaluator() if native_available() else None
        self.native_calls = 0
        self.fallback_calls = 0
        # Retained for backward-compatible diagnostics. Production dispatch no
        # longer invokes the former theorem-backed theta shortcut.
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

        This method is both the guarded production fallback and an independent
        regression oracle for the structural high-crossing path.
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
        """Evaluate a prepared diagram with exact size-aware generic dispatch."""
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
    """Return one exact dispatcher, with native acceleration when available.

    Keeping the wrapper even on pure-Python installations is essential: the
    wrapper owns the optimized high-crossing structural algorithm. Falling back
    only changes the low-level leaf evaluator, never the mathematical strategy.
    """
    return NativeCompactEvaluator(fallback_factory)
