"""Optional native kernels for exact resolved-graph Yamada evaluation.

Diagram-level production evaluation is implemented exclusively by the generic
factorized-connectivity dynamic program in :mod:`factorized_frontier`.  This
module therefore contains no crossing-count dispatch, structural skein router,
or benchmark-tuned frontier selection.

The compiled ``_yamada_native`` extension is retained for exact crossing-free
compact-graph evaluation and for the explicitly named exhaustive prepared-state
oracle used by validation.  If its int64 coefficient fast path overflows, the
arbitrary-precision Python Laurent kernel is used instead.
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
    """Return whether the resolved-graph native kernel is available."""
    return _yamada_native is not None


def native_import_error() -> Exception | None:
    return _NATIVE_IMPORT_ERROR


def _rows(graph: Any) -> list[list[int]]:
    return [list(row) for row in graph.rows]


def _as_laurent(value) -> tuple[tuple[int, int], ...]:
    return tuple((int(power), int(coefficient)) for power, coefficient in value)


class _MemoSizeProxy:
    def __init__(self, evaluator: "NativeCompactEvaluator"):
        self._evaluator = evaluator

    def __len__(self) -> int:
        return self._evaluator.memo_size


class NativeCompactEvaluator:
    """Exact compact evaluator for already-resolved crossing-free multigraphs."""

    def __init__(self, fallback_factory):
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._native = _yamada_native.NativeEvaluator() if native_available() else None
        self.native_calls = 0
        self.fallback_calls = 0
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
        """Evaluate already-resolved states; retained for compatibility APIs."""
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
        """Exhaustive 3**c prepared-state oracle for tests and validation only."""
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

    def compute(self, graph, variable):
        from .fast import to_sympy

        return to_sympy(self.compute_laurent(graph), variable)


def make_native_or_python_evaluator(fallback_factory):
    """Select native/Python only for resolved compact-graph evaluation."""
    if native_available():
        return NativeCompactEvaluator(fallback_factory)
    return fallback_factory()
