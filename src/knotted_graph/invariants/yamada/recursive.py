"""Deprecated compatibility wrappers for crossing-free Yamada evaluation.

The historical NetworkX/SymPy recurrence no longer exists in production.  The
legacy function/class names below are retained only so older user code imports
continue to work; every call immediately dispatches to the current compact/native
exact evaluator.
"""

from __future__ import annotations

import warnings

import sympy as sp

from .algebra import laurent_y_to_sigma_polynomial
from .compact import CompactYamadaEvaluator

__all__ = [
    "YamadaRecursiveEvaluator",
    "compute_yamada_polynomial_recursive",
    "laurent_y_to_sigma_polynomial",
]


def _warn() -> None:
    warnings.warn(
        "The recursive Yamada compatibility API is deprecated; use "
        "compute_graph_yamada_polynomial instead. The computation already uses "
        "the current compact/native exact evaluator.",
        DeprecationWarning,
        stacklevel=3,
    )


class YamadaRecursiveEvaluator:
    """Compatibility adapter backed exclusively by the current exact evaluator."""

    def __init__(self, variable: sp.Symbol):
        self.variable = variable
        self._evaluator = CompactYamadaEvaluator()

    def compute(self, graph) -> sp.Expr:
        _warn()
        return self._evaluator.compute(graph, self.variable)


def compute_yamada_polynomial_recursive(graph, variable: sp.Symbol) -> sp.Expr:
    """Deprecated alias for :func:`compute_graph_yamada_polynomial`."""
    _warn()
    return CompactYamadaEvaluator().compute(graph, variable)
