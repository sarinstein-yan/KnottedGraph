"""Yamada polynomial evaluation backends."""

from . import polynomial as _polynomial
from . import recursive as _recursive
from .polynomial import *
from .recursive import *


def compute_yamada_polynomial_recursive(G, variable):
    """Compute crossing-free Yamada polynomial with the fastest exact backend.

    This preserves the historical public function name and signature. The
    original ``YamadaRecursiveEvaluator`` remains available as an independent
    SymPy reference implementation; production calls use the compact/native
    evaluator with transparent arbitrary-precision fallback.
    """
    from .compact import CompactYamadaEvaluator

    return CompactYamadaEvaluator().compute(G, variable)


# A direct import from ``knotted_graph.invariants.yamada.recursive`` should see
# the same optimized public helper after package initialization.
_recursive.compute_yamada_polynomial_recursive = compute_yamada_polynomial_recursive

__all__ = _polynomial.__all__ + _recursive.__all__
