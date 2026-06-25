"""Symmetry helpers for two-band non-Hermitian nodal models."""

from typing import Optional, Tuple, TypeVar

import sympy as sp

kSymbols = TypeVar("kSymbols", bound=Tuple[sp.Symbol, sp.Symbol, sp.Symbol])

__all__ = [
    "PT",
    "is_PT_symmetric",
]


def PT(
    h: sp.Matrix,
    k_symbols: Optional[kSymbols] = None,
) -> sp.Matrix:
    """Apply the two-band PT-symmetry operation to a Hamiltonian matrix."""
    if k_symbols is None:
        k_symbols = sorted(h.free_symbols, key=lambda s: s.name)
    sx = sp.Matrix([[0, 1], [1, 0]])
    return sx * sp.conjugate(h).xreplace({k: -k for k in k_symbols}) * sx


def is_PT_symmetric(
    h: sp.Matrix,
    k_symbols: Optional[kSymbols] = None,
) -> bool:
    """Return whether a two-band Hamiltonian is PT-symmetric."""
    return sp.simplify(h - PT(h, k_symbols=k_symbols)) == sp.zeros(2, 2)
