import sympy as sp
from typing import Tuple, TypeVar
kSymbols = TypeVar('kSymbols', bound=Tuple[sp.Symbol, sp.Symbol, sp.Symbol])

__all__ = [
    'hopf_bloch_vector',
    'trefoil_bloch_vector',
    'threelink_bloch_vector',
    'awesome_bloch_vector'
]

def _k2zw(
    k_symbols: kSymbols
    ) -> tuple[sp.Expr, sp.Expr]:
    """Converts k-space symbols to z and w complex variables.

    Parameters
    ----------
    k_symbols : kSymbols
    Tuple of k-space symbols (kx, ky, kz).

    Returns
    -------
    tuple[sp.Expr, sp.Expr]
    Complex variables z and w.
    """
    kx, ky, kz = k_symbols
    z = sp.cos(2 * kz) + sp.Rational(1, 2) + sp.I * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - 2)
    w = sp.sin(kx) + sp.I * sp.sin(ky)
    return z, w

def _zw2d(
    f: sp.Expr, 
    gamma: sp.Symbol
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Converts a complex variable to Bloch vector components.

    Parameters
    ----------
    f : sp.Expr
    Complex variable.
    gamma : sp.Symbol
    Scaling factor for the imaginary component.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
    Bloch vector components (cx, gamma * I, cz).
    """
    cx = sp.simplify(sp.re(f))
    cz = sp.simplify(sp.im(f))
    return (cx, gamma * sp.I, cz)

def hopf_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for the Hopf link.

    Parameters
    ----------
    gamma : sp.Symbol
    Scaling factor for the imaginary component.
    k_symbols : kSymbols, optional
    Tuple of k-space symbols (kx, ky, kz). Defaults to real symbols.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
    Bloch vector components.
    """
    z, w = _k2zw(k_symbols)
    return _zw2d(z**2 - w**2, gamma)

def trefoil_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for the trefoil knot.

    Parameters
    ----------
    gamma : sp.Symbol
    Scaling factor for the imaginary component.
    k_symbols : kSymbols, optional
    Tuple of k-space symbols (kx, ky, kz). Defaults to real symbols.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
    Bloch vector components.
    """
    z, w = _k2zw(k_symbols)
    return _zw2d(z * (z**2 - w**4 + w), gamma)

def threelink_bloch_vector(
    gamma: sp.Symbol, k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for the three-link.

    Parameters
    ----------
    gamma : sp.Symbol
    Scaling factor for the imaginary component.
    k_symbols : kSymbols, optional
    Tuple of k-space symbols (kx, ky, kz). Defaults to real symbols.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
    Bloch vector components.
    """
    z, w = _k2zw(k_symbols)
    return _zw2d((z**2 - w**2) * z, gamma)

def awesome_bloch_vector(
    gamma: sp.Symbol, k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for an awesome knot.

    Parameters
    ----------
    gamma : sp.Symbol
    Scaling factor for the imaginary component.
    k_symbols : kSymbols, optional
    Tuple of k-space symbols (kx, ky, kz). Defaults to real symbols.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
    Bloch vector components.
    """
    z, w = _k2zw(k_symbols)
    return _zw2d(z * (z**2 - w**4 + w), gamma)