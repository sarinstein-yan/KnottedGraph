import sympy as sp
from typing import Tuple, TypeVar
kSymbols = TypeVar('kSymbols', bound=Tuple[sp.Symbol, sp.Symbol, sp.Symbol])

__all__ = [
    'pq_torus_knot_bloch_vector',
    'unknot_bloch_vector',
    'hopf_link_bloch_vector',
    'trefoil_bloch_vector',
    'solomon_bloch_vector',
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
    z = sp.cos(2 * kz) + sp.Rational(1, 3) + sp.I * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - 2)
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
    cx = sp.expand(sp.re(f))
    cz = sp.expand(sp.im(f))
    return (cx, gamma * sp.I, cz)

def pq_torus_knot_bloch_vector(
        p: int,
        q: int,
        gamma: float,
        k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for a pq-torus knot.

    Parameters
    ----------
    p : int
        The first parameter of the pq-torus knot.
        The number of toroidal twists.
    q : int
        The second parameter of the pq-torus knot.
        The number of poloidal (meridian) twists.
    gamma : float
        The scaling factor for the imaginary component.
    k_symbols : kSymbols, optional
        Tuple of k-space symbols (kx, ky, kz). Defaults to real symbols.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
        Bloch vector components.
    """
    z, w = _k2zw(k_symbols)
    return _zw2d(z**p - w**q, gamma)

def unknot_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for the unknot.

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
    return pq_torus_knot_bloch_vector(1, 1, gamma, k_symbols)

def hopf_link_bloch_vector(
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
    return pq_torus_knot_bloch_vector(2, 2, gamma, k_symbols)

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
    return pq_torus_knot_bloch_vector(2, 3, gamma, k_symbols)

def solomon_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True)
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for the Solomon knot.

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
    return pq_torus_knot_bloch_vector(2, 4, gamma, k_symbols)

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