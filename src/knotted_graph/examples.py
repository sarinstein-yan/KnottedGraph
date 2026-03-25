import sympy as sp
from typing import Tuple, TypeVar
kSymbols = TypeVar('kSymbols', bound=Tuple[sp.Symbol, sp.Symbol, sp.Symbol])

__all__ = [
    'pq_torus_knot_bloch_vector',
    'unknot_bloch_vector',
    'hopf_link_bloch_vector',
    'trefoil_bloch_vector',
    'cinquefoil_bloch_vector',
    'solomon_bloch_vector',
    'threelink_bloch_vector',
    'awesome_bloch_vector',
    # --- mirror-symmetric (legacy) variants ---
    'pq_torus_knot_mirror_bloch_vector',
    'unknot_mirror_bloch_vector',
    'hopf_link_mirror_bloch_vector',
    'trefoil_mirror_bloch_vector',
    'cinquefoil_mirror_bloch_vector',
    'solomon_mirror_bloch_vector',
    'threelink_mirror_bloch_vector',
    'awesome_mirror_bloch_vector',
]

def _k2zw(
    k_symbols: kSymbols,
    c: float = 0.0,
    m: float = 2.0,
    mirror_n3: bool = False,
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
    n1 = sp.sin(kx)
    n2 = sp.sin(ky)
    if mirror_n3:
        n3 = sp.cos(2 * kz) + sp.nsimplify(c)
    else:
        n3 = sp.sin(kz) + sp.nsimplify(c)
    n4 = sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - sp.nsimplify(m)
    
    w = n1 + sp.I * n2
    z = n3 + sp.I * n4
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
        k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
        c: float = 0.0,
        m: float = 2.0,
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
    z, w = _k2zw(k_symbols, c, m)
    return _zw2d(z**p + w**q, gamma)

def unknot_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
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
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
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
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
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

def cinquefoil_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Computes the Bloch vector components for the cinquefoil knot.

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
    return pq_torus_knot_bloch_vector(2, 5, gamma, k_symbols)

def solomon_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
    c: float = 0.0,
    m: float = 2.0,
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
    return pq_torus_knot_bloch_vector(2, 4, gamma, k_symbols, c, m)

def threelink_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
    c: float = 0.0,
    m: float = 2.0,
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
    z, w = _k2zw(k_symbols, c, m)
    return _zw2d((z**2 - w**2) * z, gamma)

def awesome_bloch_vector(
    gamma: sp.Symbol, 
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
    c: float = 0.0,
    m: float = 2.0,
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
    z, w = _k2zw(k_symbols, c, m)
    return _zw2d(z * (z**2 - w**4 + w), gamma)

# -----------------------------------------------------------------------------
# Mirror-symmetric (legacy) variants
# -----------------------------------------------------------------------------

def pq_torus_knot_mirror_bloch_vector(
        p: int,
        q: int,
        gamma: float,
        k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
        c: float = 0.5,
        m: float = 2.0,
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) pq-torus-knot ansatz.

    Compared to :func:`pq_torus_knot_bloch_vector`, this variant matches the
    older mirror-symmetric formulation (formerly in ``examples_mirror.py``):

    - Uses ``n3 = cos(2*kz) + c`` (instead of ``sin(kz) + c``)
    - Uses ``z**p - w**q`` (instead of ``z**p + w**q``)

    Parameters
    ----------
    p : int
        The first parameter of the pq-torus knot (toroidal twists).
    q : int
        The second parameter of the pq-torus knot (poloidal twists).
    gamma : float
        The scaling factor for the imaginary component.
    k_symbols : kSymbols, optional
        Tuple of k-space symbols (kx, ky, kz). Defaults to real symbols.
    c : float, optional
        Offset added to ``n3``. Defaults to 0.5 (legacy default).
    m : float, optional
        Mass-like shift in ``n4``. Defaults to 2.0.

    Returns
    -------
    tuple[sp.Expr, sp.Expr, sp.Expr]
        Bloch vector components.
    """
    z, w = _k2zw(k_symbols, c, m, mirror_n3=True)
    return _zw2d(z**p - w**q, gamma)

def unknot_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) unknot ansatz."""
    return pq_torus_knot_mirror_bloch_vector(1, 1, gamma, k_symbols)

def hopf_link_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) Hopf-link ansatz."""
    return pq_torus_knot_mirror_bloch_vector(2, 2, gamma, k_symbols)

def trefoil_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) trefoil ansatz."""
    return pq_torus_knot_mirror_bloch_vector(2, 3, gamma, k_symbols)

def cinquefoil_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) cinquefoil ansatz (p=2, q=5)."""
    return pq_torus_knot_mirror_bloch_vector(2, 5, gamma, k_symbols)

def solomon_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
    c: float = 0.333,
    m: float = 2.0,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) Solomon-link ansatz."""
    return pq_torus_knot_mirror_bloch_vector(2, 4, gamma, k_symbols, c, m)

def threelink_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
    c: float = 0.333,
    m: float = 2.0,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) three-link ansatz."""
    z, w = _k2zw(k_symbols, c, m, mirror_n3=True)
    return _zw2d((z**2 - w**2) * z, gamma)

def awesome_mirror_bloch_vector(
    gamma: sp.Symbol,
    k_symbols: kSymbols = sp.symbols('k_x k_y k_z', real=True),
    c: float = 0.333,
    m: float = 2.0,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Mirror-symmetric (legacy) awesome-knot ansatz."""
    z, w = _k2zw(k_symbols, c, m, mirror_n3=True)
    return _zw2d(z * (z**2 - w**4 + w), gamma)