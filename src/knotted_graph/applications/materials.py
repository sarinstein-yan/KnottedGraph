"""Material Hamiltonian examples for Fermi-surface workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import sympy as sp

__all__ = [
    "D6_PARAMETERS",
    "TI3AL_PARAMETERS",
    "YH3_PARAMETERS",
    "H_D6_sympy",
    "H_Ti3Al_sympy",
    "H_YH3_sympy",
]

kSymbols = tuple[sp.Symbol, sp.Symbol, sp.Symbol]

TI3AL_PARAMETERS = {
    "A1": -9.66,
    "A2": 11.37,
    "B1": 36.22,
    "B2": -25.71,
    "M1": 0.12,
    "M2": -0.52,
    "C": 22.34,
}

D6_PARAMETERS = {
    "F1": 1.787,
    "A1": 2.6,
    "B1": -3.8,
    "F2": -2.12,
    "A2": 1.63,
    "B2": 5.1,
    "L": 1.3,
    "C": 3.55,
    "M": 0.65,
    "F": 1.83,
    "D": 5.1,
}

YH3_PARAMETERS = {
    "m1": 2.99,
    "a1": 2.0,
    "r1": 1.032,
    "s1": 1.032,
    "t1": 1.032,
    "n1": 3,
    "m2": 2.96,
    "m3": 2.96,
    "a2": 4.0,
    "a3": 4.0,
}


def _symbols(k_symbols: Sequence[sp.Symbol] | None) -> kSymbols:
    if k_symbols is None:
        return sp.symbols("k_x k_y k_z", real=True)
    if len(k_symbols) != 3:
        raise ValueError("k_symbols must contain exactly three symbols.")
    return tuple(k_symbols)  # type: ignore[return-value]


def _params(defaults: Mapping[str, float], overrides: Mapping[str, float] | None) -> dict[str, float]:
    params = dict(defaults)
    if overrides:
        params.update(overrides)
    return params


def H_Ti3Al_sympy(
    params: Mapping[str, float] | None = None,
    *,
    k_symbols: Sequence[sp.Symbol] | None = None,
) -> sp.Matrix:
    """Return the two-band Ti3Al material Hamiltonian."""

    kx, ky, kz = _symbols(k_symbols)
    p = _params(TI3AL_PARAMETERS, params)

    k2xy = kx**2 + ky**2
    h1 = p["A1"] * k2xy + p["B1"] * kz**2 + p["M1"]
    h2 = p["A2"] * k2xy + p["B2"] * kz**2 + p["M2"]
    h = 2 * p["C"] * kz
    eps = sp.Rational(1, 2) * (h1 + h2 + sp.sqrt((h1 - h2) ** 2 + h**2))
    return sp.Matrix([[eps, 0], [0, -eps]])


def H_D6_sympy(
    params: Mapping[str, float] | None = None,
    *,
    k_symbols: Sequence[sp.Symbol] | None = None,
) -> sp.Matrix:
    """Return the D6-symmetric three-band material Hamiltonian."""

    kx, ky, kz = _symbols(k_symbols)
    p = _params(D6_PARAMETERS, params)

    kplus = kx + sp.I * ky
    kminus = kx - sp.I * ky
    k2xy = kx**2 + ky**2
    q1 = p["F1"] + p["A1"] * k2xy + p["B1"] * kz**2
    q2 = (
        p["F2"]
        + p["A2"] * k2xy
        + p["B2"] * kz**2
        + p["L"] * k2xy**2
        + p["M"] * (kplus**6 + kminus**6)
    )
    h12 = p["C"] * kminus**2 + p["F"] * kplus**4
    h13 = p["D"] * kminus * kz
    h23 = p["D"] * kplus * kz
    return sp.Matrix(
        [
            [q1, h12, h13],
            [sp.conjugate(h12), q1, h23],
            [sp.conjugate(h13), sp.conjugate(h23), q2],
        ]
    )


def H_YH3_sympy(
    params: Mapping[str, float] | None = None,
    *,
    k_symbols: Sequence[sp.Symbol] | None = None,
) -> sp.Matrix:
    """Return the two-band effective YH3 material Hamiltonian."""

    kx, ky, kz = _symbols(k_symbols)
    p = _params(YH3_PARAMETERS, params)

    g1 = sp.sin(kz)
    g2 = sp.sin(kx)
    g3 = sp.sin(ky)
    h1 = p["a1"] * (
        p["r1"] * sp.cos(kx) ** p["n1"]
        + p["s1"] * sp.cos(ky) ** p["n1"]
        + p["t1"] * sp.cos(kz) ** p["n1"]
        - p["m1"]
    )
    h2 = p["a2"] * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - p["m2"])
    h3 = p["a3"] * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - p["m3"])
    energy = sp.sqrt((g1**2 + h1**2) * (g2**2 + h2**2) * (g3**2 + h3**2))
    return sp.Matrix([[energy, 0], [0, -energy]])
