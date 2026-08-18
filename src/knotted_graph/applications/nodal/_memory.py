"""Memory-efficient internals for :class:`NodalSkeleton`.

The public class lives in ``skeleton.py``.  Keeping these implementation
helpers separate lets us benchmark the memory backend against the retained
stacked/reference calculations without changing the public API.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Sequence

import numpy as np
import sympy as sp

from knotted_graph.applications.nodal.symmetry import is_PT_symmetric


class _LazyGrid:
    """Materialize one legacy dense coordinate grid only on first access.

    The materialized value is a normal writable C-contiguous ndarray, matching
    ``np.meshgrid(..., indexing='ij')`` rather than exposing a read-only
    broadcast view.  Assignment/deletion are supported to preserve the old
    ordinary-instance-attribute behavior.
    """

    def __init__(self, axis: int, name: str):
        self.axis = axis
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.name)
        if value is None:
            shape = (instance.dimension,) * 3
            axis_values = (
                instance.kx_vals,
                instance.ky_vals,
                instance.kz_vals,
            )[self.axis]
            reshape = [1, 1, 1]
            reshape[self.axis] = instance.dimension
            value = np.broadcast_to(
                axis_values.reshape(reshape), shape
            ).copy()
            instance.__dict__[self.name] = value
        return value

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        instance.__dict__.pop(self.name, None)


def _optimized_init(
    self,
    char,
    k_symbols=None,
    span=((-np.pi, np.pi), (-np.pi, np.pi), (0, np.pi)),
    dimension: int = 200,
    axis_scale=(1.0, 1.0, 2.0),
):
    """Original constructor minus eager ``np.meshgrid`` materialization."""
    if isinstance(char, (sp.Matrix, sp.ImmutableMatrix)) and char.shape == (2, 2):
        self.h_k = char
        self.bloch_vec = tuple(
            sp.simplify((char * s).trace() / 2)
            for s in self.pauli_vec
        )
    elif isinstance(char, Sequence) and len(char) == 3:
        self.bloch_vec = tuple(c + sp.Integer(0) for c in char)
        self.h_k = sum(
            (h * s for h, s in zip(char, self.pauli_vec)),
            start=sp.zeros(2, 2),
        )
    else:
        raise ValueError(
            "`char` must be a 2x2 sympy Matrix or a sequence "
            "of three coefficients for the Pauli matrices."
        )

    if k_symbols is None:
        self.k_symbols = sorted(self.h_k.free_symbols, key=lambda s: s.name)
        self.kx_symbol, self.ky_symbol, self.kz_symbol = self.k_symbols
    elif len(k_symbols) == 3:
        self.k_symbols = k_symbols
        self.kx_symbol, self.ky_symbol, self.kz_symbol = k_symbols
    else:
        raise ValueError(
            "`k_symbols` must be a tuple of three sympy symbols (kx, ky, kz)."
        )

    self.is_Hermitian = sp.simplify(self.h_k - self.h_k.H) == sp.zeros(2, 2)
    self.is_PT_symmetric = is_PT_symmetric(self.h_k)

    self.bloch_vec_funcs = tuple(
        sp.lambdify(self.k_symbols, b, "numpy") for b in self.bloch_vec
    )

    self.span = np.asarray(span)
    self.dimension = dimension
    self.spacing = np.diff(self.span, axis=1).squeeze() / (dimension - 1)
    self.axis_scale = np.asarray(axis_scale, dtype=float)
    self.origin = self.span[:, 0]

    self.kx_span, self.ky_span, self.kz_span = span
    for axis, (mn, mx) in zip(("x", "y", "z"), span):
        setattr(self, f"k{axis}_min", mn)
        setattr(self, f"k{axis}_max", mx)
        setattr(self, f"k{axis}_vals", np.linspace(mn, mx, dimension))

    # kx_grid / ky_grid / kz_grid are descriptors installed below.  They are
    # intentionally not touched here, avoiding 3 * dimension**3 float arrays.

    self.skeleton_graph_cache = None
    self.skeleton_graph_cache_args = None
    self._pv_data_args = None


def _evaluated_component(self, expr: sp.Expr, func):
    grids = (
        self.kx_vals[:, None, None],
        self.ky_vals[None, :, None],
        self.kz_vals[None, None, :],
    )
    if expr.free_symbols:
        return np.asarray(func(*grids), dtype=np.complex128)
    return np.asarray(complex(expr), dtype=np.complex128)


def _streamed_spectrum(self):
    """Compute sqrt(sum(d_i**2)) without materializing the 3-component stack."""
    shape = (self.dimension,) * 3
    total = np.empty(shape, dtype=np.complex128)

    for index, (expr, func) in enumerate(
        zip(self.bloch_vec, self.bloch_vec_funcs)
    ):
        component = _evaluated_component(self, expr, func)
        # Reuse the component buffer for its square whenever possible.  Lambdify
        # results converted to complex128 are ordinary writable arrays; scalar
        # and lower-dimensional results remain cheap and broadcast naturally.
        if component.flags.writeable:
            np.multiply(component, component, out=component)
            squared = component
        else:  # defensive fallback for unusual lambdify results
            squared = np.multiply(component, component)

        if index == 0:
            np.copyto(total, squared)
        else:
            np.add(total, squared, out=total)

    np.sqrt(total, out=total)
    return total


def _skeleton_coords_without_dense_grids(self):
    indices = np.where(self._skeleton_image)
    return np.asarray(
        [
            self.kx_vals[indices[0]],
            self.ky_vals[indices[1]],
            self.kz_vals[indices[2]],
        ]
    ).T


def _install_cached_property(cls, name: str, func) -> None:
    descriptor = cached_property(func)
    descriptor.__set_name__(cls, name)
    setattr(cls, name, descriptor)


def install_memory_optimizations(cls) -> None:
    """Install exact-compatible lazy-grid and streamed-spectrum internals."""
    cls.__init__ = _optimized_init
    cls.kx_grid = _LazyGrid(0, "kx_grid")
    cls.ky_grid = _LazyGrid(1, "ky_grid")
    cls.kz_grid = _LazyGrid(2, "kz_grid")
    _install_cached_property(cls, "spectrum", _streamed_spectrum)
    _install_cached_property(
        cls,
        "skeleton_coords",
        _skeleton_coords_without_dense_grids,
    )
