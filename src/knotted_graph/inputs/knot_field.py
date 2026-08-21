"""Analytic complex fields whose zero sets represent knots and links."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Callable, Sequence

import numpy as np
import sympy as sp

from .braid_field import BraidValidationReport, SemiholomorphicPolynomial, braid_to_semiholomorphic
from .knot_catalogue import get_knot_entry

Span3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
DEFAULT_SPAN: Span3D = ((-4.0, 4.0),) * 3


def inverse_stereographic_s3(x, y, z):
    r"""Map R3 to the unit S3 in C2 using the package chart."""
    x, y, z = (np.asarray(value, dtype=float) for value in (x, y, z))
    r2 = x * x + y * y + z * z
    denominator = 1.0 + r2
    return 2.0 * (x + 1j * y) / denominator, (r2 - 1.0 + 2j * z) / denominator


def sample_s3(count: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 4))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points[:, 0] + 1j * points[:, 1], points[:, 2] + 1j * points[:, 3]


@dataclass
class KnotFunction:
    r"""Complex field F:R3->C, optionally induced by a field on S3."""

    evaluator: Callable
    name: str = "custom"
    semiholomorphic: SemiholomorphicPolynomial | None = None
    s3_evaluator: Callable | None = None
    expected_components: int | None = None
    construction_report: BraidValidationReport | None = None
    s3_chart_angle: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __call__(self, x, y, z):
        result = np.asarray(self.evaluator(x, y, z), dtype=np.complex128)
        return np.broadcast_to(result, np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z)))

    @classmethod
    def from_function(
        cls, function, *, symbols: Sequence[sp.Symbol] | None = None,
        name: str = "custom", expected_components: int | None = None,
        metadata: dict | None = None,
    ) -> "KnotFunction":
        if isinstance(function, sp.Expr):
            if symbols is None:
                symbols = tuple(sorted(function.free_symbols, key=lambda s: s.name))
            if len(symbols) != 3:
                raise ValueError("a SymPy knot field requires exactly three coordinates")
            compiled = sp.lambdify(tuple(symbols), function, "numpy")
            meta = {**dict(metadata or {}), "sympy_expression": function}
            return cls(compiled, name, expected_components=expected_components, metadata=meta)
        if not callable(function):
            raise TypeError("function must be callable or a SymPy expression")
        return cls(function, name, expected_components=expected_components, metadata=dict(metadata or {}))

    @classmethod
    def from_semiholomorphic(
        cls, polynomial: SemiholomorphicPolynomial, *, name: str = "semiholomorphic",
        expected_components: int | None = None,
        construction_report: BraidValidationReport | None = None,
        metadata: dict | None = None, chart_angle: float = 0.0,
    ) -> "KnotFunction":
        angle = float(chart_angle)
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))

        def s3_evaluator(u, v):
            transformed_u = cosine * u + sine * v
            transformed_v = -sine * u + cosine * v
            return polynomial.evaluate(transformed_u, transformed_v)

        def evaluator(x, y, z):
            return s3_evaluator(*inverse_stereographic_s3(x, y, z))

        return cls(
            evaluator=evaluator, name=name, semiholomorphic=polynomial,
            s3_evaluator=s3_evaluator, expected_components=expected_components,
            construction_report=construction_report, s3_chart_angle=angle,
            metadata={**dict(metadata or {}), "s3_chart_angle": angle},
        )

    @classmethod
    def torus(cls, p: int, q: int, *, name: str | None = None) -> "KnotFunction":
        p, q = int(p), int(q)
        if min(p, q) < 1:
            raise ValueError("p and q must be positive")
        polynomial = SemiholomorphicPolynomial({(p, 0, 0): 1.0, (0, q, 0): -1.0})
        return cls.from_semiholomorphic(
            polynomial, name=name or f"T({p},{q})", expected_components=gcd(p, q),
            metadata={"construction": "torus", "p": p, "q": q},
        )

    @classmethod
    def from_braid(
        cls, word, *, strands: int | None = None, name: str = "braid_closure",
        chart_angle: float = 0.72, **compiler_options,
    ) -> "KnotFunction":
        polynomial, report = braid_to_semiholomorphic(word, strands=strands, **compiler_options)
        return cls.from_semiholomorphic(
            polynomial, name=name, expected_components=report.components,
            construction_report=report, chart_angle=chart_angle,
            metadata={
                "construction": "braid_fourier_semiholomorphic",
                "braid_word": report.word, "strands": report.strands,
            },
        )

    @classmethod
    def figure_eight_reference(cls) -> "KnotFunction":
        r"""Return Rudolph's published semiholomorphic figure-eight field."""
        polynomial = SemiholomorphicPolynomial({
            (3, 0, 0): 1.0,
            (1, 2, 2): -3.0,
            (1, 4, 2): -3.0,
            (1, 2, 4): 3.0,
            (0, 2, 0): -2.0,
            (0, 0, 2): -2.0,
        })
        return cls.from_semiholomorphic(
            polynomial, name="4_1", expected_components=1,
            metadata={
                "construction": "published_reference_semiholomorphic",
                "reference": "Rudolph figure-eight polynomial",
            },
        )

    @classmethod
    def from_name(
        cls, name: str, *, construction: str = "preferred", **compiler_options,
    ) -> "KnotFunction":
        entry = get_knot_entry(name)
        if construction not in {"preferred", "braid"}:
            raise ValueError("construction must be 'preferred' or 'braid'")
        if construction == "braid":
            return cls.from_braid(
                entry.braid_word, strands=entry.strands,
                name=entry.canonical_name, **compiler_options,
            )
        if entry.torus_params is not None:
            result = cls.torus(*entry.torus_params, name=entry.canonical_name)
        elif entry.reference_field == "rudolph_figure_eight":
            result = cls.figure_eight_reference()
        else:
            result = cls.from_braid(
                entry.braid_word, strands=entry.strands,
                name=entry.canonical_name, **compiler_options,
            )
        result.metadata.update(
            catalogue_name=entry.canonical_name,
            standard_braid_word=entry.braid_word,
            strands=entry.strands,
        )
        if result.expected_components != entry.components:
            raise RuntimeError("catalogue component-count mismatch")
        return result

    def evaluate_s3(self, u, v):
        if self.s3_evaluator is None:
            raise ValueError(f"{self.name!r} has no S3 evaluator")
        result = np.asarray(self.s3_evaluator(u, v), dtype=np.complex128)
        return np.broadcast_to(result, np.broadcast_shapes(np.shape(u), np.shape(v)))

    @property
    def projection_pole_value(self) -> complex | None:
        if self.s3_evaluator is None:
            return None
        return complex(np.asarray(self.evaluate_s3(0.0, 1.0)).item())

    def symbolic_r3_expression(self, symbols: Sequence[sp.Symbol] | None = None) -> sp.Expr:
        if self.semiholomorphic is None:
            expression = self.metadata.get("sympy_expression")
            if expression is None:
                raise ValueError("symbolic expression is unavailable for this field")
            return expression
        x, y, z = symbols or sp.symbols("x y z", real=True)
        r2, denominator = x**2 + y**2 + z**2, 1 + x**2 + y**2 + z**2
        u, ubar = 2 * (x + sp.I * y) / denominator, 2 * (x - sp.I * y) / denominator
        v = (r2 - 1 + 2 * sp.I * z) / denominator
        vbar = (r2 - 1 - 2 * sp.I * z) / denominator
        cosine, sine = sp.Float(np.cos(self.s3_chart_angle)), sp.Float(np.sin(self.s3_chart_angle))
        us, vs, vbs = sp.symbols("u v vbar")
        expression = self.semiholomorphic.to_sympy(us, vs, vbs)
        return sp.cancel(expression.subs({
            us: cosine * u + sine * v,
            vs: -sine * u + cosine * v,
            vbs: -sine * ubar + cosine * vbar,
        }))

    def sample(self, *, span: Span3D = DEFAULT_SPAN, dimension=96):
        from .knot_levelset import sample_field
        return sample_field(self, span=span, dimension=dimension)

    def sublevel_mask(self, radius: float, *, sample=None, span: Span3D = DEFAULT_SPAN,
                      dimension=96, require_compact: bool = True):
        from .knot_levelset import sublevel_mask
        return sublevel_mask(
            self, radius, sample=sample, span=span, dimension=dimension,
            require_compact=require_compact,
        )

    def level_surface(self, radius: float, *, sample=None, span: Span3D = DEFAULT_SPAN,
                      dimension=96, require_compact: bool = True):
        from .knot_levelset import level_surface
        return level_surface(
            self, radius, sample=sample, span=span, dimension=dimension,
            require_compact=require_compact,
        )

    def to_spatial_graph(self, radius: float, *, sample=None, span: Span3D = DEFAULT_SPAN,
                         dimension=96, **options):
        from .knot_levelset import to_spatial_graph
        return to_spatial_graph(
            self, radius, sample=sample, span=span, dimension=dimension, **options
        )

    def diagnose_level(self, radius: float, *, sample=None, span: Span3D = DEFAULT_SPAN,
                       dimension=96):
        from .knot_levelset import diagnose_level
        return diagnose_level(
            self, radius, sample=sample, span=span, dimension=dimension
        )

    def tubular_convergence(self, radius: float, *, dimensions=(64, 96, 128),
                            span: Span3D = DEFAULT_SPAN):
        from .knot_levelset import tubular_convergence
        return tubular_convergence(self, radius, dimensions=dimensions, span=span)


__all__ = [
    "DEFAULT_SPAN", "KnotFunction", "Span3D", "inverse_stereographic_s3", "sample_s3",
]
