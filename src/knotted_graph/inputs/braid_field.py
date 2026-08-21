"""Construct semiholomorphic knot fields from Artin braid words.

The implementation follows the constructive strategy of Bode--Dennis:
represent the braid as the roots of a monic polynomial ``g(u, t)``, approximate
its periodic coefficient functions by finite Fourier series, and replace
``exp(i t)`` and ``exp(-i t)`` by ``v`` and ``conjugate(v)`` respectively.

The returned polynomial is exact for the retained Fourier coefficients.  The
``BraidValidationReport`` is a *numerical* certificate that the retained
trigonometric polynomial resolves the requested geometric braid on the sampled
parameter values; it is deliberately not described as a formal proof that a
particular finite scaling threshold on S^3 has been crossed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

Term = tuple[int, int, int]


def _clean_word(word: Iterable[int]) -> tuple[int, ...]:
    cleaned = tuple(int(generator) for generator in word)
    if any(generator == 0 for generator in cleaned):
        raise ValueError("Artin braid generators are non-zero integers.")
    return cleaned


def infer_braid_strands(word: Iterable[int], strands: int | None = None) -> int:
    """Return a valid strand count for an Artin braid word."""
    word = _clean_word(word)
    minimum = max((abs(generator) + 1 for generator in word), default=1)
    if strands is None:
        return minimum
    strands = int(strands)
    if strands < 1:
        raise ValueError("strands must be positive")
    if strands < minimum:
        raise ValueError(
            f"braid word requires at least {minimum} strands, got {strands}"
        )
    return strands


def braid_permutation(word: Iterable[int], strands: int | None = None) -> tuple[int, ...]:
    """Return the endpoint permutation induced by an Artin braid word."""
    word = _clean_word(word)
    strands = infer_braid_strands(word, strands)
    slots = list(range(strands))
    for generator in word:
        index = abs(generator) - 1
        slots[index], slots[index + 1] = slots[index + 1], slots[index]
    permutation = [0] * strands
    for endpoint_slot, start_label in enumerate(slots):
        permutation[start_label] = endpoint_slot
    return tuple(permutation)


def braid_component_count(word: Iterable[int], strands: int | None = None) -> int:
    """Return the number of components in the closure of the braid word."""
    permutation = braid_permutation(word, strands)
    seen: set[int] = set()
    cycles = 0
    for seed in range(len(permutation)):
        if seed in seen:
            continue
        cycles += 1
        current = seed
        while current not in seen:
            seen.add(current)
            current = permutation[current]
    return cycles


@dataclass(frozen=True)
class SemiholomorphicPolynomial:
    """A polynomial in ``u``, ``v`` and ``conjugate(v)``."""

    terms: Mapping[Term, complex]

    def __post_init__(self) -> None:
        cleaned: dict[Term, complex] = {}
        for powers, coefficient in self.terms.items():
            if len(powers) != 3:
                raise ValueError("term powers must be (u_power, v_power, vbar_power)")
            powers = tuple(int(power) for power in powers)
            if any(power < 0 for power in powers):
                raise ValueError("semiholomorphic exponents must be non-negative")
            value = complex(coefficient)
            if value != 0:
                cleaned[powers] = cleaned.get(powers, 0j) + value
        object.__setattr__(self, "terms", cleaned)

    @property
    def degree_u(self) -> int:
        return max((powers[0] for powers in self.terms), default=0)

    @property
    def total_degree(self) -> int:
        return max((sum(powers) for powers in self.terms), default=0)

    def evaluate(self, u, v):
        """Evaluate on scalar or NumPy-array arguments."""
        u = np.asarray(u, dtype=np.complex128)
        v = np.asarray(v, dtype=np.complex128)
        result = np.zeros(np.broadcast_shapes(u.shape, v.shape), dtype=np.complex128)
        vbar = np.conjugate(v)
        for (u_power, v_power, vbar_power), coefficient in self.terms.items():
            result = result + coefficient * (u**u_power) * (v**v_power) * (
                vbar**vbar_power
            )
        return result

    def to_sympy(
        self,
        u: sp.Symbol | None = None,
        v: sp.Symbol | None = None,
        vbar: sp.Symbol | None = None,
        *,
        chop: float = 1e-14,
    ) -> sp.Expr:
        """Return a SymPy expression with independent ``v``/``vbar`` symbols."""
        u = u or sp.Symbol("u")
        v = v or sp.Symbol("v")
        vbar = vbar or sp.Symbol("vbar")
        expression = sp.Integer(0)
        for powers, coefficient in sorted(self.terms.items()):
            real = 0.0 if abs(coefficient.real) < chop else coefficient.real
            imag = 0.0 if abs(coefficient.imag) < chop else coefficient.imag
            sym_coefficient = sp.Float(real, 16) + sp.I * sp.Float(imag, 16)
            expression += (
                sym_coefficient
                * u ** powers[0]
                * v ** powers[1]
                * vbar ** powers[2]
            )
        return sp.expand(expression)

    def scaled(self, scalar: complex) -> "SemiholomorphicPolynomial":
        return SemiholomorphicPolynomial(
            {powers: complex(scalar) * value for powers, value in self.terms.items()}
        )

    def add_scaled(
        self,
        other: "SemiholomorphicPolynomial",
        self_scale: complex = 1.0,
        other_scale: complex = 1.0,
    ) -> "SemiholomorphicPolynomial":
        terms: dict[Term, complex] = {}
        for powers, value in self.terms.items():
            terms[powers] = terms.get(powers, 0j) + self_scale * value
        for powers, value in other.terms.items():
            terms[powers] = terms.get(powers, 0j) + other_scale * value
        return SemiholomorphicPolynomial(terms)


@dataclass(frozen=True)
class BraidValidationReport:
    """Numerical diagnostics for a braid-to-polynomial Fourier compilation."""

    word: tuple[int, ...]
    strands: int
    permutation: tuple[int, ...]
    components: int
    root_scale: float
    root_center: complex
    fourier_mode: int
    fit_samples: int
    validation_samples: int
    max_root_error: float
    min_target_separation: float
    error_fraction: float
    passed: bool

    @property
    def interpretation(self) -> str:
        return (
            "numerical sampled braid-isotopy diagnostic; not a formal proof "
            "certificate for the S^3 scaling threshold"
        )


def _smoothstep(tau: np.ndarray) -> np.ndarray:
    return tau * tau * (3.0 - 2.0 * tau)


def geometric_braid_roots(
    word: Sequence[int],
    t,
    *,
    strands: int | None = None,
    root_scale: float = 0.25,
    crossing_height_fraction: float = 0.48,
    root_center: complex = 0.37,
) -> np.ndarray:
    """Return roots tracing a geometric representative of an Artin braid."""
    word = _clean_word(word)
    strands = infer_braid_strands(word, strands)
    if root_scale <= 0:
        raise ValueError("root_scale must be positive")
    if not (0 < crossing_height_fraction < 1):
        raise ValueError("crossing_height_fraction must lie in (0, 1)")

    t_array = np.asarray(t, dtype=float)
    flat = np.mod(t_array.ravel(), 2 * np.pi)
    if strands == 1:
        roots = np.full(
            (flat.size, 1), root_scale * complex(root_center), dtype=np.complex128
        )
        return roots.reshape(t_array.shape + (1,))

    slots = np.linspace(-1.0, 1.0, strands) + complex(root_center)
    roots = np.broadcast_to(root_scale * slots, (flat.size, strands)).astype(
        np.complex128, copy=True
    )
    if not word:
        return roots.reshape(t_array.shape + (strands,))

    scaled = flat * (len(word) / (2 * np.pi))
    interval = np.floor(scaled).astype(int)
    interval = np.minimum(interval, len(word) - 1)
    tau = scaled - interval
    progress = _smoothstep(tau)
    bump = np.sin(np.pi * tau) ** 2

    spacing = 2.0 / (strands - 1)
    height = crossing_height_fraction * spacing

    for generator_index, generator in enumerate(word):
        selected = interval == generator_index
        if not np.any(selected):
            continue
        left = abs(generator) - 1
        right = left + 1
        x_left = slots[left]
        x_right = slots[right]
        delta = x_right - x_left
        local_progress = progress[selected]
        local_bump = bump[selected]
        sign = 1.0 if generator > 0 else -1.0
        roots[selected, left] = root_scale * (
            x_left + delta * local_progress + 1j * sign * height * local_bump
        )
        roots[selected, right] = root_scale * (
            x_right - delta * local_progress - 1j * sign * height * local_bump
        )

    return roots.reshape(t_array.shape + (strands,))


def _monic_coefficients_from_roots(roots: np.ndarray) -> np.ndarray:
    roots = np.asarray(roots, dtype=np.complex128)
    result = np.empty((roots.shape[0], roots.shape[1] + 1), dtype=np.complex128)
    for index, row in enumerate(roots):
        result[index] = np.poly(row)
    return result


def _fourier_modes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.fft.fft(values, axis=0) / values.shape[0]
    integer_modes = np.rint(
        np.fft.fftfreq(values.shape[0]) * values.shape[0]
    ).astype(int)
    return integer_modes, coefficients


def _compile_fourier_polynomial(
    coefficient_samples: np.ndarray,
    max_mode: int,
    *,
    coefficient_tolerance: float,
) -> SemiholomorphicPolynomial:
    modes, fourier = _fourier_modes(coefficient_samples)
    u_degree = coefficient_samples.shape[1] - 1
    terms: dict[Term, complex] = {}
    for mode_index, mode in enumerate(modes):
        if abs(mode) > max_mode:
            continue
        for column in range(coefficient_samples.shape[1]):
            coefficient = complex(fourier[mode_index, column])
            if abs(coefficient) <= coefficient_tolerance:
                continue
            u_power = u_degree - column
            if mode >= 0:
                powers = (u_power, int(mode), 0)
            else:
                powers = (u_power, 0, int(-mode))
            terms[powers] = terms.get(powers, 0j) + coefficient
    return SemiholomorphicPolynomial(terms)


def _coefficient_vector_on_unit_circle(
    polynomial: SemiholomorphicPolynomial,
    t: float,
) -> np.ndarray:
    degree = polynomial.degree_u
    v = np.exp(1j * t)
    vbar = np.conjugate(v)
    coefficients = np.zeros(degree + 1, dtype=np.complex128)
    for (u_power, v_power, vbar_power), coefficient in polynomial.terms.items():
        column = degree - u_power
        coefficients[column] += coefficient * (v**v_power) * (vbar**vbar_power)
    if abs(coefficients[0]) == 0:
        raise RuntimeError("compiled braid polynomial lost its leading u coefficient")
    return coefficients / coefficients[0]


def _sampled_root_validation(
    polynomial: SemiholomorphicPolynomial,
    word: tuple[int, ...],
    strands: int,
    *,
    root_scale: float,
    crossing_height_fraction: float,
    root_center: complex,
    validation_samples: int,
) -> tuple[float, float]:
    t_values = 2 * np.pi * (np.arange(validation_samples) + 0.5) / validation_samples
    target = geometric_braid_roots(
        word,
        t_values,
        strands=strands,
        root_scale=root_scale,
        crossing_height_fraction=crossing_height_fraction,
        root_center=root_center,
    )

    max_root_error = 0.0
    min_separation = np.inf
    for t, target_roots in zip(t_values, target):
        if strands > 1:
            distances = np.abs(target_roots[:, None] - target_roots[None, :])
            distances[np.diag_indices_from(distances)] = np.inf
            min_separation = min(min_separation, float(np.min(distances)))
        coefficients = _coefficient_vector_on_unit_circle(polynomial, float(t))
        approx_roots = np.roots(coefficients)
        distance_matrix = np.abs(approx_roots[:, None] - target_roots[None, :])
        symmetric_hausdorff = max(
            float(np.max(np.min(distance_matrix, axis=1))),
            float(np.max(np.min(distance_matrix, axis=0))),
        )
        max_root_error = max(max_root_error, symmetric_hausdorff)

    if strands == 1:
        min_separation = np.inf
    return max_root_error, min_separation


def _default_fit_samples(word_length: int, max_mode: int) -> int:
    target = max(2048, 128 * max(1, word_length), 16 * (2 * max_mode + 1))
    return 1 << ceil(log2(target))


def braid_to_semiholomorphic(
    word: Iterable[int],
    *,
    strands: int | None = None,
    root_scale: float = 0.25,
    crossing_height_fraction: float = 0.48,
    root_center: complex = 0.37,
    fourier_modes: Sequence[int] = (4, 8, 12, 16, 24, 32, 48, 64),
    fit_samples: int | None = None,
    validation_samples: int = 1024,
    max_error_fraction: float = 0.20,
    coefficient_tolerance: float = 1e-12,
) -> tuple[SemiholomorphicPolynomial, BraidValidationReport]:
    """Compile an Artin braid word to a validated semiholomorphic polynomial.

    This validates the finite Fourier approximation on sampled ``t`` values.
    It does not turn the existence theorem's unspecified sufficiently-small S3
    scaling into a formal machine-checkable bound.
    """
    word = _clean_word(word)
    strands = infer_braid_strands(word, strands)
    if validation_samples < 32:
        raise ValueError("validation_samples must be at least 32")
    if not (0 < max_error_fraction < 0.5):
        raise ValueError("max_error_fraction must lie in (0, 0.5)")
    modes = tuple(sorted({int(mode) for mode in fourier_modes if int(mode) >= 0}))
    if not modes:
        raise ValueError("fourier_modes must contain a non-negative mode")

    largest_mode = modes[-1]
    resolved_fit_samples = fit_samples or _default_fit_samples(len(word), largest_mode)
    if resolved_fit_samples <= 2 * largest_mode:
        raise ValueError("fit_samples must exceed twice the largest Fourier mode")

    t_fit = 2 * np.pi * np.arange(resolved_fit_samples) / resolved_fit_samples
    roots = geometric_braid_roots(
        word,
        t_fit,
        strands=strands,
        root_scale=root_scale,
        crossing_height_fraction=crossing_height_fraction,
        root_center=root_center,
    )
    coefficient_samples = _monic_coefficients_from_roots(roots)

    last_polynomial: SemiholomorphicPolynomial | None = None
    last_report: BraidValidationReport | None = None
    for mode in modes:
        polynomial = _compile_fourier_polynomial(
            coefficient_samples,
            mode,
            coefficient_tolerance=coefficient_tolerance,
        )
        max_root_error, min_separation = _sampled_root_validation(
            polynomial,
            word,
            strands,
            root_scale=root_scale,
            crossing_height_fraction=crossing_height_fraction,
            root_center=root_center,
            validation_samples=validation_samples,
        )
        if np.isinf(min_separation):
            error_fraction = 0.0 if max_root_error <= 1e-10 else np.inf
            passed = max_root_error <= max(1e-10, max_error_fraction * root_scale)
        else:
            error_fraction = max_root_error / min_separation
            passed = error_fraction <= max_error_fraction
        report = BraidValidationReport(
            word=word,
            strands=strands,
            permutation=braid_permutation(word, strands),
            components=braid_component_count(word, strands),
            root_scale=float(root_scale),
            root_center=complex(root_center),
            fourier_mode=mode,
            fit_samples=resolved_fit_samples,
            validation_samples=validation_samples,
            max_root_error=float(max_root_error),
            min_target_separation=float(min_separation),
            error_fraction=float(error_fraction),
            passed=bool(passed),
        )
        last_polynomial, last_report = polynomial, report
        if passed:
            return polynomial, report

    assert last_polynomial is not None and last_report is not None
    raise RuntimeError(
        "braid Fourier approximation did not meet the requested validation "
        f"threshold; final error fraction={last_report.error_fraction:.4g}, "
        f"max mode={last_report.fourier_mode}. Increase fourier_modes/fit_samples "
        "or adjust the geometric braid parameters."
    )


__all__ = [
    "BraidValidationReport",
    "SemiholomorphicPolynomial",
    "braid_component_count",
    "braid_permutation",
    "braid_to_semiholomorphic",
    "geometric_braid_roots",
    "infer_braid_strands",
]
