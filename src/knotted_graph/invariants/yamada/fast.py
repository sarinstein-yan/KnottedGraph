"""Exact sparse integer-Laurent primitives for optimized Yamada evaluation."""

from __future__ import annotations

from typing import TypeAlias

import sympy as sp

Laurent: TypeAlias = tuple[tuple[int, int], ...]
ZERO: Laurent = ()
ONE: Laurent = ((0, 1),)
SIGMA: Laurent = ((-1, 1), (0, 1), (1, 1))


def _clean(values: dict[int, int]) -> Laurent:
    return tuple(sorted((int(k), int(v)) for k, v in values.items() if v))


def constant(value: int) -> Laurent:
    value = int(value)
    return ZERO if value == 0 else ((0, value),)


def add(left: Laurent, right: Laurent) -> Laurent:
    if not left:
        return right
    if not right:
        return left
    i = j = 0
    out: list[tuple[int, int]] = []
    while i < len(left) and j < len(right):
        lp, lc = left[i]
        rp, rc = right[j]
        if lp < rp:
            out.append((lp, lc))
            i += 1
        elif rp < lp:
            out.append((rp, rc))
            j += 1
        else:
            coefficient = lc + rc
            if coefficient:
                out.append((lp, coefficient))
            i += 1
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return tuple(out)


def scale(poly: Laurent, coefficient: int) -> Laurent:
    coefficient = int(coefficient)
    if coefficient == 0 or not poly:
        return ZERO
    if coefficient == 1:
        return poly
    return tuple((exponent, coefficient * value) for exponent, value in poly)


def shift(poly: Laurent, exponent: int) -> Laurent:
    exponent = int(exponent)
    if not poly or exponent == 0:
        return poly
    return tuple((power + exponent, coefficient) for power, coefficient in poly)


def multiply(left: Laurent, right: Laurent) -> Laurent:
    if not left or not right:
        return ZERO
    if left == ONE:
        return right
    if right == ONE:
        return left
    left_span = left[-1][0] - left[0][0] + 1
    right_span = right[-1][0] - right[0][0] + 1
    if left_span == len(left) and right_span == len(right):
        coefficients = [0] * (left_span + right_span - 1)
        for i, (_, left_coeff) in enumerate(left):
            for j, (_, right_coeff) in enumerate(right):
                coefficients[i + j] += left_coeff * right_coeff
        minimum = left[0][0] + right[0][0]
        return tuple(
            (minimum + index, coefficient)
            for index, coefficient in enumerate(coefficients)
            if coefficient
        )
    out: dict[int, int] = {}
    for p, a in left:
        for q, b in right:
            key = p + q
            out[key] = out.get(key, 0) + a * b
    return _clean(out)


def multiply_sigma(poly: Laurent, sign: int = 1) -> Laurent:
    """Multiply by ``sign * (A^-1 + 1 + A)`` without generic convolution."""
    if not poly:
        return ZERO
    minimum = poly[0][0] - 1
    maximum = poly[-1][0] + 1
    span = maximum - minimum + 1
    sign = int(sign)
    if span <= 4 * len(poly) + 16:
        coefficients = [0] * span
        for exponent, coefficient in poly:
            value = sign * coefficient
            index = exponent - minimum
            coefficients[index - 1] += value
            coefficients[index] += value
            coefficients[index + 1] += value
        return tuple(
            (minimum + index, coefficient)
            for index, coefficient in enumerate(coefficients)
            if coefficient
        )
    out: dict[int, int] = {}
    for exponent, coefficient in poly:
        value = sign * coefficient
        out[exponent - 1] = out.get(exponent - 1, 0) + value
        out[exponent] = out.get(exponent, 0) + value
        out[exponent + 1] = out.get(exponent + 1, 0) + value
    return _clean(out)


def to_sympy(poly: Laurent, variable: sp.Symbol) -> sp.Expr:
    if not poly:
        return sp.Integer(0)
    return sp.Add(*(sp.Integer(c) * variable**p for p, c in poly), evaluate=True)


def normalize_yamada(poly: Laurent) -> Laurent:
    if not poly:
        return ZERO
    lowest = poly[0][0]
    displacement = -lowest
    return scale(shift(poly, displacement), -1 if displacement % 2 else 1)


class FastYamadaEvaluator:
    """Return the fastest validated exact direct-Yamada evaluator."""

    def __new__(cls):
        from .compact import CompactYamadaEvaluator
        return CompactYamadaEvaluator()


class FastNegamiSpecializedEvaluator:
    """Return the fastest validated exact specialized-Negami evaluator."""

    def __new__(cls):
        from .compact import CompactNegamiSpecializedEvaluator
        return CompactNegamiSpecializedEvaluator()
