"""Exact algebraic utilities for Yamada Laurent polynomials."""

from __future__ import annotations

import warnings

import sympy as sp


def _laurent_polynomial_data(expr: sp.Expr, variable: sp.Symbol):
    expr = sp.expand(sp.cancel(expr))
    if expr == 0:
        return sp.Poly(0, variable), 0, 0, 0
    terms = sp.Add.make_args(expr)
    exponents = [int(term.as_powers_dict().get(variable, 0)) for term in terms]
    min_exponent = min(exponents)
    max_exponent = max(exponents)
    shift = -min_exponent
    shifted = sp.cancel(expr * variable**shift)
    numerator, denominator = sp.fraction(shifted)
    if variable in denominator.free_symbols:
        raise ValueError(
            f"Expression is not a Laurent polynomial in {variable}: "
            f"uncancelled denominator {denominator}."
        )
    shifted_poly = sp.Poly(sp.expand(numerator / denominator), variable)
    return shifted_poly, min_exponent, max_exponent, shift


def _laurent_coefficient(
    shifted_poly: sp.Poly,
    exponent: int,
    min_exponent: int,
    max_exponent: int,
    shift: int,
    variable: sp.Symbol,
):
    if exponent < min_exponent or exponent > max_exponent:
        return sp.Integer(0)
    return shifted_poly.coeff_monomial(variable ** (exponent + shift))


def _laurent_is_zero(expr: sp.Expr, variable: sp.Symbol) -> bool:
    expr = sp.expand(expr)
    if expr == 0:
        return True
    shifted_poly, _, _, _ = _laurent_polynomial_data(expr, variable)
    return shifted_poly.is_zero


def laurent_y_to_sigma_polynomial(
    expr: sp.Expr,
    y_variable: sp.Symbol,
    sigma_variable: sp.Symbol | None = None,
    *,
    verify: bool = True,
    require_inversion_symmetry: bool = True,
) -> sp.Poly:
    """Convert an inversion-symmetric Laurent polynomial to sigma form."""
    sigma_variable = sp.Symbol("sigma") if sigma_variable is None else sigma_variable
    aux_variable = sp.Symbol("t")
    expr = sp.expand(sp.cancel(expr))
    shifted_poly, min_exponent, max_exponent, shift = _laurent_polynomial_data(
        expr, y_variable
    )
    max_abs_exponent = max(abs(min_exponent), abs(max_exponent))

    def coeff(exponent: int):
        return _laurent_coefficient(
            shifted_poly,
            exponent,
            min_exponent,
            max_exponent,
            shift,
            y_variable,
        )

    p_t = sp.Poly(coeff(0), aux_variable)
    if max_abs_exponent >= 1:
        s_prev2 = sp.Poly(2, aux_variable)
        s_prev1 = sp.Poly(aux_variable, aux_variable)
        c_pos = coeff(1)
        c_neg = coeff(-1)
        if c_pos != c_neg:
            message = f"Asymmetry at k=1: coeff(+1)={c_pos}, coeff(-1)={c_neg}"
            if require_inversion_symmetry:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        p_t += c_pos * s_prev1
        for k in range(2, max_abs_exponent + 1):
            s_k = sp.Poly(
                sp.expand(aux_variable * s_prev1.as_expr() - s_prev2.as_expr()),
                aux_variable,
            )
            c_pos = coeff(k)
            c_neg = coeff(-k)
            if c_pos != c_neg:
                message = f"Asymmetry at k={k}: coeff(+{k})={c_pos}, coeff(-{k})={c_neg}"
                if require_inversion_symmetry:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
            p_t += c_pos * s_k
            s_prev2, s_prev1 = s_prev1, s_k

    p_sigma_expr = sp.expand(p_t.as_expr().subs(aux_variable, sigma_variable - 1))
    p_sigma_poly = sp.Poly(p_sigma_expr, sigma_variable)
    if verify:
        back_substituted = sp.expand(
            p_sigma_poly.as_expr().subs(
                sigma_variable, y_variable + 1 + y_variable**(-1)
            )
        )
        difference = sp.expand(back_substituted - expr)
        if not _laurent_is_zero(difference, y_variable):
            raise ValueError(
                "Verification failed: the sigma polynomial does not recover the input."
            )
    return p_sigma_poly
