"""Exact integer-Laurent kernels for Yamada evaluation.

The public API continues to return SymPy expressions. This module only replaces
hot-loop symbolic algebra with an exact sparse Laurent representation. All
coefficients are Python integers and all operations are algebraically exact.
"""

from __future__ import annotations

import threading
from typing import TypeAlias

import networkx as nx
import sympy as sp

from . import recursive as _r

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
    """Add two exponent-sorted Laurent tuples with a linear merge."""
    if not left:
        return right
    if not right:
        return left

    i = 0
    j = 0
    out: list[tuple[int, int]] = []
    while i < len(left) and j < len(right):
        left_power, left_coeff = left[i]
        right_power, right_coeff = right[j]
        if left_power < right_power:
            out.append((left_power, left_coeff))
            i += 1
        elif right_power < left_power:
            out.append((right_power, right_coeff))
            j += 1
        else:
            coefficient = left_coeff + right_coeff
            if coefficient:
                out.append((left_power, coefficient))
            i += 1
            j += 1

    if i < len(left):
        out.extend(left[i:])
    if j < len(right):
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
    """Multiply by sign*(A^-1 + 1 + A) without generic convolution."""
    if not poly:
        return ZERO

    minimum = poly[0][0] - 1
    maximum = poly[-1][0] + 1
    span = maximum - minimum + 1
    sign = int(sign)

    # Yamada Laurent polynomials are normally dense over a short exponent
    # interval. Dense local accumulation avoids dictionary creation and sorting;
    # retain the sparse dictionary path for unusually gapped polynomials.
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
    """Match polynomial._finalize_yamada_total(normalize=True) exactly."""
    if not poly:
        return ZERO
    lowest = poly[0][0]
    displacement = -lowest
    sign = -1 if displacement % 2 else 1
    return scale(shift(poly, displacement), sign)


def fast_graph_key_normalized(G: nx.MultiGraph):
    """Exact key for an already deterministically relabelled multigraph.

    Unlike ``multigraph_key`` this intentionally does not solve graph
    isomorphism. Equality of these keys still implies equality of the labelled
    multigraph, so correctness is unchanged; the tradeoff is only potentially
    fewer cache hits in exchange for O(E log E) key construction.
    """
    edges = []
    for u, v in G.edges():
        a, b = (u, v) if u <= v else (v, u)
        edges.append((int(a), int(b)))
    edges.sort()
    return (G.number_of_nodes(), tuple(edges))


def fast_graph_key(G: nx.MultiGraph):
    return fast_graph_key_normalized(_r.normalize_multigraph(G))


class _BaseFastEvaluator:
    def __init__(self):
        self.memo: dict[object, Laurent] = {}
        self._memo_lock = threading.RLock()

    def _get(self, key):
        with self._memo_lock:
            return self.memo.get(key)

    def _set(self, key, value: Laurent) -> Laurent:
        with self._memo_lock:
            self.memo[key] = value
        return value

    def _rec(self, G: nx.MultiGraph) -> Laurent:
        """Evaluate one recurrence state; implemented by concrete evaluators."""
        raise NotImplementedError

    def compute_laurent(self, G: nx.MultiGraph) -> Laurent:
        return self._rec(G)

    def compute(self, G: nx.MultiGraph, variable: sp.Symbol) -> sp.Expr:
        return to_sympy(self.compute_laurent(G), variable)


class _NetworkxLaurentYamadaEvaluator(_BaseFastEvaluator):
    """NetworkX recurrence retained to benchmark the Laurent-only optimization."""

    def _rec(self, H: nx.MultiGraph) -> Laurent:
        H = _r.normalize_multigraph(H)
        key = fast_graph_key_normalized(H)
        cached = self._get(key)
        if cached is not None:
            return cached

        n_vertices = H.number_of_nodes()
        n_edges = H.number_of_edges()

        if n_edges == 0:
            return self._set(key, constant((-1) ** n_vertices))

        components = _r.connected_components_ignoring_loops(H)
        if len(components) > 1:
            value = ONE
            for component in components:
                value = multiply(value, self._rec(H.subgraph(component).copy()))
            return self._set(key, value)

        if _r.has_isthmus_multigraph(H):
            return self._set(key, ZERO)

        if _r.is_cycle_multigraph(H):
            return self._set(key, SIGMA)

        theta_s = _r.theta_edge_count(H)
        if theta_s is not None:
            total = ZERO
            power = ONE
            for p in range(1, theta_s):
                power = multiply_sigma(power)
                total = add(total, scale(power, -1 if p % 2 == 0 else 1))
            return self._set(key, total)

        loop = _r._pick_loop_edge(H)
        if loop is not None:
            value = multiply_sigma(
                self._rec(_r.delete_multigraph_edge(H, loop)), sign=-1
            )
            return self._set(key, value)

        parts = _r._split_at_articulation(H)
        if parts is not None:
            value = ONE
            for part in parts:
                value = multiply(value, self._rec(part))
            if (len(parts) - 1) % 2:
                value = scale(value, -1)
            return self._set(key, value)

        edge = _r.pick_nonloop_edge(H)
        if edge is None:
            return self._set(key, constant((-1) ** n_vertices))

        value = add(
            self._rec(_r.delete_multigraph_edge(H, edge)),
            self._rec(_r.contract_multigraph_edge(H, edge)),
        )
        return self._set(key, value)


class _NetworkxLaurentNegamiSpecializedEvaluator(_BaseFastEvaluator):
    """NetworkX specialized-Negami kernel retained for staged benchmarking."""

    def _rec(self, H: nx.MultiGraph) -> Laurent:
        H = _r.normalize_multigraph(H)
        key = fast_graph_key_normalized(H)
        cached = self._get(key)
        if cached is not None:
            return cached

        n_vertices = H.number_of_nodes()
        n_edges = H.number_of_edges()

        if n_edges == 0:
            return self._set(key, constant((-1) ** n_vertices))

        components = _r.connected_components_ignoring_loops(H)
        if len(components) > 1:
            value = ONE
            for component in components:
                value = multiply(value, self._rec(H.subgraph(component).copy()))
            return self._set(key, value)

        if _r.has_isthmus_multigraph(H):
            return self._set(key, ZERO)

        loop = _r._pick_loop_edge(H)
        if loop is not None:
            value = multiply_sigma(
                self._rec(_r.delete_multigraph_edge(H, loop)), sign=-1
            )
            return self._set(key, value)

        parts = _r._split_at_articulation(H)
        if parts is not None:
            value = ONE
            for part in parts:
                value = multiply(value, self._rec(part))
            if (len(parts) - 1) % 2:
                value = scale(value, -1)
            return self._set(key, value)

        edge = _r.pick_nonloop_edge(H)
        if edge is None:
            return self._set(key, constant((-1) ** n_vertices))

        value = add(
            self._rec(_r.contract_multigraph_edge(H, edge)),
            self._rec(_r.delete_multigraph_edge(H, edge)),
        )
        return self._set(key, value)


NetworkxLaurentYamadaEvaluator = _NetworkxLaurentYamadaEvaluator
NetworkxLaurentNegamiSpecializedEvaluator = _NetworkxLaurentNegamiSpecializedEvaluator


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
