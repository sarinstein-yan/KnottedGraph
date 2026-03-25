from __future__ import annotations

import warnings

import networkx as nx
import sympy as sp


__all__ = [
    "normalize_multigraph",
    "multigraph_key",
    "connected_components_ignoring_loops",
    "pick_nonloop_edge",
    "delete_multigraph_edge",
    "contract_multigraph_edge",
    "compute_yamada_polynomial_recursive",
    "laurent_y_to_sigma_polynomial",
]


def normalize_multigraph(G: nx.MultiGraph) -> nx.MultiGraph:
    """
    Relabel nodes in a deterministic order to obtain a canonical 0..n-1
    multigraph.

    The ordering uses ``repr(node)`` so mixed label types such as integers,
    strings, and tuples can all be handled uniformly.
    """
    nodes = sorted(G.nodes(), key=repr)
    mapping = {node: i for i, node in enumerate(nodes)}

    H = nx.MultiGraph()
    H.add_nodes_from(range(len(nodes)))
    for u, v, key in G.edges(keys=True):
        H.add_edge(mapping[u], mapping[v])

    return H


def multigraph_key(G: nx.MultiGraph):
    """
    Return a memoization key based on the canonicalized edge multiset.
    """
    G = normalize_multigraph(G)

    edges = []
    for u, v, key in G.edges(keys=True):
        a, b = (u, v) if u <= v else (v, u)
        edges.append((a, b))
    edges.sort()

    return (G.number_of_nodes(), tuple(edges))


def connected_components_ignoring_loops(G: nx.MultiGraph):
    """
    Connected components of the underlying simple graph, ignoring loops.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if u != v:
            H.add_edge(u, v)
    return list(nx.connected_components(H))


def pick_nonloop_edge(G: nx.MultiGraph):
    """
    Pick one non-loop edge occurrence (u, v, key), or return None.
    """
    for u, v, key in G.edges(keys=True):
        if u != v:
            return (u, v, key)
    return None


def delete_multigraph_edge(G: nx.MultiGraph, edge) -> nx.MultiGraph:
    """
    Delete one chosen edge occurrence from a multigraph.
    """
    u, v, key = edge
    H = G.copy()
    H.remove_edge(u, v, key)
    return H


def contract_multigraph_edge(G: nx.MultiGraph, edge) -> nx.MultiGraph:
    """
    Contract one chosen non-loop edge occurrence by merging v into u.

    Multiplicity is preserved, and parallel u-v edges become loops at u.
    """
    u, v, key = edge
    if u == v:
        raise ValueError("Loop contraction is not allowed.")

    H = G.copy()
    H.remove_edge(u, v, key)

    incident = list(H.edges(v, keys=True, data=True))
    H.remove_node(v)

    for a, b, incident_key, data in incident:
        other = b if a == v else a
        new_u = u
        new_v = u if other == v else other
        H.add_edge(new_u, new_v, **data)

    return H


def _laurent_polynomial_data(expr: sp.Expr, variable: sp.Symbol):
    expr = sp.expand(expr)
    if expr == 0:
        return sp.Poly(0, variable), 0, 0, 0

    terms = sp.Add.make_args(expr)
    exponents = [int(term.as_powers_dict().get(variable, 0)) for term in terms]
    min_exponent = min(exponents)
    max_exponent = max(exponents)
    shift = -min_exponent

    shifted_poly = sp.Poly(sp.expand(expr * variable**shift), variable)
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


def compute_yamada_polynomial_recursive(G: nx.MultiGraph, variable: sp.Symbol) -> sp.Expr:
    """
    Compute the Yamada polynomial of a crossing-free abstract multigraph by
    direct deletion-contraction recursion.

    Conventions:
    - empty graph -> 1
    - disjoint union -> product over components
    - bouquet with n loops -> -(-sigma)^n, where sigma = A + 1 + A^-1
    - otherwise R(G) = R(G - e) + R(G / e) for a non-loop edge e
    """
    sigma = variable + 1 + variable**(-1)
    memo = {}

    def rec(H: nx.MultiGraph):
        H = normalize_multigraph(H)
        key = multigraph_key(H)
        if key in memo:
            return memo[key]

        n_vertices = H.number_of_nodes()
        n_edges = H.number_of_edges()

        if n_vertices == 0 and n_edges == 0:
            memo[key] = sp.Integer(1)
            return memo[key]

        components = connected_components_ignoring_loops(H)
        if len(components) > 1:
            value = sp.Integer(1)
            for component in components:
                value *= rec(H.subgraph(component).copy())
            memo[key] = sp.simplify(value)
            return memo[key]

        edge = pick_nonloop_edge(H)
        if edge is None:
            if n_vertices == 1:
                loops = sum(1 for u, v, key in H.edges(keys=True) if u == v == 0)
                value = -((-sigma) ** loops)
            else:
                value = sp.Integer(0)
            memo[key] = sp.simplify(value)
            return memo[key]

        value = sp.simplify(
            rec(delete_multigraph_edge(H, edge))
            + rec(contract_multigraph_edge(H, edge))
        )
        memo[key] = value
        return value

    return sp.simplify(rec(G))


def laurent_y_to_sigma_polynomial(
    expr: sp.Expr,
    y_variable: sp.Symbol,
    sigma_variable: sp.Symbol | None = None,
    *,
    verify: bool = True,
    require_inversion_symmetry: bool = True,
) -> sp.Poly:
    """
    Convert a Laurent polynomial in Y that is symmetric under Y <-> Y^-1 into
    a polynomial in sigma = Y + 1 + Y^-1.

    The input expression is expanded internally, so callers do not need to
    apply ``sp.expand`` before passing it in.

    Parameters
    ----------
    expr:
        Laurent polynomial in ``y_variable``.
    y_variable:
        The Laurent variable, typically ``Y``.
    sigma_variable:
        The sigma variable to use in the output polynomial. Defaults to a new
        symbol named ``sigma``.
    verify:
        When True, substitute sigma = Y + 1 + Y^-1 back into the result and
        verify that the original Laurent polynomial is recovered.
    require_inversion_symmetry:
        When True, raise an error if the coefficients of Y^k and Y^-k differ.
        When False, continue and emit a warning.
    """
    sigma_variable = sp.Symbol("sigma") if sigma_variable is None else sigma_variable
    aux_variable = sp.Symbol("t")
    expr = sp.expand(expr)

    shifted_poly, min_exponent, max_exponent, shift = _laurent_polynomial_data(expr, y_variable)
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

    # Write P(Y) = c0 + sum_{k>=1} c_k (Y^k + Y^-k) using
    # S_0 = 2, S_1 = t, S_{k+1} = t S_k - S_{k-1}, where t = Y + Y^-1.
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
            s_k = sp.Poly(sp.expand(aux_variable * s_prev1.as_expr() - s_prev2.as_expr()), aux_variable)
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
            p_sigma_poly.as_expr().subs(sigma_variable, y_variable + 1 + y_variable**(-1))
        )
        difference = sp.expand(back_substituted - expr)
        if not _laurent_is_zero(difference, y_variable):
            raise ValueError("Verification failed: the sigma polynomial does not recover the input.")

    return p_sigma_poly
