from __future__ import annotations

import warnings
import itertools
import math

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
    """Relabel nodes deterministically to a canonical 0..n-1 multigraph."""

    nodes = sorted(G.nodes(), key=repr)
    mapping = {node: i for i, node in enumerate(nodes)}

    H = nx.MultiGraph()
    H.add_nodes_from(range(len(nodes)))
    for u, v, key in G.edges(keys=True):
        H.add_edge(mapping[u], mapping[v])

    return H


def multigraph_key(G: nx.MultiGraph):
    """Return a memoization key based on canonical edge multiplicities.

    The key is exact and label-invariant for small/medium graphs. For larger
    highly symmetric graphs, it falls back to a deterministic label-based key to
    avoid factorial blowups; that fallback preserves correctness but may miss
    some memoization reuse.
    """

    nodes = list(G.nodes())
    node_count = len(nodes)
    index = {node: idx for idx, node in enumerate(nodes)}
    matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]

    for u, v, key in G.edges(keys=True):
        i = index[u]
        j = index[v]
        a, b = (i, j) if i <= j else (j, i)
        matrix[a][b] += 1

    def sequence_for(order: tuple[int, ...]) -> tuple[int, ...]:
        sequence = []
        for new_i, old_i in enumerate(order):
            for old_j in order[new_i:]:
                a, b = (old_i, old_j) if old_i <= old_j else (old_j, old_i)
                sequence.append(matrix[a][b])
        return tuple(sequence)

    def node_signature(node_index: int) -> tuple[int, int, tuple[int, ...]]:
        incident = []
        for other in range(node_count):
            a, b = (
                (node_index, other)
                if node_index <= other
                else (other, node_index)
            )
            if other != node_index:
                incident.append(matrix[a][b])
        return (
            matrix[node_index][node_index],
            sum(incident),
            tuple(sorted(incident, reverse=True)),
        )

    signatures: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
    for node_index in range(node_count):
        signatures.setdefault(node_signature(node_index), []).append(node_index)

    ordered_groups = [
        tuple(nodes_in_group)
        for _, nodes_in_group in sorted(signatures.items(), key=lambda item: item[0])
    ]
    grouped_permutation_count = math.prod(
        math.factorial(len(group)) for group in ordered_groups
    )

    if node_count <= 8 or grouped_permutation_count <= 100_000:
        best = None
        for group_permutations in itertools.product(
            *(itertools.permutations(group) for group in ordered_groups)
        ):
            order = tuple(
                node_index
                for group_order in group_permutations
                for node_index in group_order
            )
            candidate = sequence_for(order)
            if best is None or candidate < best:
                best = candidate
        return (node_count, best)

    H = normalize_multigraph(G)
    edges = []
    for u, v, key in H.edges(keys=True):
        a, b = (u, v) if u <= v else (v, u)
        edges.append((a, b))
    edges.sort()
    return (
        node_count,
        "label-fallback",
        tuple(sorted(node_signature(i) for i in range(node_count))),
        tuple(edges),
    )


def connected_components_ignoring_loops(G: nx.MultiGraph):
    """Connected components of the underlying simple graph, ignoring loops."""

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if u != v:
            H.add_edge(u, v)
    return list(nx.connected_components(H))


def pick_nonloop_edge(G: nx.MultiGraph):
    """Pick one non-loop edge occurrence as ``(u, v, key)``, or return None."""

    for u, v, key in G.edges(keys=True):
        if u != v:
            return (u, v, key)
    return None


def delete_multigraph_edge(G: nx.MultiGraph, edge) -> nx.MultiGraph:
    """Delete one chosen edge occurrence from a multigraph."""

    u, v, key = edge
    H = G.copy()
    H.remove_edge(u, v, key)
    return H


def contract_multigraph_edge(G: nx.MultiGraph, edge) -> nx.MultiGraph:
    """Contract one chosen non-loop edge occurrence, preserving multiplicity."""

    u, v, key = edge
    if u == v:
        raise ValueError("Loop contraction is not allowed.")

    H = G.copy()
    H.remove_edge(u, v, key)

    incident = list(H.edges(v, keys=True, data=True))
    H.remove_node(v)

    for a, b, incident_key, data in incident:
        other = b if a == v else a
        new_v = u if other == v else other
        H.add_edge(u, new_v, **data)

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
    """Compute the crossing-free Yamada polynomial by deletion-contraction.

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
    """Convert an inversion-symmetric Laurent polynomial to sigma form."""

    sigma_variable = sp.Symbol("sigma") if sigma_variable is None else sigma_variable
    aux_variable = sp.Symbol("t")
    expr = sp.expand(expr)

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
