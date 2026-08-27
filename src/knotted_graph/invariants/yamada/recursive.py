from __future__ import annotations

import itertools
import math
import threading
import warnings

import networkx as nx
import sympy as sp


__all__ = [
    "normalize_multigraph",
    "multigraph_key",
    "connected_components_ignoring_loops",
    "has_isthmus_multigraph",
    "is_cycle_multigraph",
    "theta_edge_count",
    "pick_nonloop_edge",
    "delete_multigraph_edge",
    "contract_multigraph_edge",
    "YamadaRecursiveEvaluator",
    "NegamiRecursiveEvaluator",
    "compute_yamada_polynomial_recursive",
    "compute_negami_recursive",
    "laurent_y_to_sigma_polynomial",
]


def _as_undirected_multigraph(
    graph: nx.Graph,
    *,
    parameter_name: str = "G",
) -> nx.MultiGraph:
    """Copy an undirected NetworkX graph into the evaluator's graph type."""

    graph_types = (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)
    if not isinstance(graph, graph_types):
        raise TypeError(
            f"{parameter_name} must be an undirected networkx.Graph or "
            "networkx.MultiGraph."
        )
    if graph.is_directed():
        raise TypeError(
            f"{parameter_name} must be undirected; directed graphs are not supported."
        )
    return nx.MultiGraph(graph)


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

    if node_count <= 6 or grouped_permutation_count <= 10_000:
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


def _underlying_simple_graph(G: nx.MultiGraph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if u != v:
            H.add_edge(u, v)
    return H


def has_isthmus_multigraph(G: nx.MultiGraph) -> bool:
    """Return whether ``G`` has an isthmus/bridge edge occurrence.

    Parallel edges are handled correctly: if two or more edges join the same
    pair of vertices, none of those parallel occurrences is an isthmus.
    Loops are never isthmuses.
    """

    simple = _underlying_simple_graph(G)
    if simple.number_of_edges() == 0:
        return False

    for u, v in nx.bridges(simple):
        if G.number_of_edges(u, v) == 1:
            return True
    return False


def is_cycle_multigraph(G: nx.MultiGraph) -> bool:
    """Return whether a connected multigraph is a cycle.

    This includes the multigraph conventions ``C_1`` (one loop) and ``C_2``
    (two parallel edges).  NetworkX counts a loop twice in the degree, which is
    exactly what is needed for the degree-two characterization.
    """

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return False
    if len(connected_components_ignoring_loops(G)) != 1:
        return False
    return all(G.degree[node] == 2 for node in G.nodes())


def theta_edge_count(G: nx.MultiGraph) -> int | None:
    """Return ``s`` if ``G`` is the abstract ``Theta_s`` multigraph."""

    if G.number_of_nodes() != 2 or G.number_of_edges() == 0:
        return None

    nodes = list(G.nodes())
    u, v = nodes
    for a, b in G.edges():
        if a == b:
            return None
        if {a, b} != {u, v}:
            return None
    return G.number_of_edges()


def pick_nonloop_edge(G: nx.MultiGraph):
    """Pick one non-loop edge occurrence as ``(u, v, key)``, or return None."""

    for u, v, key in G.edges(keys=True):
        if u != v:
            return (u, v, key)
    return None


def _pick_loop_edge(G: nx.MultiGraph):
    for u, v, key in G.edges(keys=True):
        if u == v:
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


def _split_at_articulation(G: nx.MultiGraph) -> list[nx.MultiGraph] | None:
    """Split a connected multigraph into one-point-union factors if possible."""

    simple = _underlying_simple_graph(G)
    if simple.number_of_nodes() < 3:
        return None

    articulation = next(iter(nx.articulation_points(simple)), None)
    if articulation is None:
        return None

    reduced = simple.copy()
    reduced.remove_node(articulation)
    components = list(nx.connected_components(reduced))
    if len(components) < 2:
        return None

    parts = []
    for index, component in enumerate(components):
        nodes = set(component)
        nodes.add(articulation)
        part = G.subgraph(nodes).copy()

        # A loop based at the common articulation vertex belongs to the whole
        # graph only once.  Keep all such loops in the first factor and remove
        # duplicated copies from the remaining induced factors.
        if index > 0:
            loop_keys = [
                key
                for u, v, key in part.edges(articulation, keys=True)
                if u == v == articulation
            ]
            for key in loop_keys:
                part.remove_edge(articulation, articulation, key)
        parts.append(part)

    return parts


class YamadaRecursiveEvaluator:
    """Reusable crossing-free Yamada evaluator with shared memoization."""

    def __init__(self, variable: sp.Symbol):
        self.variable = variable
        self.sigma = variable + 1 + variable**(-1)
        self.memo: dict[object, sp.Expr] = {}
        self._memo_lock = threading.RLock()

    def _cache_get(self, key):
        with self._memo_lock:
            return self.memo.get(key)

    def _cache_set(self, key, value):
        value = sp.simplify(value)
        with self._memo_lock:
            self.memo[key] = value
        return value

    def compute(self, G: nx.Graph) -> sp.Expr:
        graph = _as_undirected_multigraph(G)
        return sp.simplify(self._rec(graph))

    def _rec(self, H: nx.MultiGraph) -> sp.Expr:
        H = normalize_multigraph(H)
        key = multigraph_key(H)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        n_vertices = H.number_of_nodes()
        n_edges = H.number_of_edges()

        # Edgeless graphs are disjoint unions of one-point graphs, H(.) = -1.
        if n_edges == 0:
            return self._cache_set(key, (-1) ** n_vertices)

        # H is multiplicative on disjoint unions.
        components = connected_components_ignoring_loops(H)
        if len(components) > 1:
            value = sp.Integer(1)
            for component in components:
                value *= self._rec(H.subgraph(component).copy())
            return self._cache_set(key, value)

        # Literature shortcut: any isthmus forces H(G)=0.
        if has_isthmus_multigraph(H):
            return self._cache_set(key, sp.Integer(0))

        # Closed forms from the standard Yamada identities.
        if is_cycle_multigraph(H):
            return self._cache_set(key, self.sigma)

        s = theta_edge_count(H)
        if s is not None:
            # Algebraically equivalent to
            #     (sigma + (-sigma)**s) / (sigma + 1),
            # but denominator-free.  This avoids removable (A+1) factors
            # after substituting sigma=A+1+A**(-1).
            value = sum(
                (-1) ** (power - 1) * self.sigma**power
                for power in range(1, s)
            )
            return self._cache_set(key, value)

        # Loop relation H(G) = -sigma H(G-e).
        loop = _pick_loop_edge(H)
        if loop is not None:
            value = -self.sigma * self._rec(delete_multigraph_edge(H, loop))
            return self._cache_set(key, value)

        # One-point union: H(G1 . ... . Gk) = (-1)^(k-1) product_i H(Gi).
        parts = _split_at_articulation(H)
        if parts is not None:
            value = (-1) ** (len(parts) - 1)
            for part in parts:
                value *= self._rec(part)
            return self._cache_set(key, value)

        edge = pick_nonloop_edge(H)
        if edge is None:
            # Reaching this branch would mean the graph has no edges, loops, or
            # non-loop edges; the edgeless base case above already handles that.
            return self._cache_set(key, (-1) ** n_vertices)

        value = (
            self._rec(delete_multigraph_edge(H, edge))
            + self._rec(contract_multigraph_edge(H, edge))
        )
        return self._cache_set(key, value)


class NegamiRecursiveEvaluator:
    """Recursive evaluator for Yamada's auxiliary Negami specialization h(G;x,y).

    The defining edge-subset sum is retained elsewhere as an independent
    reference implementation.  This evaluator uses the equivalent recurrences

        h(G) = h(G/e) - x^{-1} h(G-e)      for a non-loop edge,
        h(G) = (y - x^{-1}) h(G-e)         for a loop.

    It also exploits disjoint-union, one-point-union, and isthmus identities.
    """

    def __init__(self, x: sp.Symbol, y: sp.Symbol):
        self.x = x
        self.y = y
        self.memo: dict[object, sp.Expr] = {}
        self._memo_lock = threading.RLock()

    def _cache_get(self, key):
        with self._memo_lock:
            return self.memo.get(key)

    def _cache_set(self, key, value):
        value = sp.simplify(value)
        with self._memo_lock:
            self.memo[key] = value
        return value

    def compute(self, G: nx.Graph) -> sp.Expr:
        graph = _as_undirected_multigraph(G)
        return sp.expand(sp.simplify(self._rec(graph)))

    def _rec(self, H: nx.MultiGraph) -> sp.Expr:
        H = normalize_multigraph(H)
        key = multigraph_key(H)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        n_vertices = H.number_of_nodes()
        n_edges = H.number_of_edges()

        if n_edges == 0:
            return self._cache_set(key, self.x ** n_vertices)

        components = connected_components_ignoring_loops(H)
        if len(components) > 1:
            value = sp.Integer(1)
            for component in components:
                value *= self._rec(H.subgraph(component).copy())
            return self._cache_set(key, value)

        if has_isthmus_multigraph(H):
            return self._cache_set(key, sp.Integer(0))

        loop = _pick_loop_edge(H)
        if loop is not None:
            value = (self.y - self.x**(-1)) * self._rec(
                delete_multigraph_edge(H, loop)
            )
            return self._cache_set(key, value)

        parts = _split_at_articulation(H)
        if parts is not None:
            value = self.x ** (-(len(parts) - 1))
            for part in parts:
                value *= self._rec(part)
            return self._cache_set(key, value)

        edge = pick_nonloop_edge(H)
        if edge is None:
            return self._cache_set(key, self.x ** n_vertices)

        value = (
            self._rec(contract_multigraph_edge(H, edge))
            - self.x**(-1) * self._rec(delete_multigraph_edge(H, edge))
        )
        return self._cache_set(key, value)


def compute_yamada_polynomial_recursive(
    G: nx.Graph,
    variable: sp.Symbol,
) -> sp.Expr:
    """Compute the crossing-free Yamada polynomial by optimized recursion.

    ``G`` may be an undirected ``networkx.Graph`` or ``MultiGraph``. It is
    copied into a ``MultiGraph`` before evaluation; directed graphs are not
    supported.

    The evaluator uses the standard Yamada graph identities before falling back
    to deletion-contraction:

    - edgeless graph with ``n`` vertices -> ``(-1)^n``;
    - disjoint union -> product;
    - any graph containing an isthmus -> 0;
    - cycle ``C_n`` -> ``sigma``;
    - theta graph ``Theta_s`` -> ``(sigma + (-sigma)^s)/(sigma + 1)``;
    - loop edge ``e`` -> ``-sigma H(G-e)``;
    - one-point union -> minus the product of the two factors;
    - otherwise ``H(G)=H(G-e)+H(G/e)`` for a non-loop edge.
    """

    return YamadaRecursiveEvaluator(variable).compute(G)


def compute_negami_recursive(
    G: nx.Graph,
    x: sp.Symbol,
    y: sp.Symbol,
) -> sp.Expr:
    """Compute Yamada's auxiliary Negami polynomial recursively.

    ``G`` may be an undirected ``networkx.Graph`` or ``MultiGraph``. It is
    copied into a ``MultiGraph`` before evaluation; directed graphs are not
    supported.
    """

    return NegamiRecursiveEvaluator(x, y).compute(G)


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

    shifted_poly = sp.Poly(
        sp.expand(numerator / denominator),
        variable,
    )
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
