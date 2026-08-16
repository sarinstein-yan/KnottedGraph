import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    compute_negami,
)
from knotted_graph.invariants.yamada.recursive import (
    connected_components_ignoring_loops,
    contract_multigraph_edge,
    delete_multigraph_edge,
    multigraph_key,
    normalize_multigraph,
    pick_nonloop_edge,
)
from knotted_graph.projection import PDCode, compute_yamada_polynomial


LEGACY_RECURSIVE_BLOB_SHA = "8ecc715e84fda650c370475c1a43ce016a127e45"
LEGACY_POLYNOMIAL_BLOB_SHA = "8c228fab4ab90f76dfe13ff69279205ae48cc187"


def _assert_expr_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _legacy_recursive_yamada(G: nx.MultiGraph, A: sp.Symbol) -> sp.Expr:
    """Frozen pre-upgrade deletion-contraction implementation.

    This reproduces the recursive algorithm from the Latest_Workplace file
    identified by LEGACY_RECURSIVE_BLOB_SHA.  It intentionally does not use the
    new bridge/cycle/theta/one-point-union shortcuts.
    """
    sigma = A + 1 + A**-1
    memo = {}

    def rec(H):
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
                loops = sum(
                    1
                    for u, v, key in H.edges(keys=True)
                    if u == v == 0
                )
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



def _legacy_state_sum(processor: PDCode, A: sp.Symbol, backend: str):
    """Evaluate one already-computed PD diagram with the pre-upgrade algorithms."""
    calculator = Yamada(
        vertices=list(processor.vertices.values()),
        crossings=list(processor.crossings.values()),
        arcs=list(processor.arcs.values()),
    )
    state_graphs, exponents = calculator._build_state_graphs()

    if backend == "negami":
        x, y = sp.symbols("x y")
        state_values = [
            compute_negami(G, x, y).xreplace(
                {x: sp.Integer(-1), y: -A - 2 - A**-1}
            )
            for G in state_graphs
        ]
    elif backend == "recursive":
        state_values = [
            _legacy_recursive_yamada(G, A)
            for G in state_graphs
        ]
    else:
        raise ValueError(backend)

    total = sp.expand(
        sp.simplify(
            sum(
                A**exponent * value
                for exponent, value in zip(exponents, state_values)
            )
        )
    )
    return total


def _embedded_planar_theta():
    G = nx.MultiGraph()
    G.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    G.add_node("v", pos=np.array([2.0, 0.0, 0.0]))

    curves = [
        np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
    ]
    for pts in curves:
        G.add_edge("u", "v", pts=pts)
    return G


def _embedded_one_crossing_theta(*, mirror=False):
    zsign = -1.0 if mirror else 1.0
    G = nx.MultiGraph()
    G.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    G.add_node("v", pos=np.array([2.0, 0.0, 0.0]))

    curves = [
        np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, -1.0, 0.5 * zsign],
            [1.0, 1.0, 0.5 * zsign],
            [2.0, 0.0, 0.0],
        ]),
        np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, 1.0, -0.5 * zsign],
            [1.0, -1.0, -0.5 * zsign],
            [2.0, 0.0, 0.0],
        ]),
        np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
    ]
    for pts in curves:
        G.add_edge("u", "v", pts=pts)
    return G


def _embedded_one_crossing_k4():
    """A tetrahedral K4 embedding whose xy projection has one diagonal crossing."""
    positions = {
        0: np.array([-1.0, -1.0, 0.0]),
        1: np.array([1.0, -1.0, 0.5]),
        2: np.array([1.0, 1.0, -0.4]),
        3: np.array([-1.0, 1.0, 0.7]),
    }
    G = nx.MultiGraph()
    for node, pos in positions.items():
        G.add_node(node, pos=pos)

    for u, v in nx.complete_graph(4).edges():
        G.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return G


def _embedded_braid_theta(crossing_count: int, *, mirror=False, samples=200):
    """Odd positive two-braid plus an exterior third edge.

    For odd crossing_count this realizes the theta-graph members of the
    Dobrynin--Vesnin Theta(n) family.  The z-sign convention is chosen so n=1
    agrees with their R(Theta(1)) orientation.
    """
    if crossing_count < 1 or crossing_count % 2 == 0:
        raise ValueError("Use a positive odd crossing count.")

    sign = -1.0 if mirror else 1.0
    t = np.linspace(0.0, 1.0, int(samples))
    x = -2.0 + 4.0 * t

    y1 = -0.8 * np.sin((crossing_count + 1) * np.pi * t)
    y2 = -y1

    z1 = (
        sign
        * -0.55
        * np.sin(np.pi * t)
        * np.cos((crossing_count + 1) * np.pi * t)
    )
    z2 = -z1

    edge1 = np.column_stack([x, y1, z1])
    edge2 = np.column_stack([x, y2, z2])
    edge3 = np.array([
        [-2.0, 0.0, 0.0],
        [-1.7, 2.0, 0.0],
        [1.7, 2.0, 0.0],
        [2.0, 0.0, 0.0],
    ])

    G = nx.MultiGraph()
    G.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    G.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    G.add_edge("u", "v", pts=edge1)
    G.add_edge("u", "v", pts=edge2)
    G.add_edge("u", "v", pts=edge3)
    return G


def _run_full_regression(name, graph, A, *, expected_crossings=None):
    """Projection is performed first; every invariant is evaluated from that PD data."""
    baseline_processor = PDCode(graph)
    baseline_pd = baseline_processor.compute(
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
    )

    if expected_crossings is not None:
        assert len(baseline_processor.crossings) == expected_crossings

    before_negami = _legacy_state_sum(
        baseline_processor,
        A,
        "negami",
    )
    before_recursive = _legacy_state_sum(
        baseline_processor,
        A,
        "recursive",
    )

    after_negami = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        normalize=False,
        n_jobs=1,
        method="negami",
        return_result=True,
    )
    after_recursive = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        normalize=False,
        n_jobs=1,
        method="recursive",
        return_result=True,
    )

    assert after_negami.projection.pd_code == baseline_pd
    assert after_recursive.projection.pd_code == baseline_pd

    _assert_expr_equal(before_negami, before_recursive)
    _assert_expr_equal(before_negami, after_negami.polynomial)
    _assert_expr_equal(before_negami, after_recursive.polynomial)

    # Protect the user-facing normalization using KnottedGraph's own API.
    after_normalized_negami = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        normalize=True,
        n_jobs=1,
        method="negami",
    )
    after_normalized_recursive = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        normalize=True,
        n_jobs=1,
        method="recursive",
    )
    _assert_expr_equal(
        after_normalized_negami,
        after_normalized_recursive,
    )

    return {
        "name": name,
        "pd_code": baseline_pd,
        "crossings": len(baseline_processor.crossings),
        "polynomial": sp.expand(before_negami),
    }


def test_before_after_agree_for_spatial_graphs_through_pd_codes():
    """No algorithmic result changes after the recursive optimization."""
    A = sp.Symbol("A")

    cases = [
        ("planar theta", _embedded_planar_theta(), 0),
        ("one-crossing theta", _embedded_one_crossing_theta(), 1),
        ("mirror one-crossing theta", _embedded_one_crossing_theta(mirror=True), 1),
        ("one-crossing K4", _embedded_one_crossing_k4(), 1),
        ("three-crossing braid theta", _embedded_braid_theta(3), 3),
    ]

    for name, graph, crossings in cases:
        _run_full_regression(
            name,
            graph,
            A,
            expected_crossings=crossings,
        )


def test_published_one_crossing_theta_after_full_3d_pd_pipeline():
    """Peddada et al., Example 2."""
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    expected = A**3 + A**2 + 2*A + 1 + A**-1

    result = _run_full_regression(
        "Peddada one-crossing theta",
        _embedded_one_crossing_theta(),
        A,
        expected_crossings=1,
    )
    _assert_expr_equal(result["polynomial"], expected)
    _assert_expr_equal(result["polynomial"], -A * (sigma - sigma**2))


def test_three_crossing_theta_repeated_twist_identity_through_pd():
    """Three repeated Y5/R6 twists give (-A)^3 times the planar theta value.

    Peddada et al. give the single-twist identity R(g_twist)=-A R(g), and
    their case-study table explicitly lists repeated representatives with
    factors (-A)^1, (-A)^2, and (-A)^3.
    """
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    expected = (-A)**3 * (sigma - sigma**2)

    result = _run_full_regression(
        "three-crossing repeated-twist theta",
        _embedded_braid_theta(3),
        A,
        expected_crossings=3,
    )
    _assert_expr_equal(result["polynomial"], expected)


def test_mirror_and_A_plus_minus_one_spatial_identities_through_pd():
    """Dobrynin--Vesnin Proposition 1 and Proposition 2 / Corollary 2."""
    A = sp.Symbol("A")

    original = _run_full_regression(
        "theta crossing",
        _embedded_one_crossing_theta(),
        A,
        expected_crossings=1,
    )["polynomial"]
    mirror = _run_full_regression(
        "theta mirror",
        _embedded_one_crossing_theta(mirror=True),
        A,
        expected_crossings=1,
    )["polynomial"]

    mirrored_by_substitution = sp.expand(
        original.subs(A, A**-1, simultaneous=True)
    )
    _assert_expr_equal(mirror, mirrored_by_substitution)

    # The underlying abstract graph is theta, so |R(1)|=6 and |R(-1)|=2.
    assert abs(int(original.subs(A, 1))) == 6
    assert abs(int(original.subs(A, -1))) == 2
    assert abs(int(mirror.subs(A, 1))) == 6
    assert abs(int(mirror.subs(A, -1))) == 2
