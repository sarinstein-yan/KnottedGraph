import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import PDCode, compute_yamada_polynomial


def _assert_expr_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _embedded_planar_theta():
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))

    curves = [
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
    ]
    for points in curves:
        graph.add_edge("u", "v", pts=points)
    return graph


def _embedded_one_crossing_theta(*, mirror=False):
    zsign = -1.0 if mirror else 1.0
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))

    curves = [
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, -1.0, 0.5 * zsign],
                [1.0, 1.0, 0.5 * zsign],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 1.0, -0.5 * zsign],
                [1.0, -1.0, -0.5 * zsign],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
    ]
    for points in curves:
        graph.add_edge("u", "v", pts=points)
    return graph


def _embedded_one_crossing_k4():
    positions = {
        0: np.array([-1.0, -1.0, 0.0]),
        1: np.array([1.0, -1.0, 0.5]),
        2: np.array([1.0, 1.0, -0.4]),
        3: np.array([-1.0, 1.0, 0.7]),
    }
    graph = nx.MultiGraph()
    for node, pos in positions.items():
        graph.add_node(node, pos=pos)
    for u, v in nx.complete_graph(4).edges():
        graph.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return graph


def _embedded_braid_theta(crossing_count: int, *, mirror=False, samples=200):
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
    edge3 = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.7, 2.0, 0.0],
            [1.7, 2.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    graph.add_edge("u", "v", pts=edge1)
    graph.add_edge("u", "v", pts=edge2)
    graph.add_edge("u", "v", pts=edge3)
    return graph


def _run_current_pipeline(name, graph, A, *, expected_crossings=None):
    processor = PDCode(graph)
    pd_code = processor.compute(
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
    )
    if expected_crossings is not None:
        assert len(processor.crossings) == expected_crossings

    result = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        normalize=False,
        n_jobs=1,
        return_result=True,
    )
    assert result.projection.pd_code == pd_code

    normalized = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        normalize=True,
        n_jobs=1,
    )
    assert normalized != 0

    return {
        "name": name,
        "pd_code": pd_code,
        "crossings": len(processor.crossings),
        "polynomial": sp.expand(result.polynomial),
    }


def test_current_spatial_graph_pipeline_through_pd_codes():
    A = sp.Symbol("A")
    cases = [
        ("planar theta", _embedded_planar_theta(), 0),
        ("one-crossing theta", _embedded_one_crossing_theta(), 1),
        ("mirror one-crossing theta", _embedded_one_crossing_theta(mirror=True), 1),
        ("one-crossing K4", _embedded_one_crossing_k4(), 1),
        ("three-crossing braid theta", _embedded_braid_theta(3), 3),
    ]
    for name, graph, crossings in cases:
        result = _run_current_pipeline(
            name,
            graph,
            A,
            expected_crossings=crossings,
        )
        assert result["polynomial"] != 0


def test_published_one_crossing_theta_after_full_3d_pd_pipeline():
    """Peddada et al., Example 2."""
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    expected = A**3 + A**2 + 2 * A + 1 + A**-1

    result = _run_current_pipeline(
        "Peddada one-crossing theta",
        _embedded_one_crossing_theta(),
        A,
        expected_crossings=1,
    )
    _assert_expr_equal(result["polynomial"], expected)
    _assert_expr_equal(result["polynomial"], -A * (sigma - sigma**2))


def test_three_crossing_theta_repeated_twist_identity_through_pd():
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    expected = (-A) ** 3 * (sigma - sigma**2)

    result = _run_current_pipeline(
        "three-crossing repeated-twist theta",
        _embedded_braid_theta(3),
        A,
        expected_crossings=3,
    )
    _assert_expr_equal(result["polynomial"], expected)


def test_mirror_and_A_plus_minus_one_spatial_identities_through_pd():
    """Dobrynin--Vesnin Proposition 1 and Proposition 2 / Corollary 2."""
    A = sp.Symbol("A")

    original = _run_current_pipeline(
        "theta crossing",
        _embedded_one_crossing_theta(),
        A,
        expected_crossings=1,
    )["polynomial"]
    mirror = _run_current_pipeline(
        "theta mirror",
        _embedded_one_crossing_theta(mirror=True),
        A,
        expected_crossings=1,
    )["polynomial"]

    mirrored_by_substitution = sp.expand(
        original.subs(A, A**-1, simultaneous=True)
    )
    _assert_expr_equal(mirror, mirrored_by_substitution)

    assert abs(int(original.subs(A, 1))) == 6
    assert abs(int(original.subs(A, -1))) == 2
    assert abs(int(mirror.subs(A, 1))) == 6
    assert abs(int(mirror.subs(A, -1))) == 2
