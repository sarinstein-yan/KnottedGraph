import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial


def _assert_expr_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _embedded_theta(*, crossed: bool, mirror: bool = False) -> nx.MultiGraph:
    """A theta embedding with either zero or one transverse projected crossing."""
    zsign = -1.0 if mirror else 1.0

    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))

    if not crossed:
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
    else:
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
        graph.add_edge("u", "v", pts=pts)

    return graph


def _compute(graph, A, method):
    return compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=False,
        n_jobs=1,
        method=method,
        return_result=True,
    )


def test_public_backends_reproduce_planar_theta_literature_value():
    """Peddada et al.: R(theta)=B-B^2, B=A+1+A^-1."""
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    expected = sigma - sigma**2

    negami = _compute(_embedded_theta(crossed=False), A, "negami")
    recursive = _compute(_embedded_theta(crossed=False), A, "recursive")

    assert negami.projection.num_crossings == 0
    assert recursive.projection.num_crossings == 0
    _assert_expr_equal(negami.polynomial, expected)
    _assert_expr_equal(recursive.polynomial, expected)
    _assert_expr_equal(negami.polynomial, recursive.polynomial)


def test_public_backends_agree_for_one_crossing_theta_and_mirror():
    """A one-crossing theta differs from the planar theta by -A or -A^-1.

    Peddada et al., Example 2 / relation Y5, gives the corresponding factor.
    Which factor appears depends on which strand is over; mirroring swaps them.
    """
    A = sp.Symbol("A")
    sigma = A + 1 + A**-1
    planar = sigma - sigma**2

    crossed_negami = _compute(_embedded_theta(crossed=True), A, "negami")
    crossed_recursive = _compute(_embedded_theta(crossed=True), A, "recursive")
    mirror_negami = _compute(
        _embedded_theta(crossed=True, mirror=True),
        A,
        "negami",
    )
    mirror_recursive = _compute(
        _embedded_theta(crossed=True, mirror=True),
        A,
        "recursive",
    )

    assert crossed_negami.projection.num_crossings == 1
    assert crossed_recursive.projection.num_crossings == 1
    assert mirror_negami.projection.num_crossings == 1
    assert mirror_recursive.projection.num_crossings == 1

    _assert_expr_equal(
        crossed_negami.polynomial,
        crossed_recursive.polynomial,
    )
    _assert_expr_equal(
        mirror_negami.polynomial,
        mirror_recursive.polynomial,
    )

    crossed_factor = sp.simplify(crossed_recursive.polynomial / planar)
    mirror_factor = sp.simplify(mirror_recursive.polynomial / planar)

    assert crossed_factor in {-A, -A**-1}
    assert mirror_factor in {-A, -A**-1}
    assert sp.simplify(crossed_factor * mirror_factor - 1) == 0

    mirrored_by_substitution = sp.expand(
        crossed_recursive.polynomial.subs(A, A**-1, simultaneous=True)
    )
    _assert_expr_equal(
        mirror_recursive.polynomial,
        mirrored_by_substitution,
    )
