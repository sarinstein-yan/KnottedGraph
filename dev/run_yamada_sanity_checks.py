from __future__ import annotations

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada import (
    compute_negami,
    compute_negami_recursive,
    compute_yamada_polynomial_recursive,
)
from knotted_graph.projection import compute_yamada_polynomial

A = sp.Symbol("A")
x, y = sp.symbols("x y")
sigma = A + 1 + A**-1


def same_polynomial(left, right):
    return sp.simplify(sp.together(sp.expand(left - right))) == 0


def require_same(label, computed, expected):
    if not same_polynomial(computed, expected):
        raise AssertionError(
            f"{label} failed.\ncomputed={sp.expand(computed)}\nexpected={sp.expand(expected)}"
        )
    print(f"PASS  {label}")


def tree_graph(q):
    return nx.MultiGraph(nx.path_graph(q + 1))


def cycle_graph(n):
    if n == 1:
        graph = nx.MultiGraph()
        graph.add_node(0)
        graph.add_edge(0, 0)
        return graph
    if n == 2:
        graph = nx.MultiGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1)
        graph.add_edge(0, 1)
        return graph
    return nx.MultiGraph(nx.cycle_graph(n))


def bouquet_graph(q):
    graph = nx.MultiGraph()
    graph.add_node(0)
    for _ in range(q):
        graph.add_edge(0, 0)
    return graph


def theta_graph(s):
    graph = nx.MultiGraph()
    graph.add_nodes_from([0, 1])
    for _ in range(s):
        graph.add_edge(0, 1)
    return graph


def one_point_union(g1, g2):
    g1 = nx.convert_node_labels_to_integers(g1, first_label=0)
    g2 = nx.convert_node_labels_to_integers(g2, first_label=g1.number_of_nodes() - 1)
    return nx.MultiGraph(nx.compose(g1, g2))


def embedded_planar_theta():
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    curves = [
        np.array([[-2, 0, 0], [-1, 1, 0], [1, 1, 0], [2, 0, 0]], dtype=float),
        np.array([[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
        np.array([[-2, 0, 0], [-1, -1, 0], [1, -1, 0], [2, 0, 0]], dtype=float),
    ]
    for pts in curves:
        graph.add_edge("u", "v", pts=pts)
    return graph


def embedded_one_crossing_theta(*, mirror=False):
    zsign = -1.0 if mirror else 1.0
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    curves = [
        np.array([[-2, 0, 0], [-1, -1, 0.5 * zsign], [1, 1, 0.5 * zsign], [2, 0, 0]], dtype=float),
        np.array([[-2, 0, 0], [-1, 1, -0.5 * zsign], [1, -1, -0.5 * zsign], [2, 0, 0]], dtype=float),
        np.array([[-2, 0, 0], [-1, 2, 0], [1, 2, 0], [2, 0, 0]], dtype=float),
    ]
    for pts in curves:
        graph.add_edge("u", "v", pts=pts)
    return graph


def public_yamada(graph, method):
    return compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=False,
        n_jobs=1,
        method=method,
        return_result=True,
    )


def main():
    for q in range(1, 7):
        require_same(f"Tree T_{q}", compute_yamada_polynomial_recursive(tree_graph(q), A), 0)
    for n in range(1, 8):
        require_same(f"Cycle C_{n}", compute_yamada_polynomial_recursive(cycle_graph(n), A), sigma)
    for q in range(1, 7):
        require_same(
            f"Bouquet B_{q}",
            compute_yamada_polynomial_recursive(bouquet_graph(q), A),
            (-1) ** (q - 1) * sigma**q,
        )
    for s in range(1, 9):
        require_same(
            f"Theta_{s}",
            compute_yamada_polynomial_recursive(theta_graph(s), A),
            (sigma + (-sigma) ** s) / (sigma + 1),
        )

    left = cycle_graph(3)
    right = nx.relabel_nodes(cycle_graph(4), lambda node: node + 10)
    bridged = nx.compose(left, right)
    bridged.add_edge(0, 10)
    require_same(
        "Composite graph containing an isthmus",
        compute_yamada_polynomial_recursive(bridged, A),
        0,
    )

    g1 = theta_graph(3)
    g2 = bouquet_graph(2)
    wedge = one_point_union(g1, g2)
    require_same(
        "One-point union",
        compute_yamada_polynomial_recursive(wedge, A),
        -compute_yamada_polynomial_recursive(g1, A)
        * compute_yamada_polynomial_recursive(g2, A),
    )

    k4 = nx.MultiGraph(nx.complete_graph(4))
    require_same(
        "Planar K4",
        compute_yamada_polynomial_recursive(k4, A),
        A**3 + 2 * A + 2 * A**-1 + A**-3,
    )

    small_graphs = {
        "Bouquet B2": bouquet_graph(2),
        "Cycle C3": cycle_graph(3),
        "Theta3": theta_graph(3),
        "K4": k4,
        "Tree T2": tree_graph(2),
    }
    for name, graph in small_graphs.items():
        subset_h = compute_negami(graph, x, y)
        recursive_h = compute_negami_recursive(graph, x, y)
        require_same(f"{name}: direct vs recursive Negami", subset_h, recursive_h)
        specialized = recursive_h.xreplace({x: sp.Integer(-1), y: -A - 2 - A**-1})
        require_same(
            f"{name}: Negami specialization",
            specialized,
            compute_yamada_polynomial_recursive(graph, A),
        )

    planar_target = sigma - sigma**2
    planar = {method: public_yamada(embedded_planar_theta(), method) for method in ("negami", "recursive")}
    for method, result in planar.items():
        if result.projection.num_crossings != 0:
            raise AssertionError("Planar theta unexpectedly has a crossing")
        require_same(f"Public {method} backend, planar theta", result.polynomial, planar_target)
    require_same("Public backend agreement, planar theta", planar["negami"].polynomial, planar["recursive"].polynomial)

    crossed = {method: public_yamada(embedded_one_crossing_theta(), method) for method in ("negami", "recursive")}
    mirrored = {method: public_yamada(embedded_one_crossing_theta(mirror=True), method) for method in ("negami", "recursive")}
    for label, group in (("crossed", crossed), ("mirror", mirrored)):
        for method, result in group.items():
            if result.projection.num_crossings != 1:
                raise AssertionError(f"{label}/{method}: expected one crossing")
        require_same(f"{label}: backend agreement", group["negami"].polynomial, group["recursive"].polynomial)

    crossed_factor = sp.simplify(crossed["recursive"].polynomial / planar_target)
    mirror_factor = sp.simplify(mirrored["recursive"].polynomial / planar_target)
    if crossed_factor not in {-A, -A**-1}:
        raise AssertionError(f"Unexpected crossed theta factor {crossed_factor}")
    if mirror_factor not in {-A, -A**-1}:
        raise AssertionError(f"Unexpected mirror theta factor {mirror_factor}")
    if sp.simplify(crossed_factor * mirror_factor - 1) != 0:
        raise AssertionError("Mirroring did not exchange -A and -A^-1")
    require_same(
        "Mirror relation R_mirror(A)=R(A^-1)",
        mirrored["recursive"].polynomial,
        sp.expand(crossed["recursive"].polynomial.subs(A, A**-1, simultaneous=True)),
    )

    print("PASS: all published/independent Yamada sanity checks succeeded.")


if __name__ == "__main__":
    main()
