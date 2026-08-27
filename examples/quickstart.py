"""Deterministic smoke test for the core and embedded-graph APIs."""

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive
from knotted_graph.projection import compute_yamada_polynomial


def build_planar_theta() -> nx.MultiGraph:
    """Return a planar 3D embedding of the three-edge theta graph."""
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))

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
    for points in curves:
        graph.add_edge("u", "v", pts=points)
    return graph


def compute_quickstart():
    """Compute matching nonzero values through the two public entry points."""
    Y = sp.Symbol("Y")
    abstract = sp.expand(
        compute_yamada_polynomial_recursive(ThetaGraph(3), Y)
    )
    embedded = compute_yamada_polynomial(
        build_planar_theta(),
        Y,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=False,
        n_jobs=1,
        method="recursive",
        return_result=True,
    )

    expected = -Y**2 - Y - 2 - Y**-1 - Y**-2
    if sp.simplify(abstract - expected) != 0:
        raise RuntimeError("The abstract quick-start result changed unexpectedly.")
    if sp.simplify(embedded.polynomial - expected) != 0:
        raise RuntimeError("The embedded quick-start result changed unexpectedly.")
    if embedded.projection.num_crossings != 0:
        raise RuntimeError("The quick-start projection should be crossing-free.")

    return abstract, embedded


def main() -> None:
    """Print the stable output used in the README and installation guide."""
    abstract, embedded = compute_quickstart()
    print(f"Abstract Upsilon(Theta_3; Y) = {abstract}")
    print(f"Embedded Upsilon(Theta_3; Y) = {embedded.polynomial}")
    print(
        "Selected projection crossings = "
        f"{embedded.projection.num_crossings}"
    )


if __name__ == "__main__":
    main()
