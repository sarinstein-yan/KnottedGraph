from __future__ import annotations

import contextlib
import io
import json
from typing import Any

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.applications.mathematical import (
    NOTEBOOK_YAMADA_EXAMPLES,
    build_graph_case,
)
from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import (
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)
from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import (
    compute_graph_yamada_polynomial,
    laurent_y_to_sigma_polynomial,
)
from knotted_graph.projection import PDCode, compute_yamada_polynomial, select_projection

Y = sp.Symbol("Y")
sigma = sp.Symbol("sigma")
kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


def _graph_signature(graph: nx.MultiGraph) -> dict[str, Any]:
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "degree_sequence": sorted(int(degree) for _, degree in graph.degree()),
        "components": nx.number_connected_components(graph),
    }


def _expanded(expr: sp.Expr) -> str:
    return str(sp.expand(expr))


def _rotation_tuple(rotation) -> list[float]:
    return [float(value) for value in rotation]


def _pd_at_rotation(graph: nx.MultiGraph, rotation) -> tuple[str, int]:
    processor = PDCode(graph)
    pd_code = processor.compute(rotation_angles=rotation)
    return pd_code, len(processor.crossings)


def _physics_cases() -> list[dict[str, Any]]:
    """Reproduce Section 7 of 01_physics_applications.ipynb without plotting."""
    yamada_table_specs = [
        ("Hopf link", hopf_link_bloch_vector, [0.10, 0.20, 0.50]),
        ("Trefoil", trefoil_bloch_vector, [0.10, 0.19, 0.25]),
        (
            "Torus (1,2)",
            lambda gamma, k_symbols: pq_torus_knot_bloch_vector(
                1,
                2,
                gamma,
                k_symbols=k_symbols,
            ),
            [0.12, 0.50, 0.70],
        ),
        ("Solomon", solomon_bloch_vector, [0.12, 1.00, 2.00]),
    ]

    rows: list[dict[str, Any]] = []
    for family, builder, gammas in yamada_table_specs:
        for gamma in gammas:
            skeleton = NodalSkeleton(
                builder(gamma, k_symbols=(kx, ky, kz)),
                k_symbols=(kx, ky, kz),
                dimension=200,
                axis_scale=(1.0, 1.0, 1.5),
            )
            graph = skeleton.skeleton_graph(
                simplify=True,
                smooth_epsilon=2,
            )
            projection = select_projection(graph, num_rotation_samples=8)
            rotation = projection.rotation_angles
            polynomial = compute_yamada_polynomial(
                graph,
                Y,
                rotation_angles=rotation,
                n_jobs=1,
            )
            pd_code, pd_crossings = _pd_at_rotation(graph, rotation)
            if pd_crossings != projection.num_crossings:
                raise AssertionError(
                    f"{family} gamma={gamma}: projection/PD crossing count mismatch "
                    f"({projection.num_crossings} != {pd_crossings})"
                )

            rows.append(
                {
                    "application": "physics",
                    "case": family,
                    "gamma": gamma,
                    "dimension": 200,
                    **_graph_signature(graph),
                    "rotation_angles": _rotation_tuple(rotation),
                    "crossings": int(projection.num_crossings),
                    "pd_code": pd_code,
                    "yamada": _expanded(polynomial),
                }
            )
    return rows


def _mathematical_k4_spine(samples: int = 90, amplitude: float = 0.75) -> nx.MultiGraph:
    """Exact constructor from Section 2 of 02_mathematics_applications.ipynb."""
    vertices = {
        "a": np.array([-1.15, -0.78, -0.38]),
        "b": np.array([1.18, -0.64, 0.30]),
        "c": np.array([0.86, 0.95, -0.26]),
        "d": np.array([-0.88, 0.84, 0.52]),
    }
    edge_specs = [
        ("a", "b", "ab", np.array([0.00, 0.90, 0.70]), 0.0),
        ("a", "c", "ac", np.array([0.35, -0.15, 1.00]), 1.1),
        ("a", "d", "ad", np.array([0.95, 0.15, -0.25]), 2.2),
        ("b", "c", "bc", np.array([-0.90, 0.25, 0.35]), 0.7),
        ("b", "d", "bd", np.array([-0.20, 1.00, -0.60]), 1.7),
        ("c", "d", "cd", np.array([0.10, -0.90, -0.85]), 2.8),
    ]
    s = np.linspace(0.0, 1.0, samples)
    graph = nx.MultiGraph()
    for vertex_id, pos in vertices.items():
        graph.add_node(vertex_id, pos=pos.copy())

    for u, v, key, bend, phase in edge_specs:
        start = vertices[u]
        end = vertices[v]
        chord = end - start
        bend = bend / np.linalg.norm(bend)
        side = np.cross(chord, bend)
        side = side / np.linalg.norm(side)
        envelope = np.sin(np.pi * s)
        pts = (1 - s)[:, None] * start + s[:, None] * end
        pts += amplitude * envelope[:, None] * (
            np.cos(phase + np.pi * s)[:, None] * bend
            + 0.6 * np.sin(2 * np.pi * s + phase)[:, None] * side
        )
        pts[0] = start
        pts[-1] = end
        graph.add_edge(u, v, key=key, pts=pts)

    graph.graph.update(
        graph_id="mathematical_k4",
        input_kind="internal_mathematical_geometry",
        is_closed=True,
    )
    return graph


def _math_k4_case() -> list[dict[str, Any]]:
    graph = _mathematical_k4_spine()
    projection = select_projection(graph, num_rotation_samples=16)
    rotation = projection.rotation_angles
    negami = compute_yamada_polynomial(
        graph,
        Y,
        rotation_angles=rotation,
        method="negami",
        n_jobs=1,
    )
    recursive = compute_yamada_polynomial(
        graph,
        Y,
        rotation_angles=rotation,
        method="recursive",
        n_jobs=1,
    )
    if sp.expand(negami - recursive) != 0:
        raise AssertionError("Mathematics K4 application backends disagree")
    pd_code, pd_crossings = _pd_at_rotation(graph, rotation)
    if pd_crossings != projection.num_crossings:
        raise AssertionError("Mathematics K4 projection/PD crossing count mismatch")

    return [
        {
            "application": "mathematics",
            "case": "embedded K4 spine",
            **_graph_signature(graph),
            "rotation_angles": _rotation_tuple(rotation),
            "crossings": int(projection.num_crossings),
            "pd_code": pd_code,
            "yamada_negami": _expanded(negami),
            "yamada_recursive": _expanded(recursive),
        }
    ]


def _math_theta_cases() -> list[dict[str, Any]]:
    rows = []
    for s in range(2, 8):
        graph = ThetaGraph(s)
        polynomial = compute_graph_yamada_polynomial(graph, Y)
        rows.append(
            {
                "application": "mathematics",
                "case": f"Theta_{s}",
                "s": s,
                **_graph_signature(graph),
                "yamada": _expanded(polynomial),
            }
        )
    return rows


def _math_catalog_cases() -> list[dict[str, Any]]:
    """Reproduce Section 4.1 without graph plotting."""
    rows = []
    for family_name, args, label in NOTEBOOK_YAMADA_EXAMPLES:
        graph, _ = build_graph_case(family_name, *args)
        yamada_y = sp.expand(compute_graph_yamada_polynomial(graph, Y))
        yamada_sigma = laurent_y_to_sigma_polynomial(yamada_y, Y, sigma).as_expr()
        rows.append(
            {
                "application": "mathematics",
                "case": label,
                "family": family_name,
                "args": list(args),
                **_graph_signature(graph),
                "yamada": _expanded(yamada_y),
                "yamada_sigma": _expanded(yamada_sigma),
            }
        )
    return rows


def _math_cylinder_scan() -> list[dict[str, Any]]:
    """Reproduce Section 4.2 without plotting."""
    rows = []
    for cols in range(3, 7):
        graph, _ = build_graph_case("cylinder", 2, cols)
        yamada_y = sp.expand(compute_graph_yamada_polynomial(graph, Y))
        yamada_sigma = laurent_y_to_sigma_polynomial(yamada_y, Y, sigma).as_expr()
        rows.append(
            {
                "application": "mathematics",
                "case": f"Cylinder(2,{cols})",
                "rows": 2,
                "cols": cols,
                **_graph_signature(graph),
                "yamada": _expanded(yamada_y),
                "yamada_sigma": _expanded(yamada_sigma),
            }
        )
    return rows


def evaluate_application_yamada_cases() -> list[dict[str, Any]]:
    # Some optional scientific routines print progress messages. Suppress them so stdout
    # remains one machine-readable JSON document when this file is used cross-branch.
    with contextlib.redirect_stdout(io.StringIO()):
        rows = []
        rows.extend(_physics_cases())
        rows.extend(_math_k4_case())
        rows.extend(_math_theta_cases())
        rows.extend(_math_catalog_cases())
        rows.extend(_math_cylinder_scan())
    return rows


def main() -> None:
    rows = evaluate_application_yamada_cases()
    print(json.dumps(rows, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
