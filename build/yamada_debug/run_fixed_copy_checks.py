from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

kg_pkg = types.ModuleType("knotted_graph")
kg_pkg.__path__ = [str(SRC / "knotted_graph")]
sys.modules.setdefault("knotted_graph", kg_pkg)
yamada_pkg = types.ModuleType("knotted_graph.yamada")
yamada_pkg.__path__ = [str(SRC / "knotted_graph" / "yamada")]
sys.modules.setdefault("knotted_graph.yamada", yamada_pkg)

from knotted_graph.yamada.pd_code import PDCode
from yamada_fixed_copy import compute_graph_fixed_copy


Y = sp.Symbol("Y")


def theta_graph(edge0, edge1, edge2=None):
    a = (-3.0, 0.0, 0.0)
    b = (3.0, 0.0, 0.0)
    if edge2 is None:
        edge2 = [a, (-1.0, -2.5, 0.0), (1.0, -2.5, 0.0), b]
    graph = nx.MultiGraph()
    graph.add_node("A", pos=a)
    graph.add_node("B", pos=b)
    graph.add_edge("A", "B", key="e0", pts=edge0)
    graph.add_edge("A", "B", key="e1", pts=edge1)
    graph.add_edge("A", "B", key="e2", pts=edge2)
    return graph


def crossing_free_theta():
    a = (-3.0, 0.0, 0.0)
    b = (3.0, 0.0, 0.0)
    return theta_graph(
        [a, (-1.0, 1.2, 0.0), (1.0, 1.2, 0.0), b],
        [a, (-1.0, 2.5, 0.0), (1.0, 2.5, 0.0), b],
    )


def r2_bigon_theta():
    a = (-3.0, 0.0, 0.0)
    b = (3.0, 0.0, 0.0)
    return theta_graph(
        [a, b],
        [a, (-2.0, 1.0, 1.0), (0.0, -1.0, 1.0), (2.0, 1.0, 1.0), b],
    )


def self_crossing_theta():
    a = (-3.0, 0.0, 0.0)
    b = (3.0, 0.0, 0.0)
    return theta_graph(
        [
            a,
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            b,
        ],
        [a, (-1.0, 2.5, 0.0), (1.0, 2.5, 0.0), b],
    )


def graph_from_layout(layout):
    graph = nx.MultiGraph()
    node_positions = {
        node: np.asarray(layout["node_positions"][node], dtype=float)
        for node in layout["node_order"]
    }
    for node, pos in node_positions.items():
        graph.add_node(node, pos=tuple(float(x) for x in pos))
    for edge_name in layout["edge_order"]:
        pts = np.asarray(layout["edge_polylines"][edge_name], dtype=float)
        start = min(node_positions, key=lambda n: float(np.linalg.norm(pts[0] - node_positions[n])))
        end = min(node_positions, key=lambda n: float(np.linalg.norm(pts[-1] - node_positions[n])))
        graph.add_edge(
            start,
            end,
            key=edge_name,
            pts=[tuple(float(x) for x in point) for point in pts],
        )
    return graph


def original_raw(graph, angles=(0.0, 0.0, 0.0)):
    pd = PDCode(graph, tolerance=1e-7)
    code = pd.compute(angles, "ZYX")
    return code, sp.expand(pd.compute_yamada(Y, normalize=False, n_jobs=1))


def is_monomial_equivalent(a, b):
    if a == 0 or b == 0:
        return a == b, None
    for shift in range(-40, 41):
        for sign in (1, -1):
            if sp.simplify(a - sign * (Y**shift) * b) == 0:
                return True, (sign, shift)
    return False, None


def minimal_checks():
    base = crossing_free_theta()
    r2 = r2_bigon_theta()
    self_crossing = self_crossing_theta()

    _, base_original = original_raw(base)
    _, r2_original = original_raw(r2)
    base_fixed = compute_graph_fixed_copy(base, Y)
    r2_fixed = compute_graph_fixed_copy(r2, Y)
    self_fixed = compute_graph_fixed_copy(self_crossing, Y)

    print("minimal checks")
    print(f"  base original raw: {base_original}")
    print(f"  r2 original raw:   {r2_original}")
    print(f"  base fixed raw:    {base_fixed.raw}")
    print(f"  r2 fixed raw:      {r2_fixed.raw}")
    print(f"  fixed R2 equivalent? {is_monomial_equivalent(base_fixed.raw, r2_fixed.raw)}")
    print(f"  self-crossing fixed raw: {self_fixed.raw}")


def protein_checks():
    rotations = {
        "1aoc": [(0.0, 89.6419, 0.0), (137.5078, 88.9256, 0.0)],
        "3ulk": [(75.5124, 33.7726, 0.0), (-146.9798, 32.4617, 0.0)],
        "5osq": [(0.0, 89.6419, 0.0), (-84.9845, 88.2092, 0.0)],
    }
    base_dir = ROOT / "build" / "repulsive_layout_route_c_stronger_final_relax"
    print("\nprotein checks")
    for sample, angle_list in rotations.items():
        graph = graph_from_layout(
            json.loads(
                (base_dir / sample / "04_route_c_stronger_smooth_relax" / "layout.json").read_text()
            )
        )
        fixed = [
            compute_graph_fixed_copy(graph, Y, rotation_angles=angles)
            for angles in angle_list
        ]
        print(f"  {sample}")
        for angles, result in zip(angle_list, fixed):
            print(f"    angles={angles} crossings={result.crossings}")
            print(f"      raw={result.raw}")
            print(f"      normalized={result.normalized}")
        print(f"    fixed raw equivalent? {is_monomial_equivalent(fixed[0].raw, fixed[1].raw)}")
        print(f"    fixed normalized equal? {sp.simplify(fixed[0].normalized - fixed[1].normalized) == 0}")


def main():
    minimal_checks()
    protein_checks()


if __name__ == "__main__":
    main()
