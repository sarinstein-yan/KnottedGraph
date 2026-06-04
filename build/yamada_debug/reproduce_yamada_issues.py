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

# Avoid importing optional top-level dependencies while debugging Yamada only.
kg_pkg = types.ModuleType("knotted_graph")
kg_pkg.__path__ = [str(SRC / "knotted_graph")]
sys.modules.setdefault("knotted_graph", kg_pkg)
yamada_pkg = types.ModuleType("knotted_graph.yamada")
yamada_pkg.__path__ = [str(SRC / "knotted_graph" / "yamada")]
sys.modules.setdefault("knotted_graph.yamada", yamada_pkg)

from knotted_graph.yamada.pd_code import PDCode
from knotted_graph.yamada.util import get_rotation_matrix


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


def transform_graph(graph, matrix):
    transformed = nx.MultiGraph()
    for node, data in graph.nodes(data=True):
        pos = matrix @ np.asarray(data["pos"], dtype=float)
        transformed.add_node(node, pos=tuple(float(x) for x in pos))
    for u, v, key, data in graph.edges(keys=True, data=True):
        pts = [
            tuple(float(x) for x in matrix @ np.asarray(point, dtype=float))
            for point in data["pts"]
        ]
        transformed.add_edge(u, v, key=key, pts=pts)
    return transformed


def compute(graph, angles=(0.0, 0.0, 0.0)):
    pd = PDCode(graph, tolerance=1e-7)
    code = pd.compute(rotation_angles=angles, rotation_order="ZYX")
    raw = sp.expand(pd.compute_yamada(Y, normalize=False, n_jobs=1))
    norm = sp.expand(pd.compute_yamada(Y, normalize=True, n_jobs=1))
    return pd, code, raw, norm


def is_monomial_equivalent(a, b, max_shift=24):
    if a == 0 or b == 0:
        return a == b, None
    for shift in range(-max_shift, max_shift + 1):
        for sign in (1, -1):
            if sp.simplify(a - sign * (Y**shift) * b) == 0:
                return True, (sign, shift)
    return False, None


def print_case(title, graph, angles=(0.0, 0.0, 0.0)):
    pd, code, raw, norm = compute(graph, angles)
    print(f"\n{title}")
    print(f"  crossings={len(pd.crossings)} arcs={len(pd.arcs)}")
    print(f"  PD={code}")
    print(f"  raw={raw}")
    print(f"  normalized={norm}")
    return raw, norm


def diagnose_minimal_examples():
    base_raw, base_norm = print_case("crossing-free theta", crossing_free_theta())
    r2_raw, r2_norm = print_case("theta with removable R2 bigon", r2_bigon_theta())
    same, factor = is_monomial_equivalent(base_raw, r2_raw)
    print(f"  R2 raw monomial-equivalent to base? {same}, factor={factor}")
    print(f"  R2 normalized equals base? {sp.simplify(base_norm - r2_norm) == 0}")

    pd, code, raw, norm = compute(self_crossing_theta())
    skipped = [xid for xid, crossing in pd.crossings.items() if not crossing.ccw_ordered_arcs]
    print("\nself-crossing theta")
    print(f"  crossings={len(pd.crossings)} skipped_crossings={skipped}")
    print(f"  PD={code}")
    print(f"  raw={raw}")


def diagnose_in_plane_rotation():
    graph = crossing_free_theta()
    # Use a graph with no crossings to keep this check cheap and deterministic.
    values = []
    for angle in (0.0, 30.0, 90.0, 180.0):
        matrix = get_rotation_matrix((angle, 0.0, 0.0), "ZYX")
        _, code, _, norm = compute(transform_graph(graph, matrix))
        values.append(norm)
        print(f"  z-rotation={angle:g} PD={code} normalized={norm}")
    print(f"  in-plane relabel stable? {all(sp.simplify(values[0] - v) == 0 for v in values[1:])}")


def diagnose_protein_projection_dependence():
    cases = {
        "1aoc": [(0.0, 89.6419, 0.0), (137.5078, 88.9256, 0.0)],
        "3ulk": [(75.5124, 33.7726, 0.0), (-146.9798, 32.4617, 0.0)],
        "5osq": [(0.0, 89.6419, 0.0), (-84.9845, 88.2092, 0.0)],
    }
    base_dir = ROOT / "build" / "repulsive_layout_route_c_stronger_final_relax"
    for sample, rotations in cases.items():
        path = base_dir / sample / "04_route_c_stronger_smooth_relax" / "layout.json"
        if not path.exists():
            print(f"\n{sample}: missing {path}")
            continue
        graph = graph_from_layout(json.loads(path.read_text()))
        raws = []
        norms = []
        print(f"\n{sample}")
        for angles in rotations:
            pd, code, raw, norm = compute(graph, angles)
            raws.append(raw)
            norms.append(norm)
            skipped = [xid for xid, crossing in pd.crossings.items() if not crossing.ccw_ordered_arcs]
            print(f"  angles={angles} crossings={len(pd.crossings)} skipped={skipped}")
            print(f"    PD={code}")
            print(f"    raw={raw}")
            print(f"    normalized={norm}")
        same, factor = is_monomial_equivalent(raws[0], raws[1])
        print(f"  raw monomial-equivalent? {same}, factor={factor}")
        print(f"  normalized equal? {sp.simplify(norms[0] - norms[1]) == 0}")


def main():
    print("Yamada debugging with unmodified src implementation")
    diagnose_minimal_examples()
    print("\nin-plane rotation sanity")
    diagnose_in_plane_rotation()
    print("\nprotein projection dependence")
    diagnose_protein_projection_dependence()


if __name__ == "__main__":
    main()
