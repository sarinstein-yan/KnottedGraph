from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

import certify_theta_symmetric_quandle as sq
import discover_yamada_theta_collisions as core

TARGETS = [
    ("pair13_left", (32, 58, 0.12), 0),
    ("pair13_right", (39, 153, 0.05), 32),
    ("pair16_left", (32, 197, 0.12), 0),
    ("pair16_right", (39, 102, 0.05), 32),
]

ROTATIONS = [
    (0.0, 0.0, 0.0),
    (0.173, 0.291, 0.137),
    (0.411, -0.227, 0.319),
    (-0.283, 0.367, 0.191),
    (0.619, 0.143, -0.257),
    (-0.337, -0.419, 0.523),
]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)


def rotate_graph(graph: nx.MultiGraph, angles):
    a, b, c = angles
    matrix = rot_z(c) @ rot_y(b) @ rot_x(a)
    out = graph.copy()
    for node, data in out.nodes(data=True):
        data["pos"] = (np.asarray(data["pos"], dtype=float) @ matrix.T).tolist()
    for u, v, key, data in out.edges(keys=True, data=True):
        data["pts"] = (np.asarray(data["pts"], dtype=float) @ matrix.T).tolist()
    return out


def trivial_theta():
    graph = nx.MultiGraph()
    u = np.array([-1.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    graph.add_node("u", pos=u.tolist())
    graph.add_node("v", pos=v.tolist())
    ys = [0.8, 0.0, -0.8]
    zs = [0.05, 0.0, -0.05]
    for role, (y, z) in enumerate(zip(ys, zs)):
        pts = np.array([
            u,
            [-0.5, y, z],
            [0.5, y, z],
            v,
        ], dtype=float)
        graph.add_edge("u", "v", role, pts=pts.tolist(), role=role)
    return graph


def count(graph):
    constraints = sq.diagram_constraints(graph)
    solutions, diagnostics = sq.solve_coloring(constraints)
    return {
        "count": len(solutions),
        "crossings": len(constraints["crossings"]),
        "variables": constraints["variable_count"],
        "vertex_allowed_sizes": diagnostics["vertex_allowed_sizes"],
    }


def run(plantri: str, output: Path):
    sq.verify_symmetric_quandle()
    shadows = {s.index: s for s in core.generate_shadows(plantri, 8)}
    records = []
    for label, desc, expected in TARGETS:
        shadow, bits, fraction = desc
        graph, _ = core.spatial_theta(shadows[shadow], bits, approach_fraction=fraction)
        trials = []
        for angles in ROTATIONS:
            trial = count(rotate_graph(graph, angles))
            trial["rotation"] = list(angles)
            trials.append(trial)
        observed = {trial["count"] for trial in trials}
        if observed != {expected}:
            raise AssertionError(f"{label}: projection-dependent counts {observed}, expected {expected}")
        records.append({"label": label, "expected": expected, "trials": trials})
        print("QUANDLE_INVARIANCE_TARGET=" + json.dumps(records[-1], sort_keys=True), flush=True)

    trivial_trials = []
    for angles in ROTATIONS:
        trial = count(rotate_graph(trivial_theta(), angles))
        trial["rotation"] = list(angles)
        trivial_trials.append(trial)
    trivial_counts = {trial["count"] for trial in trivial_trials}
    if len(trivial_counts) != 1:
        raise AssertionError(f"trivial theta projection-dependent counts: {trivial_counts}")
    # Jang--Oshiro Example 2.3 gives 2*4^2 = 32 for the standard theta curve.
    if trivial_counts != {32}:
        raise AssertionError(f"trivial theta count {trivial_counts}, expected published value 32")

    payload = {
        "validation": (
            "six generic projections per candidate preserve the exact symmetric-quandle "
            "coloring count; independently constructed standard theta reproduces the "
            "published Jang-Oshiro value 32"
        ),
        "targets": records,
        "trivial_theta": trivial_trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("QUANDLE_INVARIANCE_SUMMARY=" + json.dumps({
        "candidate_targets": len(records),
        "rotations_per_target": len(ROTATIONS),
        "trivial_theta_count": next(iter(trivial_counts)),
    }, sort_keys=True), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output)


if __name__ == "__main__":
    main()
