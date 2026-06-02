from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def node_distance(node_positions: dict[str, np.ndarray], node_order: tuple[str, ...]) -> float:
    return float(np.linalg.norm(node_positions[node_order[1]] - node_positions[node_order[0]]))


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-20:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def segment_segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    # Static sanity metric only. The C++ driver performs the per-step swept topology check.
    u = b - a
    v = d - c
    w = a - c
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    uv = float(np.dot(u, v))
    uw = float(np.dot(u, w))
    vw = float(np.dot(v, w))

    if uu <= 1e-20 and vv <= 1e-20:
        return float(np.linalg.norm(a - c))
    if uu <= 1e-20:
        return point_segment_distance(a, c, d)
    if vv <= 1e-20:
        return point_segment_distance(c, a, b)

    denom = uu * vv - uv * uv
    if denom > 1e-20:
        s = float(np.clip((uv * vw - vv * uw) / denom, 0.0, 1.0))
    else:
        s = 0.0

    t = (uv * s + vw) / vv
    if t < 0.0:
        t = 0.0
        s = float(np.clip(-uw / uu, 0.0, 1.0))
    elif t > 1.0:
        t = 1.0
        s = float(np.clip((uv - uw) / uu, 0.0, 1.0))

    closest_left = a + s * u
    closest_right = c + t * v
    return float(np.linalg.norm(closest_left - closest_right))


def clearance_report(
    vertices: np.ndarray,
    arc_indices: dict[str, list[int]],
    arc_order: tuple[str, ...],
) -> dict[str, Any]:
    segments = []
    for arc_name in arc_order:
        indices = arc_indices[arc_name]
        for local_index, (a, b) in enumerate(zip(indices, indices[1:])):
            segments.append(
                {
                    "arc": arc_name,
                    "local_index": local_index,
                    "a": int(a),
                    "b": int(b),
                    "p": vertices[int(a)],
                    "q": vertices[int(b)],
                }
            )

    best = math.inf
    best_pair = None
    for i, left in enumerate(segments):
        left_vertices = {left["a"], left["b"]}
        for right in segments[i + 1 :]:
            if left_vertices.intersection({right["a"], right["b"]}):
                continue
            dist = segment_segment_distance(left["p"], left["q"], right["p"], right["q"])
            if dist < best:
                best = dist
                best_pair = {
                    "left": {
                        "arc": left["arc"],
                        "local_index": left["local_index"],
                        "vertices": [left["a"], left["b"]],
                    },
                    "right": {
                        "arc": right["arc"],
                        "local_index": right["local_index"],
                        "vertices": [right["a"], right["b"]],
                    },
                }

    return {
        "min_non_adjacent_segment_distance": best,
        "min_pair": best_pair,
        "segment_count": len(segments),
    }


def read_certificate(history_csv: Path) -> dict[str, Any]:
    rows = []
    with history_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    accepted = [row for row in rows if row["accepted"] == "1"]
    margins = [float(row["margin"]) for row in accepted]
    safe_ratios = [
        float(row["step_size"]) / float(row["safe_t"])
        for row in accepted
        if float(row["safe_t"]) > 0
    ]
    valid = all(float(row["step_size"]) < float(row["safe_t"]) for row in accepted)
    topology_rows = [row for row in accepted if row.get("topology_enabled") == "1"]
    topology_valid = all(row.get("topology_safe") == "1" for row in topology_rows)
    topology_min_distances = [
        float(row["topology_min_distance"])
        for row in topology_rows
        if row.get("topology_min_distance") not in (None, "", "-1")
    ]
    topology_rejections = sum(int(row.get("topology_rejections", "0")) for row in rows)
    return {
        "valid": valid and topology_valid,
        "repulsor_safe_step_valid": valid,
        "swept_topology_valid": topology_valid,
        "accepted_steps": len(accepted),
        "rows": len(rows),
        "min_margin": min(margins) if margins else None,
        "max_step_to_safe_ratio": max(safe_ratios) if safe_ratios else None,
        "min_swept_topology_sample_distance": min(topology_min_distances) if topology_min_distances else None,
        "topology_rejections": topology_rejections,
    }


def read_history_summary(history_csv: Path) -> dict[str, Any]:
    rows = []
    with history_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    accepted = [row for row in rows if row["accepted"] == "1"]
    return {
        "accepted_steps": len(accepted),
        "rows": len(rows),
        "energy_initial": float(accepted[0]["energy_before"]) if accepted else None,
        "energy_final": float(accepted[-1]["energy_after"]) if accepted else None,
        "topology_rejections": sum(int(row.get("topology_rejections", "0")) for row in rows),
    }
