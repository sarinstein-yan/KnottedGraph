from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import clearance_report, point_segment_distance


@dataclass(frozen=True)
class DecimationOptions:
    """Conservative shortcut simplification for embedded curve networks."""

    max_passes: int = 8
    min_points_per_edge: int = 2
    max_points_per_edge: int | dict[str, int] | None = None
    clearance_fraction: float = 0.02
    min_clearance: float = 1e-5
    max_deviation: float | None = None
    preserve_pinned_neighbors: bool = True


@dataclass(frozen=True)
class DecimationResult:
    vertices: np.ndarray
    edge_indices: dict[str, list[int]]
    old_to_new: dict[int, int]
    report: dict[str, Any]


def _segment_segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    eps = 1e-20
    u = b - a
    v = d - c
    w = a - c
    aa = float(np.dot(u, u))
    bb = float(np.dot(u, v))
    cc = float(np.dot(v, v))
    dd = float(np.dot(u, w))
    ee = float(np.dot(v, w))

    if aa <= eps and cc <= eps:
        return float(np.linalg.norm(a - c))
    if aa <= eps:
        t = float(np.clip(ee / cc, 0.0, 1.0))
        return float(np.linalg.norm(a - (c + t * v)))
    if cc <= eps:
        s = float(np.clip(-dd / aa, 0.0, 1.0))
        return float(np.linalg.norm((a + s * u) - c))

    denom = aa * cc - bb * bb
    if denom != 0.0:
        s = float(np.clip((bb * ee - cc * dd) / denom, 0.0, 1.0))
    else:
        s = 0.0

    tnom = bb * s + ee
    if tnom < 0.0:
        t = 0.0
        s = float(np.clip(-dd / aa, 0.0, 1.0))
    elif tnom > cc:
        t = 1.0
        s = float(np.clip((bb - dd) / aa, 0.0, 1.0))
    else:
        t = tnom / cc

    return float(np.linalg.norm((a + s * u) - (c + t * v)))


def _point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    if float(np.dot(normal, normal)) <= 1e-20:
        return min(
            point_segment_distance(p, a, b),
            point_segment_distance(p, b, c),
            point_segment_distance(p, c, a),
        )

    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (a + v * ab)))

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (a + w * ac)))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + w * (c - b))))

    normal = normal / np.linalg.norm(normal)
    return abs(float(np.dot(p - a, normal)))


def _segment_intersects_triangle(
    p: np.ndarray,
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    direction = q - p
    edge1 = b - a
    edge2 = c - a
    h = np.cross(direction, edge2)
    det = float(np.dot(edge1, h))
    eps = 1e-12
    if abs(det) <= eps:
        return False

    inv_det = 1.0 / det
    s = p - a
    u = inv_det * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps:
        return False

    qvec = np.cross(s, edge1)
    v = inv_det * float(np.dot(direction, qvec))
    if v < -eps or u + v > 1.0 + eps:
        return False

    t = inv_det * float(np.dot(edge2, qvec))
    return -eps <= t <= 1.0 + eps


def _segment_triangle_distance(
    p: np.ndarray,
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    if _segment_intersects_triangle(p, q, a, b, c):
        return 0.0
    return min(
        _point_triangle_distance(p, a, b, c),
        _point_triangle_distance(q, a, b, c),
        _segment_segment_distance(p, q, a, b),
        _segment_segment_distance(p, q, b, c),
        _segment_segment_distance(p, q, c, a),
    )


def _segments(
    vertices: np.ndarray,
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
) -> list[tuple[str, int, int, int]]:
    segments: list[tuple[str, int, int, int]] = []
    for edge_id in edge_order:
        indices = edge_indices[edge_id]
        for local_index, (a, b) in enumerate(zip(indices, indices[1:])):
            segments.append((edge_id, local_index, int(a), int(b)))
    return segments


def _clearance_threshold(
    vertices: np.ndarray,
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
    options: DecimationOptions,
) -> tuple[float, dict[str, Any]]:
    report = clearance_report(vertices, edge_indices, edge_order)
    clearance = float(report["min_non_adjacent_segment_distance"])
    if math.isfinite(clearance):
        threshold = max(options.min_clearance, options.clearance_fraction * clearance)
    else:
        threshold = options.min_clearance
    return threshold, report


def _candidate_deviation(vertices: np.ndarray, indices: list[int], local_pos: int) -> float:
    a = vertices[indices[local_pos - 1]]
    b = vertices[indices[local_pos]]
    c = vertices[indices[local_pos + 1]]
    return point_segment_distance(b, a, c)


def _edge_target_point_count(edge_id: str, options: DecimationOptions) -> int | None:
    if options.max_points_per_edge is None:
        return None
    if isinstance(options.max_points_per_edge, dict):
        value = options.max_points_per_edge.get(edge_id)
        return int(value) if value is not None else None
    return int(options.max_points_per_edge)


def _edge_stop_count(edge_id: str, current_count: int, options: DecimationOptions) -> int:
    if isinstance(options.max_points_per_edge, dict) and edge_id not in options.max_points_per_edge:
        return current_count
    target_count = _edge_target_point_count(edge_id, options)
    return max(options.min_points_per_edge, target_count or options.min_points_per_edge)


def _shortcut_is_safe(
    vertices: np.ndarray,
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
    edge_id: str,
    local_pos: int,
    min_clearance: float,
) -> bool:
    indices = edge_indices[edge_id]
    a_idx = int(indices[local_pos - 1])
    b_idx = int(indices[local_pos])
    c_idx = int(indices[local_pos + 1])
    a = vertices[a_idx]
    b = vertices[b_idx]
    c = vertices[c_idx]

    if float(np.linalg.norm(c - a)) <= 1e-12:
        return False

    protected = {a_idx, b_idx, c_idx}
    for _, _, s_idx, t_idx in _segments(vertices, edge_indices, edge_order):
        if protected.intersection({s_idx, t_idx}):
            continue
        s = vertices[s_idx]
        t = vertices[t_idx]
        if _segment_segment_distance(a, c, s, t) <= min_clearance:
            return False
        if _segment_triangle_distance(s, t, a, b, c) <= min_clearance:
            return False
    return True


def decimate_curve_network(
    vertices: np.ndarray,
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
    *,
    pinned_indices: set[int] | None = None,
    options: DecimationOptions | None = None,
) -> DecimationResult:
    """Delete curve-internal points only when the shortcut passes safety checks."""

    options = options or DecimationOptions()
    if options.max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    if options.min_points_per_edge < 2:
        raise ValueError("min_points_per_edge must be at least 2")
    if isinstance(options.max_points_per_edge, int) and options.max_points_per_edge < options.min_points_per_edge:
        raise ValueError("max_points_per_edge must be greater than or equal to min_points_per_edge")
    if isinstance(options.max_points_per_edge, dict):
        for edge_id, count in options.max_points_per_edge.items():
            if int(count) < options.min_points_per_edge:
                raise ValueError(
                    f"max_points_per_edge for {edge_id!r} must be greater than or equal to min_points_per_edge"
                )

    vertices = np.asarray(vertices, dtype=float)
    pinned = {int(i) for i in (pinned_indices or set())}
    working = {edge_id: list(map(int, edge_indices[edge_id])) for edge_id in edge_order}

    min_clearance, initial_clearance = _clearance_threshold(vertices, working, edge_order, options)
    before_counts = {edge_id: len(working[edge_id]) for edge_id in edge_order}
    removed = 0
    accepted_shortcuts = 0
    passes = 0

    for pass_index in range(options.max_passes):
        passes = pass_index + 1
        candidates: list[tuple[float, str, int, int]] = []
        for edge_id in edge_order:
            indices = working[edge_id]
            stop_count = _edge_stop_count(edge_id, len(indices), options)
            if len(indices) <= stop_count:
                continue
            for local_pos in range(1, len(indices) - 1):
                vertex_index = int(indices[local_pos])
                if vertex_index in pinned:
                    continue
                if options.preserve_pinned_neighbors:
                    if local_pos == 1 and int(indices[0]) in pinned:
                        continue
                    if local_pos == len(indices) - 2 and int(indices[-1]) in pinned:
                        continue
                deviation = _candidate_deviation(vertices, indices, local_pos)
                if options.max_deviation is not None and deviation > options.max_deviation:
                    continue
                candidates.append((deviation, edge_id, local_pos, vertex_index))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        removed_this_pass = 0
        for _, edge_id, local_pos, vertex_index in candidates:
            indices = working[edge_id]
            stop_count = _edge_stop_count(edge_id, len(indices), options)
            if len(indices) <= stop_count:
                continue
            if local_pos <= 0 or local_pos >= len(indices) - 1:
                continue
            if int(indices[local_pos]) != vertex_index:
                continue
            if options.preserve_pinned_neighbors:
                if local_pos == 1 and int(indices[0]) in pinned:
                    continue
                if local_pos == len(indices) - 2 and int(indices[-1]) in pinned:
                    continue
            if not _shortcut_is_safe(vertices, working, edge_order, edge_id, local_pos, min_clearance):
                continue

            del indices[local_pos]
            removed += 1
            removed_this_pass += 1
            accepted_shortcuts += 1

        if removed_this_pass == 0:
            break

    used = sorted({int(index) for edge_id in edge_order for index in working[edge_id]})
    old_to_new = {old: new for new, old in enumerate(used)}
    compacted_vertices = vertices[np.asarray(used, dtype=int)].copy()
    compacted_edges = {
        edge_id: [old_to_new[int(index)] for index in working[edge_id]]
        for edge_id in edge_order
    }
    final_clearance = clearance_report(compacted_vertices, compacted_edges, edge_order)

    report = {
        "initial_vertex_count": int(len(vertices)),
        "final_vertex_count": int(len(compacted_vertices)),
        "removed_vertices": int(removed),
        "accepted_shortcuts": int(accepted_shortcuts),
        "passes": int(passes),
        "min_clearance_threshold": float(min_clearance),
        "initial_clearance": initial_clearance,
        "final_clearance": final_clearance,
        "edge_point_counts_before": before_counts,
        "edge_point_counts_after": {edge_id: len(compacted_edges[edge_id]) for edge_id in edge_order},
    }
    return DecimationResult(
        vertices=compacted_vertices,
        edge_indices=compacted_edges,
        old_to_new=old_to_new,
        report=report,
    )
