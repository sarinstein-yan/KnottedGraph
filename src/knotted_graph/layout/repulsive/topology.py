from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObjCurve:
    vertices: np.ndarray
    edges: tuple[tuple[int, int], ...]


def read_obj_curve(path: Path | str) -> ObjCurve:
    vertices: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("l "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Only two-vertex OBJ line elements are supported: {line.strip()}")
                edges.append((int(parts[1]) - 1, int(parts[2]) - 1))
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    if not edges:
        raise ValueError(f"No line edges found in {path}")
    return ObjCurve(np.asarray(vertices, dtype=float), tuple(edges))


def _segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
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
        t = float(np.clip(vw / vv, 0.0, 1.0))
        return float(np.linalg.norm(a - (c + t * v)))
    if vv <= 1e-20:
        s = float(np.clip(-uw / uu, 0.0, 1.0))
        return float(np.linalg.norm((a + s * u) - c))

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

    return float(np.linalg.norm((a + s * u) - (c + t * v)))


def _triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.dot(a, np.cross(b, c)))


def _coplanarity_cubic(
    start: np.ndarray,
    end: np.ndarray,
    left: tuple[int, int],
    right: tuple[int, int],
) -> np.ndarray:
    a, b = left
    c, d = right
    a0 = start[a]
    b0 = start[b]
    c0 = start[c]
    d0 = start[d]
    da = end[a] - start[a]
    db = end[b] - start[b]
    dc = end[c] - start[c]
    dd = end[d] - start[d]

    u0 = b0 - a0
    u1 = db - da
    v0 = d0 - c0
    v1 = dd - dc
    w0 = c0 - a0
    w1 = dc - da

    return np.asarray(
        [
            _triple(w0, u0, v0),
            _triple(w1, u0, v0) + _triple(w0, u1, v0) + _triple(w0, u0, v1),
            _triple(w1, u1, v0) + _triple(w1, u0, v1) + _triple(w0, u1, v1),
            _triple(w1, u1, v1),
        ],
        dtype=float,
    )


def _roots_in_unit_interval(coeffs: np.ndarray) -> list[float]:
    scale = float(np.max(np.abs(coeffs))) if coeffs.size else 0.0
    if scale <= 1e-14:
        return []

    trimmed = coeffs.copy()
    while len(trimmed) > 1 and abs(trimmed[-1]) <= max(1e-12 * scale, 1e-14):
        trimmed = trimmed[:-1]
    if len(trimmed) <= 1:
        return []

    roots = np.roots(trimmed[::-1])
    result: list[float] = []
    for root in roots:
        if abs(float(root.imag)) > 1e-7:
            continue
        value = float(root.real)
        if -1e-9 <= value <= 1.0 + 1e-9:
            value = float(np.clip(value, 0.0, 1.0))
            if not any(abs(value - existing) <= 1e-7 for existing in result):
                result.append(value)
    return sorted(result)


def _swept_aabb_separated(
    start: np.ndarray,
    end: np.ndarray,
    left: tuple[int, int],
    right: tuple[int, int],
    epsilon: float,
) -> bool:
    left_points = np.vstack([start[left[0]], start[left[1]], end[left[0]], end[left[1]]])
    right_points = np.vstack([start[right[0]], start[right[1]], end[right[0]], end[right[1]]])
    left_min = left_points.min(axis=0)
    left_max = left_points.max(axis=0)
    right_min = right_points.min(axis=0)
    right_max = right_points.max(axis=0)
    return bool(np.any(left_max + epsilon < right_min) or np.any(right_max + epsilon < left_min))


def _segment_at(vertices0: np.ndarray, vertices1: np.ndarray, edge: tuple[int, int], tau: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = vertices0 + tau * (vertices1 - vertices0)
    return vertices[edge[0]], vertices[edge[1]]


def _check_moving_pair(
    start: np.ndarray,
    end: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    left_index: int,
    right_index: int,
    epsilon: float,
) -> dict[str, Any] | None:
    left = edges[left_index]
    right = edges[right_index]
    if _swept_aabb_separated(start, end, left, right, epsilon):
        return None

    coeffs = _coplanarity_cubic(start, end, left, right)
    scale = float(np.max(np.abs(coeffs)))
    candidates = [0.0, 0.5, 1.0]
    degenerate = scale <= 1e-14
    if not degenerate:
        candidates.extend(_roots_in_unit_interval(coeffs))

    best = {
        "distance": float("inf"),
        "tau": None,
        "left_edge": left_index,
        "right_edge": right_index,
        "left_vertices": list(left),
        "right_vertices": list(right),
        "degenerate_coplanarity": degenerate,
    }
    for tau in sorted(set(round(float(t), 12) for t in candidates)):
        a, b = _segment_at(start, end, left, tau)
        c, d = _segment_at(start, end, right, tau)
        distance = _segment_distance(a, b, c, d)
        if distance < best["distance"]:
            best["distance"] = distance
            best["tau"] = tau

    best["unsafe"] = bool(best["distance"] <= epsilon or degenerate)
    return best


def verify_obj_transition(
    start_curve: ObjCurve,
    end_curve: ObjCurve,
    *,
    epsilon: float = 1e-7,
    transition_index: int | None = None,
) -> dict[str, Any]:
    if start_curve.vertices.shape != end_curve.vertices.shape:
        raise ValueError("Step vertex arrays have different shapes")
    if start_curve.edges != end_curve.edges:
        raise ValueError("Step edge lists differ")

    edges = start_curve.edges
    checked_pairs = 0
    violations: list[dict[str, Any]] = []
    min_checked_distance = float("inf")
    min_pair: dict[str, Any] | None = None
    for left_index, left in enumerate(edges):
        left_vertices = set(left)
        for right_index in range(left_index + 1, len(edges)):
            right = edges[right_index]
            if left_vertices.intersection(right):
                continue
            checked_pairs += 1
            pair_result = _check_moving_pair(
                start_curve.vertices,
                end_curve.vertices,
                edges,
                left_index,
                right_index,
                epsilon,
            )
            if pair_result is None:
                continue
            if transition_index is not None:
                pair_result["transition_index"] = transition_index
            if pair_result["distance"] < min_checked_distance:
                min_checked_distance = float(pair_result["distance"])
                min_pair = dict(pair_result)
            if pair_result["unsafe"]:
                violations.append(pair_result)

    return {
        "verified": not violations,
        "epsilon": epsilon,
        "checked_pairs": checked_pairs,
        "violations": violations,
        "violation_count": len(violations),
        "min_checked_distance": min_checked_distance if min_pair is not None else None,
        "min_pair": min_pair,
    }


def verify_obj_step_sequence(
    steps_dir: Path | str,
    *,
    epsilon: float = 1e-7,
    pattern: str = "step_*.obj",
    max_reported_violations: int = 20,
) -> dict[str, Any]:
    directory = Path(steps_dir)
    step_paths = sorted(directory.glob(pattern))
    if len(step_paths) < 2:
        raise ValueError(f"Need at least two step OBJ files in {directory}")

    curves = [read_obj_curve(path) for path in step_paths]
    total_pairs = 0
    violations: list[dict[str, Any]] = []
    total_violations = 0
    global_min: dict[str, Any] | None = None
    for index, (start, end) in enumerate(zip(curves, curves[1:])):
        result = verify_obj_transition(start, end, epsilon=epsilon, transition_index=index)
        total_pairs += int(result["checked_pairs"])
        if result["min_pair"] is not None:
            candidate = dict(result["min_pair"])
            candidate["from_step"] = step_paths[index].name
            candidate["to_step"] = step_paths[index + 1].name
            if global_min is None or candidate["distance"] < global_min["distance"]:
                global_min = candidate
        for violation in result["violations"]:
            total_violations += 1
            record = dict(violation)
            record["from_step"] = step_paths[index].name
            record["to_step"] = step_paths[index + 1].name
            if len(violations) < max_reported_violations:
                violations.append(record)

    return {
        "verified": total_violations == 0,
        "epsilon": epsilon,
        "steps_dir": str(directory),
        "step_count": len(step_paths),
        "transition_count": len(step_paths) - 1,
        "checked_pairs": total_pairs,
        "violation_count": total_violations,
        "reported_violation_count": len(violations),
        "violations": violations,
        "min_checked_distance": global_min["distance"] if global_min is not None else None,
        "min_pair": global_min,
        "note": "Independent swept centerline check over saved OBJ steps; verified up to floating-point tolerance.",
    }
