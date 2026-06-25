from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import CurveNetwork


DEFAULT_ARC_COLORS = ("#c33a2f", "#2c67c7", "#2e9f55", "#d8d8d2")
DEFAULT_NODE_COLORS = ("#c6a21f", "#1e4ca0", "#16a34a", "#a12bbd")


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def arbitrary_normal(tangent: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(ref, tangent))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    return normalize(np.cross(tangent, ref))


def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    out = np.asarray(points, dtype=float)
    for _ in range(iterations):
        if len(out) < 3:
            break
        new_pts = [out[0]]
        for i in range(len(out) - 1):
            p = out[i]
            q = out[i + 1]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        new_pts.append(out[-1])
        out = np.asarray(new_pts, dtype=float)
    return out


def polyline_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(points)
    tangents = np.zeros((n, 3), dtype=float)
    for i in range(n):
        if i == 0:
            tangents[i] = normalize(points[1] - points[0])
        elif i == n - 1:
            tangents[i] = normalize(points[-1] - points[-2])
        else:
            tangents[i] = normalize(points[i + 1] - points[i - 1])

    normals = np.zeros((n, 3), dtype=float)
    binormals = np.zeros((n, 3), dtype=float)
    normals[0] = arbitrary_normal(tangents[0])
    binormals[0] = normalize(np.cross(tangents[0], normals[0]))

    for i in range(1, n):
        v = tangents[i - 1]
        w = tangents[i]
        axis = np.cross(v, w)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            normal = normals[i - 1] - np.dot(normals[i - 1], w) * w
            if np.linalg.norm(normal) < 1e-8:
                normal = arbitrary_normal(w)
            normals[i] = normalize(normal)
        else:
            angle = math.atan2(axis_norm, float(np.dot(v, w)))
            normals[i] = normalize(rotate_about_axis(normals[i - 1], axis, angle))
            normals[i] = normalize(normals[i] - np.dot(normals[i], w) * w)
        binormals[i] = normalize(np.cross(w, normals[i]))

    return tangents, normals, binormals


def build_tube_mesh(
    points: np.ndarray,
    radius: float,
    sides: int = 14,
    smooth_iters: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = chaikin(points, iterations=smooth_iters)
    _, normals, binormals = polyline_frames(pts)
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)

    verts: list[np.ndarray] = []
    faces_i: list[int] = []
    faces_j: list[int] = []
    faces_k: list[int] = []

    for idx, p in enumerate(pts):
        n = normals[idx]
        b = binormals[idx]
        for theta in angles:
            verts.append(p + radius * (math.cos(theta) * n + math.sin(theta) * b))

    for r in range(len(pts) - 1):
        base0 = r * sides
        base1 = (r + 1) * sides
        for s in range(sides):
            a = base0 + s
            b = base0 + (s + 1) % sides
            c = base1 + s
            d = base1 + (s + 1) % sides
            faces_i.extend([a, b])
            faces_j.extend([c, c])
            faces_k.extend([b, d])

    return np.asarray(verts, dtype=float), np.asarray(faces_i), np.asarray(faces_j), np.asarray(faces_k)


def build_sphere_mesh(
    center: np.ndarray,
    radius: float,
    n_lat: int = 12,
    n_lon: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts: list[np.ndarray] = []
    for i in range(n_lat + 1):
        phi = math.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * math.pi * j / n_lon
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            verts.append(center + np.array([x, y, z], dtype=float))

    faces_i: list[int] = []
    faces_j: list[int] = []
    faces_k: list[int] = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            if i != 0:
                faces_i.append(a)
                faces_j.append(c)
                faces_k.append(b)
            if i != n_lat - 1:
                faces_i.append(b)
                faces_j.append(c)
                faces_k.append(d)

    return np.asarray(verts, dtype=float), np.asarray(faces_i), np.asarray(faces_j), np.asarray(faces_k)


def _points_bbox(points: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_points = np.concatenate([np.asarray(p, dtype=float) for p in points], axis=0)
    return all_points.min(axis=0), all_points.max(axis=0)


def render_tube_html(network: CurveNetwork, output: Path, title: str | None = None) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for HTML tube rendering. Install knotted_graph[repulsion].") from exc

    bbox_min, bbox_max = _points_bbox(network.arc_polylines[name] for name in network.arc_order)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    tube_radius = max(diag * 0.02, 0.8)
    sphere_radius = max(diag * 0.03, 1.2)

    fig = go.Figure()
    for idx, arc_name in enumerate(network.arc_order):
        color = network.arc_colors.get(arc_name, DEFAULT_ARC_COLORS[idx % len(DEFAULT_ARC_COLORS)])
        verts, i, j, k = build_tube_mesh(
            network.arc_polylines[arc_name],
            radius=tube_radius,
            sides=16,
            smooth_iters=2,
        )
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i,
                j=j,
                k=k,
                color=color,
                flatshading=False,
                lighting=dict(ambient=0.35, diffuse=0.95, specular=1.05, roughness=0.22, fresnel=0.12),
                lightposition=dict(x=120, y=80, z=180),
                hovertext=f"{arc_name}: {network.arc_specs.get(arc_name, '')}",
                hoverinfo="text",
                name=arc_name,
                showscale=False,
            )
        )

    for idx, node_name in enumerate(network.node_order):
        color = network.node_colors.get(node_name, DEFAULT_NODE_COLORS[idx % len(DEFAULT_NODE_COLORS)])
        verts, i, j, k = build_sphere_mesh(network.node_positions[node_name], sphere_radius)
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i,
                j=j,
                k=k,
                color=color,
                flatshading=False,
                lighting=dict(ambient=0.3, diffuse=0.95, specular=1.2, roughness=0.14, fresnel=0.15),
                lightposition=dict(x=120, y=80, z=180),
                hovertext=node_name,
                hoverinfo="text",
                name=node_name,
                showscale=False,
            )
        )

    label_points = np.vstack([network.node_positions[name] for name in network.node_order])
    fig.add_trace(
        go.Scatter3d(
            x=label_points[:, 0],
            y=label_points[:, 1],
            z=label_points[:, 2],
            mode="text",
            text=list(network.node_order),
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title or network.name,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.1, z=0.8)),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.75)"),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True)
