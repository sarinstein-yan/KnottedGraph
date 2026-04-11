import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go


PALETTES = {
    "default": None,
    "k5": ["#b59a18", "#275fb2", "#8b2ca8", "#b1222e", "#22a84b"],
    "k33": ["#b1222e", "#b1222e", "#b1222e", "#275fb2", "#275fb2", "#275fb2"],
}


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


def polyline_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_tube_mesh(points: np.ndarray, radius: float, sides: int = 14, smooth_iters: int = 2):
    pts = chaikin(points, iterations=smooth_iters)
    _, normals, binormals = polyline_frames(pts)
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)

    verts: List[np.ndarray] = []
    faces_i: List[int] = []
    faces_j: List[int] = []
    faces_k: List[int] = []

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


def build_sphere_mesh(center: np.ndarray, radius: float, n_lat: int = 12, n_lon: int = 20):
    verts: List[np.ndarray] = []
    for i in range(n_lat + 1):
        phi = math.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * math.pi * j / n_lon
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            verts.append(center + np.array([x, y, z], dtype=float))

    faces_i: List[int] = []
    faces_j: List[int] = []
    faces_k: List[int] = []
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


def merge_meshes(meshes: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    verts_all: List[np.ndarray] = []
    i_all: List[np.ndarray] = []
    j_all: List[np.ndarray] = []
    k_all: List[np.ndarray] = []
    offset = 0
    for verts, i, j, k in meshes:
        if len(verts) == 0:
            continue
        verts_all.append(verts)
        i_all.append(i + offset)
        j_all.append(j + offset)
        k_all.append(k + offset)
        offset += len(verts)
    return (
        np.concatenate(verts_all, axis=0),
        np.concatenate(i_all, axis=0),
        np.concatenate(j_all, axis=0),
        np.concatenate(k_all, axis=0),
    )


def palette_for(node_order: Sequence[str], palette_name: str) -> List[str]:
    palette = PALETTES[palette_name]
    if palette is None:
        import plotly.express as px

        base = px.colors.qualitative.Bold
        return [base[i % len(base)] for i in range(len(node_order))]
    if len(palette) < len(node_order):
        return [palette[i % len(palette)] for i in range(len(node_order))]
    return list(palette[: len(node_order)])


def maybe_write_png(fig: go.Figure, output: Path) -> bool:
    if importlib.util.find_spec("kaleido") is None:
        return False
    fig.write_image(str(output))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render layout.json with tube/sphere materials using Plotly meshes.")
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Output .html or .png. Default is .html")
    parser.add_argument("--palette", choices=list(PALETTES.keys()), default="default")
    parser.add_argument("--tube-radius", type=float, default=0.055)
    parser.add_argument("--sphere-radius", type=float, default=0.10)
    parser.add_argument("--tube-sides", type=int, default=16)
    parser.add_argument("--smooth-iters", type=int, default=2)
    parser.add_argument("--eye-x", type=float, default=1.45)
    parser.add_argument("--eye-y", type=float, default=1.10)
    parser.add_argument("--eye-z", type=float, default=0.85)
    args = parser.parse_args()

    data = json.loads(args.layout_json.read_text(encoding="utf-8"))
    node_order = data["node_order"]
    node_positions = {k: np.asarray(v, dtype=float) for k, v in data["node_positions_final"].items()}
    edge_polylines = {k: np.asarray(v, dtype=float) for k, v in data["edge_polylines_final"].items()}

    tube_meshes = []
    for poly in edge_polylines.values():
        tube_meshes.append(
            build_tube_mesh(poly, radius=args.tube_radius, sides=args.tube_sides, smooth_iters=args.smooth_iters)
        )
    tube_v, tube_i, tube_j, tube_k = merge_meshes(tube_meshes)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=tube_v[:, 0],
            y=tube_v[:, 1],
            z=tube_v[:, 2],
            i=tube_i,
            j=tube_j,
            k=tube_k,
            color="#c7c7c7",
            flatshading=False,
            lighting=dict(ambient=0.35, diffuse=0.9, specular=1.0, roughness=0.28, fresnel=0.08),
            lightposition=dict(x=120, y=80, z=180),
            hoverinfo="skip",
            name="edges",
            showscale=False,
        )
    )

    colors = palette_for(node_order, args.palette)
    for node, color in zip(node_order, colors):
        sphere_v, sphere_i, sphere_j, sphere_k = build_sphere_mesh(node_positions[node], args.sphere_radius)
        fig.add_trace(
            go.Mesh3d(
                x=sphere_v[:, 0],
                y=sphere_v[:, 1],
                z=sphere_v[:, 2],
                i=sphere_i,
                j=sphere_j,
                k=sphere_k,
                color=color,
                flatshading=False,
                lighting=dict(ambient=0.3, diffuse=0.95, specular=1.2, roughness=0.15, fresnel=0.15),
                lightposition=dict(x=120, y=80, z=180),
                hovertext=node,
                hoverinfo="text",
                name=node,
                showscale=False,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=args.eye_x, y=args.eye_y, z=args.eye_z)),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    output = args.output or args.layout_json.with_name(args.layout_json.stem + "_material.html")
    if output.suffix.lower() == ".png":
        ok = maybe_write_png(fig, output)
        if not ok:
            html_fallback = output.with_suffix(".html")
            fig.write_html(str(html_fallback), include_plotlyjs=True)
            raise RuntimeError(f"kaleido not installed; wrote HTML instead: {html_fallback}")
    else:
        fig.write_html(str(output), include_plotlyjs=True)

    print("Wrote:", output)


if __name__ == "__main__":
    main()
