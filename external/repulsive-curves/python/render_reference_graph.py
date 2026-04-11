import argparse
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


PALETTE = [
    "#b22222",
    "#1f5aa6",
    "#1f9d55",
    "#c9a227",
    "#7a2ca2",
    "#d95f02",
    "#008b8b",
    "#c05195",
]


def load_curve_obj(path: Path):
    vertices = []
    edges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith("l "):
            _, a, b = line.split()
            edges.append((int(a) - 1, int(b) - 1))
    return np.asarray(vertices, dtype=float), edges


def build_adjacency(n_vertices, edges):
    adj = [[] for _ in range(n_vertices)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def extract_paths(vertices, edges):
    adj = build_adjacency(len(vertices), edges)
    special = {i for i, nbrs in enumerate(adj) if len(nbrs) != 2}
    visited = set()
    paths = []

    for start in sorted(special):
        for nxt in adj[start]:
            edge_key = tuple(sorted((start, nxt)))
            if edge_key in visited:
                continue

            path = [start, nxt]
            visited.add(edge_key)
            prev = start
            cur = nxt

            while cur not in special:
                candidates = [u for u in adj[cur] if u != prev]
                if not candidates:
                    break
                nxt2 = candidates[0]
                edge_key = tuple(sorted((cur, nxt2)))
                if edge_key in visited:
                    break
                path.append(nxt2)
                visited.add(edge_key)
                prev, cur = cur, nxt2

            paths.append(path)

    special_positions = vertices[sorted(special)]
    return paths, sorted(special), special_positions


def make_sphere(center, radius=0.055, nu=18, nv=12):
    u = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    v = np.linspace(0, math.pi, nv)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(uu) * np.sin(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(vv)

    xs = x.ravel()
    ys = y.ravel()
    zs = z.ravel()
    i_idx, j_idx, k_idx = [], [], []
    for row in range(nv - 1):
        for col in range(nu):
            c0 = row * nu + col
            c1 = row * nu + (col + 1) % nu
            c2 = (row + 1) * nu + col
            c3 = (row + 1) * nu + (col + 1) % nu
            i_idx.extend([c0, c1])
            j_idx.extend([c2, c2])
            k_idx.extend([c1, c3])
    return xs, ys, zs, i_idx, j_idx, k_idx


def tube_mesh(polyline, radius=0.03, sides=14):
    pts = np.asarray(polyline, dtype=float)
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)

    normals = np.zeros_like(pts)
    binormals = np.zeros_like(pts)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, tangents[0])) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    normals[0] = np.cross(tangents[0], ref)
    normals[0] /= max(np.linalg.norm(normals[0]), 1e-12)
    binormals[0] = np.cross(tangents[0], normals[0])
    binormals[0] /= max(np.linalg.norm(binormals[0]), 1e-12)

    for idx in range(1, len(pts)):
        n = np.cross(binormals[idx - 1], tangents[idx])
        if np.linalg.norm(n) < 1e-12:
            n = normals[idx - 1]
        normals[idx] = n / max(np.linalg.norm(n), 1e-12)
        b = np.cross(tangents[idx], normals[idx])
        binormals[idx] = b / max(np.linalg.norm(b), 1e-12)

    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    rings = []
    for p, n, b in zip(pts, normals, binormals):
        ring = [p + radius * (math.cos(a) * n + math.sin(a) * b) for a in angles]
        rings.append(np.asarray(ring))
    rings = np.asarray(rings)

    x = rings[:, :, 0].ravel()
    y = rings[:, :, 1].ravel()
    z = rings[:, :, 2].ravel()

    ii, jj, kk = [], [], []
    for r in range(len(pts) - 1):
        for s in range(sides):
            a = r * sides + s
            b = r * sides + (s + 1) % sides
            c = (r + 1) * sides + s
            d = (r + 1) * sides + (s + 1) % sides
            ii.extend([a, b])
            jj.extend([c, c])
            kk.extend([b, d])
    return x, y, z, ii, jj, kk


def render_reference(obj_path: Path, output_html: Path, output_png: Path | None = None):
    vertices, edges = load_curve_obj(obj_path)
    paths, special_nodes, special_positions = extract_paths(vertices, edges)

    fig = go.Figure()

    for path in paths:
        poly = vertices[path]
        x, y, z, i_idx, j_idx, k_idx = tube_mesh(poly, radius=0.028, sides=16)
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color="#c8c8c8",
                opacity=1.0,
                flatshading=False,
                lighting={"ambient": 0.55, "diffuse": 0.85, "specular": 0.25, "roughness": 0.7},
                showscale=False,
                hoverinfo="skip",
            )
        )

    for idx, center in enumerate(special_positions):
        x, y, z, i_idx, j_idx, k_idx = make_sphere(center, radius=0.06)
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color=PALETTE[idx % len(PALETTE)],
                opacity=1.0,
                flatshading=False,
                lighting={"ambient": 0.45, "diffuse": 0.95, "specular": 0.35, "roughness": 0.55},
                showscale=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.45, "y": 1.35, "z": 0.95},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print("Wrote:", output_html)

    if output_png is not None:
        try:
            fig.write_image(str(output_png), width=1100, height=900, scale=2)
            print("Wrote:", output_png)
        except Exception as exc:
            print("PNG export skipped:", exc)


def main():
    parser = argparse.ArgumentParser(description="Render the official graph drawing OBJ as tube geometry.")
    parser.add_argument("obj", type=Path)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    render_reference(args.obj, args.html, args.png)


if __name__ == "__main__":
    main()
