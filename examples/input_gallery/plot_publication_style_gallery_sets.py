"""Additional publication-style Task 2 gallery sets.

This script generates three extra 3x3 galleries.  Each panel has a converted
KnottedGraph-compatible object on the left and a compact source-domain view on
the right.  Outputs use new filenames and do not overwrite the main Task 2
gallery figures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
ROOT = EXAMPLES_DIR.parent
DATA_DIR = HERE / "data" / "gallery_sets"
FIGURE_DIR = HERE / "figures"
SET_PANEL_DIR = FIGURE_DIR / "publication_style_extra_sets"

for relative in [
    "coordinate_chains",
    "spatial_graphs",
    "surfaces",
    "volumetric_fields",
]:
    sys.path.insert(0, str(EXAMPLES_DIR / relative))

from coordinate_curve_adapter import build_curve_from_xyz, write_xyz_coords
from mesh_surface_adapter import build_surface_from_mesh_file
from spatial_graph_adapter import build_spatial_graph_from_csv, write_spatial_graph_csv
from volumetric_field_adapter import build_surface_from_scalar_field_file, write_npz_scalar_field
from knotted_graph.inputs import (
    from_gromacs_gro,
    from_lammps_dump,
    from_mmcif_backbone,
    from_nucleic_acid_backbone,
    from_protein_ca_backbone,
    write_gro_coords,
    write_lammps_dump,
)

from plot_publication_style_gallery import (
    EDGE_COLOR,
    EDGE_COLOR_2,
    FERMI_COLOR,
    NODE_COLOR,
    SURFACE_COLOR,
    VOLUME_COLOR,
    add_curve,
    add_point_cloud,
    add_raw_spatial_graph,
    add_source_backbone_trace,
    add_source_beads_and_bonds,
    add_spatial_graph,
    add_surface,
    crop_white,
    edge_points,
    graph_points,
    load_mmcif_atom_points,
    load_pdb_atom_points,
    make_plotter,
    point_span,
    polyline_mesh,
    save_image_array,
    set_camera,
)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "figure.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SET_PANEL_DIR.mkdir(parents=True, exist_ok=True)


def render_scene(draw_fn, image_path: Path) -> tuple[np.ndarray, Path, list[str]]:
    plotter = make_plotter()
    result = draw_fn(plotter)
    if len(result) == 2:
        meshes, points = result
        issues: list[str] = []
    else:
        meshes, points, issues = result
    set_camera(plotter, points)
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    cropped = crop_white(image, threshold=249, pad=10)
    save_image_array(cropped, image_path)
    return cropped, image_path, issues


def render_source(draw_fn, image_path: Path) -> tuple[np.ndarray, Path]:
    plotter = make_plotter(window_size=(900, 760))
    meshes, points = draw_fn(plotter)
    set_camera(plotter, points)
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    cropped = crop_white(image, threshold=249, pad=8)
    save_image_array(cropped, image_path)
    return cropped, image_path


def render_scalar_slices(values: np.ndarray, image_path: Path) -> tuple[np.ndarray, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.35), dpi=240)
    slices = [
        values[:, :, values.shape[2] // 2].T,
        values[:, values.shape[1] // 2, :].T,
        values[values.shape[0] // 2, :, :].T,
    ]
    for ax, slice_data in zip(axes, slices):
        ax.imshow(slice_data, cmap="Greys", origin="lower")
        ax.contour(slice_data, levels=[0.0], colors=[EDGE_COLOR], linewidths=0.65, origin="lower")
        ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    save_image_array(image, image_path)
    return image, image_path


def assemble_gallery(panels: list[dict], *, output_stem: str) -> tuple[Path, Path, Path]:
    labels = [f"({chr(ord('a') + i)})" for i in range(len(panels))]
    fig = plt.figure(figsize=(17.5, 14.0), facecolor="white")
    outer = fig.add_gridspec(
        3,
        3,
        left=0.012,
        right=0.988,
        top=0.985,
        bottom=0.010,
        wspace=0.020,
        hspace=0.085,
    )
    for i, panel in enumerate(panels):
        r = i // 3
        c = i % 3
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[0.12, 1.0], hspace=0.04)
        title_ax = fig.add_subplot(inner[0, 0])
        body = inner[1, 0].subgridspec(1, 2, width_ratios=[0.78, 0.22], wspace=0.02)
        image_ax = fig.add_subplot(body[0, 0])
        source_ax = fig.add_subplot(body[0, 1])
        for ax in (title_ax, image_ax, source_ax):
            ax.axis("off")
        title_ax.text(
            0.00,
            0.52,
            labels[i],
            transform=title_ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="center",
        )
        title_ax.text(
            0.53,
            0.52,
            panel["title"],
            transform=title_ax.transAxes,
            fontsize=15.5,
            fontweight="semibold",
            ha="center",
            va="center",
        )
        image_ax.imshow(panel["image"])
        image_ax.set_box_aspect(1)
        source_ax.imshow(panel["source_image"])
        source_ax.set_box_aspect(1)
        for spine in source_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.75)
        source_ax.text(
            0.07,
            0.92,
            "source",
            transform=source_ax.transAxes,
            fontsize=8,
            fontweight="semibold",
            ha="left",
            va="top",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
        )

    base = FIGURE_DIR / output_stem
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(svg_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(pdf_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    plt.close(fig)
    return png_path, svg_path, pdf_path


def make_torus_knot(n_points: int = 240, p: int = 2, q: int = 3, scale: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    major = 1.6 + 0.45 * np.cos(q * t)
    return scale * np.column_stack(
        [
            major * np.cos(p * t),
            major * np.sin(p * t),
            0.75 * np.sin(q * t),
        ]
    )


def make_figure_eight(n_points: int = 260, scale: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return scale * np.column_stack(
        [
            (2.0 + np.cos(2.0 * t)) * np.cos(3.0 * t),
            (2.0 + np.cos(2.0 * t)) * np.sin(3.0 * t),
            np.sin(4.0 * t),
        ]
    )


def make_helix(n_points: int = 180, turns: float = 5.0, radius: float = 1.0, pitch: float = 0.16) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * turns, n_points)
    z = pitch * (t - t.mean())
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), z])


def make_bottlebrush(n_points: int = 240) -> np.ndarray:
    t = np.linspace(0.0, 7.0 * np.pi, n_points)
    backbone = np.column_stack([np.linspace(-2.0, 2.0, n_points), 0.25 * np.sin(0.8 * t), 0.15 * np.cos(0.6 * t)])
    brush = 0.34 * np.column_stack([np.zeros_like(t), np.cos(t), np.sin(t)])
    return backbone + brush


def make_torus_mesh(major: float = 1.15, minor: float = 0.34, n_major: int = 128, n_minor: int = 42) -> pv.PolyData:
    u_values = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v_values = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    points = []
    faces = []
    for u in u_values:
        for v in v_values:
            points.append(
                [
                    (major + minor * np.cos(v)) * np.cos(u),
                    (major + minor * np.cos(v)) * np.sin(u),
                    minor * np.sin(v),
                ]
            )
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + ((j + 1) % n_minor)
            d = i * n_minor + ((j + 1) % n_minor)
            faces.extend([4, a, b, c, d])
    return pv.PolyData(np.asarray(points), np.asarray(faces)).triangulate().clean()


def make_theta_paths(n_points: int = 160) -> list[np.ndarray]:
    t = np.linspace(0.0, 1.0, n_points)
    x = -1.18 + 2.36 * t
    top = np.column_stack([x, 0.82 * np.sin(np.pi * t), 0.22 * np.sin(2.0 * np.pi * t)])
    middle = np.column_stack([x, 0.0 * t, 0.70 * np.sin(np.pi * t)])
    bottom = np.column_stack([x, -0.82 * np.sin(np.pi * t), -0.22 * np.sin(2.0 * np.pi * t)])
    return [top, middle, bottom]


def make_genus2_surface_mesh(n_grid: int = 62, tube_radius: float = 0.235) -> pv.PolyData:
    axis = np.linspace(-1.70, 1.70, n_grid)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    min_dist2 = np.full(grid_points.shape[0], np.inf)
    for path in make_theta_paths():
        for start, end in zip(path[:-1], path[1:]):
            segment = end - start
            length2 = float(np.dot(segment, segment))
            if length2 == 0.0:
                continue
            rel = grid_points - start
            weight = np.clip((rel @ segment) / length2, 0.0, 1.0)
            closest = start + weight[:, None] * segment
            dist2 = np.sum((grid_points - closest) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
    values = np.sqrt(min_dist2).reshape((n_grid, n_grid, n_grid)) - tube_radius
    spacing = tuple(float(axis[1] - axis[0]) for _ in range(3))
    origin = (float(axis[0]), float(axis[0]), float(axis[0]))
    grid = pv.ImageData(dimensions=values.shape, spacing=spacing, origin=origin)
    grid.point_data["distance"] = values.ravel(order="F")
    return grid.contour(isosurfaces=[0.0], scalars="distance").triangulate().clean()


def make_nodal_fermi_mesh() -> pv.PolyData:
    n_major = 180
    n_minor = 60
    u_values = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v_values = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    points = []
    faces = []
    for u in u_values:
        ring_radius = 1.12 + 0.10 * np.cos(4.0 * u)
        tube_radius = 0.25 * (1.0 + 0.20 * np.cos(2.0 * u))
        for v in v_values:
            local_radius = tube_radius * (1.0 + 0.08 * np.cos(3.0 * v - u))
            points.append(
                [
                    (ring_radius + local_radius * np.cos(v)) * np.cos(u),
                    (ring_radius + local_radius * np.cos(v)) * np.sin(u),
                    0.72 * local_radius * np.sin(v) + 0.06 * np.sin(3.0 * u),
                ]
            )
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + ((j + 1) % n_minor)
            d = i * n_minor + ((j + 1) % n_minor)
            faces.extend([4, a, b, c, d])
    return pv.PolyData(np.asarray(points), np.asarray(faces)).triangulate().clean()


def make_capsid_mesh() -> pv.PolyData:
    mesh = pv.Icosphere(radius=1.25, nsub=3).triangulate().clean()
    pts = mesh.points.copy()
    r = np.linalg.norm(pts, axis=1)
    pts *= (1.0 + 0.08 * np.sin(8.0 * pts[:, 0]) * np.cos(7.0 * pts[:, 1]))[:, None]
    mesh.points = pts / r[:, None] * np.linalg.norm(pts, axis=1)[:, None]
    return mesh.triangulate().clean()


def make_gyroid_field(n: int = 76, radius_limit: float = 2.35):
    axis = np.linspace(-2.8, 2.8, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    k = 2.1
    gyroid = (
        np.sin(k * x) * np.cos(k * y)
        + np.sin(k * y) * np.cos(k * z)
        + np.sin(k * z) * np.cos(k * x)
    )
    radius = np.sqrt(x * x + y * y + z * z)
    values = np.maximum(np.abs(gyroid) - 0.12, radius - radius_limit)
    spacing = (float(axis[1] - axis[0]),) * 3
    origin = (float(axis[0]),) * 3
    return values, spacing, origin


def make_schwarz_p_field(n: int = 76):
    axis = np.linspace(-np.pi, np.pi, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.cos(x) + np.cos(y) + np.cos(z)
    spacing = (float(axis[1] - axis[0]),) * 3
    origin = (float(axis[0]),) * 3
    return values, spacing, origin


def arc_between(nodes: dict[str, list[float]], source: str, target: str, *, lift: float, side: float, n: int = 70) -> list[list[float]]:
    start = np.asarray(nodes[source], dtype=float)
    end = np.asarray(nodes[target], dtype=float)
    t = np.linspace(0.0, 1.0, n)
    base = (1.0 - t)[:, None] * start + t[:, None] * end
    tangent = end - start
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(normal) < 1e-8:
        normal = np.array([1.0, 0.0, 0.0])
    normal = normal / np.linalg.norm(normal)
    base += side * np.sin(np.pi * t)[:, None] * normal
    base[:, 2] += lift * np.sin(np.pi * t)
    return base.tolist()


def write_and_load_curve(points: np.ndarray, path: Path, *, curve_id: str, closed: bool = True):
    write_xyz_coords(points, path, comment=curve_id)
    return build_curve_from_xyz(path, closed=closed, curve_id=curve_id)


def write_and_load_spatial(payload: dict, set_dir: Path, stem: str):
    nodes_path = set_dir / f"{stem}_nodes.csv"
    edges_path = set_dir / f"{stem}_edges.csv"
    write_spatial_graph_csv(payload, nodes_path, edges_path)
    return build_spatial_graph_from_csv(nodes_path, edges_path, graph_id=stem)


def write_and_load_mesh(mesh: pv.PolyData, path: Path, *, mesh_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path)
    return build_surface_from_mesh_file(path, mesh_id=mesh_id)


def write_and_load_volume(values: np.ndarray, spacing, origin, path: Path, *, field_id: str):
    write_npz_scalar_field(values, path, spacing=spacing, origin=origin)
    return build_surface_from_scalar_field_file(path, level=0.0, field_id=field_id)


def draw_curve_result(result, *, closed: bool = True):
    def draw(plotter):
        pts = edge_points(result.graph)
        meshes = add_curve(plotter, pts, color=EDGE_COLOR, add_endpoints=not closed, closed=closed)
        return meshes, pts, result.issues

    def source(plotter):
        pts = edge_points(result.graph)
        meshes = add_source_beads_and_bonds(plotter, pts)
        return meshes, pts

    return draw, source


def draw_surface_result(result, *, color: str = SURFACE_COLOR):
    def add_surface_safe(plotter, mesh: pv.PolyData) -> list[pv.DataSet]:
        mesh = mesh.triangulate().clean()
        if mesh.n_cells > 6000:
            try:
                mesh = mesh.decimate_pro(0.55, preserve_topology=True)
            except Exception:
                pass
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=0.94,
            smooth_shading=True,
            specular=0.45,
            specular_power=22,
            metallic=0.08,
        )
        return [mesh]

    def draw(plotter):
        meshes = add_surface_safe(plotter, result.mesh)
        return meshes, result.mesh.points, [issue for issue in result.issues if "open boundary edges" not in issue]

    def source(plotter):
        mesh = result.mesh.triangulate().clean()
        plotter.add_mesh(mesh, color="#c3c9cc", opacity=0.28, smooth_shading=True)
        plotter.add_mesh(mesh, color="#687177", style="wireframe", opacity=0.72, line_width=0.8)
        return [mesh], mesh.points

    return draw, source


def draw_graph_result(result):
    def draw(plotter):
        meshes, pts = add_spatial_graph(plotter, result.graph)
        return meshes, pts, result.issues

    def source(plotter):
        return add_raw_spatial_graph(plotter, result.graph)

    return draw, source


def draw_polymer_result(result, *, closed: bool = True):
    def draw(plotter):
        pts = edge_points(result.graph)
        meshes = add_curve(plotter, pts, color=EDGE_COLOR, add_endpoints=not closed, closed=closed)
        return meshes, pts, result.issues

    def source(plotter):
        meshes = add_source_beads_and_bonds(plotter, result.coords)
        return meshes, result.coords

    return draw, source


def draw_pdb_result(result):
    def draw(plotter):
        pts = edge_points(result.graph)
        meshes = add_curve(plotter, pts, color=EDGE_COLOR, add_endpoints=True, direct_closure=True)
        return meshes, pts, result.issues

    def source(plotter):
        atom_points = load_pdb_atom_points(result.pdb_path, chain_id=result.chain_id, model_id=result.model_id)
        meshes = []
        meshes.extend(add_point_cloud(plotter, atom_points, color="#a8afb3", opacity=0.34, point_size=5.5))
        meshes.extend(add_source_backbone_trace(plotter, result.coords, color=EDGE_COLOR, radius_scale=0.008))
        return meshes, np.vstack([atom_points, result.coords])

    return draw, source


def draw_mmcif_result(result):
    def draw(plotter):
        pts = edge_points(result.graph)
        meshes = add_curve(plotter, pts, color=EDGE_COLOR, add_endpoints=True, direct_closure=True)
        return meshes, pts, result.issues

    def source(plotter):
        atom_points = load_mmcif_atom_points(result.cif_path, chain_id=result.chain_id, model_id=result.model_id)
        meshes = []
        meshes.extend(add_point_cloud(plotter, atom_points, color="#a8afb3", opacity=0.34, point_size=5.5))
        meshes.extend(add_source_backbone_trace(plotter, result.coords, color=EDGE_COLOR, radius_scale=0.008))
        return meshes, np.vstack([atom_points, result.coords])

    return draw, source


def draw_dna_duplex(dna_a, dna_b):
    def draw(plotter):
        pts_a = edge_points(dna_a.graph)
        pts_b = edge_points(dna_b.graph)
        pts = np.vstack([pts_a, pts_b])
        _, _, _, span = point_span(pts)
        radius = 0.018 * span
        meshes = []
        meshes.extend(add_curve(plotter, pts_a, color=EDGE_COLOR, add_endpoints=True, direct_closure=True))
        meshes.extend(add_curve(plotter, pts_b, color=EDGE_COLOR_2, add_endpoints=True, direct_closure=True))
        return meshes, pts, dna_a.issues + dna_b.issues

    def source(plotter):
        atoms_a = load_pdb_atom_points(dna_a.pdb_path, chain_id=dna_a.chain_id, model_id=dna_a.model_id)
        atoms_b = load_pdb_atom_points(dna_b.pdb_path, chain_id=dna_b.chain_id, model_id=dna_b.model_id)
        atom_points = np.vstack([atoms_a, atoms_b])
        meshes = []
        meshes.extend(add_point_cloud(plotter, atom_points, color="#a8afb3", opacity=0.34, point_size=5.5))
        meshes.extend(add_source_backbone_trace(plotter, dna_a.coords, color=EDGE_COLOR, radius_scale=0.008))
        meshes.extend(add_source_backbone_trace(plotter, dna_b.coords, color=EDGE_COLOR_2, radius_scale=0.008))
        return meshes, np.vstack([atom_points, dna_a.coords, dna_b.coords])

    return draw, source


def add_panel(panels: list[dict], summary: list[dict], set_name: str, title: str, domain: str, input_format: str, draw_fn, source_fn) -> None:
    panel_dir = SET_PANEL_DIR / set_name
    panel_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{chr(ord('a') + len(panels))}_{title.lower().replace(' ', '_').replace('/', '_')}"
    image, panel_path, issues = render_scene(draw_fn, panel_dir / f"{stem}.png")
    source_image, source_path = render_source(source_fn, panel_dir / f"{stem}_source.png")
    panels.append({"title": title, "image": image, "source_image": source_image})
    summary.append(
        {
            "title": title,
            "domain": domain,
            "input_format": input_format,
            "panel_path": str(panel_path),
            "source_view_path": str(source_path),
            "success": panel_path.exists() and panel_path.stat().st_size > 0 and source_path.exists() and source_path.stat().st_size > 0,
            "issues": issues,
        }
    )


def make_basic_network(name: str, kind: str) -> dict:
    if kind == "pipe":
        nodes = {
            "tank": [-1.8, -0.5, 0.0],
            "pump": [-0.7, 0.85, 0.25],
            "valve": [0.55, 0.55, -0.15],
            "chiller": [1.6, -0.35, 0.25],
            "sensor": [-0.20, -1.05, -0.25],
        }
        specs = [
            ("tank", "pump", 0.75, 0.20, "suction"),
            ("pump", "valve", 0.45, -0.28, "feed"),
            ("valve", "chiller", 0.30, 0.18, "supply"),
            ("chiller", "sensor", -0.35, 0.35, "return"),
            ("sensor", "tank", 0.25, -0.18, "monitor"),
        ]
    elif kind == "cooling":
        nodes = {
            "pump": [-1.8, -0.85, 0.0],
            "inlet": [-1.05, 0.85, 0.15],
            "cold_plate_1": [-0.35, 0.62, 0.46],
            "cold_plate_2": [0.35, -0.62, -0.22],
            "cold_plate_3": [1.02, 0.56, 0.28],
            "outlet": [1.75, -0.82, 0.05],
        }
        specs = [
            ("pump", "inlet", 0.22, 0.25, "inlet"),
            ("inlet", "cold_plate_1", 0.56, -0.18, "plate_a"),
            ("cold_plate_1", "cold_plate_2", -0.45, 0.58, "serpentine_a"),
            ("cold_plate_2", "cold_plate_3", 0.58, -0.58, "serpentine_b"),
            ("cold_plate_3", "outlet", 0.26, 0.20, "outlet"),
            ("outlet", "pump", 0.98, -0.76, "return_loop"),
            ("inlet", "outlet", 1.12, 0.72, "bypass"),
        ]
    elif kind == "circuit":
        nodes = {
            "source": [-1.7, 0.0, 0.0],
            "ic": [-0.35, 0.80, 0.28],
            "resistor": [0.85, 0.55, -0.16],
            "capacitor": [1.55, -0.65, 0.20],
            "ground": [-0.55, -1.05, -0.24],
        }
        specs = [
            ("source", "ic", 0.70, 0.25, "power"),
            ("source", "ground", -0.20, -0.15, "return"),
            ("ic", "resistor", 0.22, -0.25, "signal"),
            ("ic", "capacitor", -0.45, 0.42, "clock"),
            ("resistor", "capacitor", 0.20, -0.15, "filter"),
            ("capacitor", "ground", 0.55, 0.32, "shield"),
        ]
    elif kind == "truss":
        nodes = {
            "A": [-1.5, -1.0, 0.0],
            "B": [1.5, -1.0, 0.0],
            "C": [1.5, 1.0, 0.0],
            "D": [-1.5, 1.0, 0.0],
            "E": [0.0, 0.0, 1.25],
        }
        specs = [
            ("A", "B", 0.05, 0.00, "beam"),
            ("B", "C", 0.05, 0.00, "beam"),
            ("C", "D", 0.05, 0.00, "beam"),
            ("D", "A", 0.05, 0.00, "beam"),
            ("A", "E", 0.22, 0.10, "strut"),
            ("B", "E", 0.22, -0.10, "strut"),
            ("C", "E", 0.22, 0.10, "strut"),
            ("D", "E", 0.22, -0.10, "strut"),
            ("A", "C", -0.18, 0.20, "brace"),
            ("B", "D", 0.18, -0.20, "brace"),
        ]
    else:
        nodes = {
            "root": [-1.5, 0.0, 0.0],
            "b1": [-0.25, 0.85, 0.25],
            "b2": [-0.25, -0.85, -0.20],
            "tip1": [1.35, 1.10, 0.05],
            "tip2": [1.45, 0.0, 0.38],
            "tip3": [1.35, -1.05, -0.05],
        }
        specs = [
            ("root", "b1", 0.35, 0.10, "branch"),
            ("root", "b2", -0.25, -0.10, "branch"),
            ("b1", "tip1", 0.45, 0.20, "branch"),
            ("b1", "tip2", 0.20, -0.18, "branch"),
            ("b2", "tip2", 0.10, 0.18, "branch"),
            ("b2", "tip3", -0.35, -0.22, "branch"),
        ]
    return {
        "graph_id": name,
        "nodes": nodes,
        "edges": [
            {"id": edge_id, "source": u, "target": v, "type": edge_id, "points": arc_between(nodes, u, v, lift=lift, side=side)}
            for u, v, lift, side, edge_id in specs
        ],
    }


def make_hopf_payload() -> dict:
    nodes = {"A": [1.0, 0.0, 0.0], "B": [-1.0, 0.0, 0.0], "C": [0.0, 1.0, 0.0], "D": [0.0, -1.0, 0.0]}
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=True)
    loop1 = np.column_stack([np.cos(t), np.sin(t), 0.28 * np.sin(2.0 * t)]).tolist()
    loop2 = np.column_stack([0.28 * np.sin(2.0 * t), np.cos(t), np.sin(t)]).tolist()
    nodes["A"] = loop1[0]
    nodes["B"] = loop1[len(loop1) // 2]
    nodes["C"] = loop2[0]
    nodes["D"] = loop2[len(loop2) // 2]
    return {
        "graph_id": "hopf_link_csv",
        "nodes": nodes,
        "edges": [
            {"id": "ring1a", "source": "A", "target": "B", "points": loop1[: len(loop1) // 2 + 1]},
            {"id": "ring1b", "source": "B", "target": "A", "points": loop1[len(loop1) // 2 :] + [loop1[0]]},
            {"id": "ring2a", "source": "C", "target": "D", "points": loop2[: len(loop2) // 2 + 1]},
            {"id": "ring2b", "source": "D", "target": "C", "points": loop2[len(loop2) // 2 :] + [loop2[0]]},
        ],
    }


def make_three_ring_payload() -> dict:
    nodes = {}
    edges = []
    for idx, axis in enumerate("xyz"):
        t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=True)
        if axis == "x":
            pts = np.column_stack([0.25 * np.sin(2.0 * t), 1.0 * np.cos(t), 0.72 * np.sin(t)])
        elif axis == "y":
            pts = np.column_stack([1.0 * np.cos(t), 0.25 * np.sin(2.0 * t), 0.72 * np.sin(t)])
        else:
            pts = np.column_stack([1.0 * np.cos(t), 0.72 * np.sin(t), 0.25 * np.sin(2.0 * t)])
        a = f"{axis}0"
        b = f"{axis}1"
        nodes[a] = pts[0].tolist()
        nodes[b] = pts[len(pts) // 2].tolist()
        edges.append({"id": f"{axis}a", "source": a, "target": b, "points": pts[: len(pts) // 2 + 1].tolist()})
        edges.append({"id": f"{axis}b", "source": b, "target": a, "points": (pts[len(pts) // 2 :].tolist() + [pts[0].tolist()])})
    return {"graph_id": "three_ring_link_csv", "nodes": nodes, "edges": edges}


def build_biology_set() -> tuple[str, str, list[dict]]:
    set_name = "set2_biology"
    output_stem = "task2_input_gallery_set2_biology_with_source_insets"
    panels: list[dict] = []
    summary: list[dict] = []
    set_dir = DATA_DIR / set_name
    protein_dir = EXAMPLES_DIR / "proteins" / "data"
    input_protein_dir = HERE / "data" / "proteins"
    dna_dir = HERE / "data" / "dna"
    mmcif_dir = HERE / "data" / "mmcif"

    entries = [
        ("Crambin PDB", "protein", ".pdb", draw_pdb_result(from_protein_ca_backbone("1CRN", chain_id="A", data_dir=protein_dir, save_coords=True))),
        ("Ubiquitin PDB", "protein", ".pdb", draw_pdb_result(from_protein_ca_backbone("1UBQ", chain_id="A", data_dir=input_protein_dir, save_coords=True))),
        ("Knotted Protein PDB", "protein", ".pdb", draw_pdb_result(from_protein_ca_backbone("1J85", chain_id="A", data_dir=input_protein_dir, save_coords=True))),
        ("Hemoglobin PDB", "protein complex", ".pdb", draw_pdb_result(from_protein_ca_backbone("4HHB", chain_id="A", data_dir=protein_dir, save_coords=True))),
    ]
    for title, domain, fmt, (draw, source) in entries:
        add_panel(panels, summary, set_name, title, domain, fmt, draw, source)

    dna_a = from_nucleic_acid_backbone("1BNA", chain_id="A", atom_name="P", data_dir=dna_dir, save_coords=True)
    dna_b = from_nucleic_acid_backbone("1BNA", chain_id="B", atom_name="P", data_dir=dna_dir, save_coords=True)
    add_panel(panels, summary, set_name, "B-DNA Duplex PDB", "DNA", ".pdb", *draw_dna_duplex(dna_a, dna_b))

    rna = from_mmcif_backbone("1EHZ", chain_id="A", atom_name="P", data_dir=mmcif_dir, save_coords=True)
    add_panel(panels, summary, set_name, "tRNA mmCIF", "RNA", ".cif", *draw_mmcif_result(rna))

    ubq_cif = from_mmcif_backbone("1UBQ", chain_id="A", atom_name="CA", data_dir=mmcif_dir, save_coords=True)
    add_panel(panels, summary, set_name, "Ubiquitin mmCIF", "protein", ".cif", *draw_mmcif_result(ubq_cif))

    helix = write_and_load_curve(
        make_helix(n_points=170, turns=4.6, radius=0.86, pitch=0.17),
        set_dir / "alpha_helix.xyz",
        curve_id="alpha_helix_xyz",
        closed=False,
    )
    add_panel(panels, summary, set_name, "Alpha Helix XYZ", "peptide coordinate chain", ".xyz", *draw_curve_result(helix, closed=False))

    capsid = write_and_load_mesh(make_capsid_mesh(), set_dir / "viral_capsid_mesh.ply", mesh_id="viral_capsid_ply")
    add_panel(panels, summary, set_name, "Viral Capsid PLY", "biomolecular surface", ".ply", *draw_surface_result(capsid))

    return set_name, output_stem, panels, summary


def build_engineering_set() -> tuple[str, str, list[dict], list[dict]]:
    set_name = "set3_engineering_polymers"
    output_stem = "task2_input_gallery_set3_engineering_polymers_with_source_insets"
    panels: list[dict] = []
    summary: list[dict] = []
    set_dir = DATA_DIR / set_name

    for title, kind in [
        ("Pipe Manifold CSV", "pipe"),
        ("Circuit Harness CSV", "circuit"),
        ("Cooling Network CSV", "cooling"),
        ("Vascular Branch CSV", "vascular"),
        ("Lattice Truss CSV", "truss"),
    ]:
        payload = make_basic_network(title.lower().replace(" ", "_"), kind)
        graph = write_and_load_spatial(payload, set_dir, title.lower().replace(" ", "_"))
        add_panel(panels, summary, set_name, title, "spatial network", "node/edge CSV", *draw_graph_result(graph))

    ring_path = set_dir / "ring_polymer.gro"
    ring_pts = make_torus_knot(n_points=230, p=1, q=2, scale=0.88)
    write_gro_coords(ring_pts, ring_path, title="ring polymer source snapshot")
    ring = from_gromacs_gro(ring_path, closed=True, closure="direct", polymer_id="ring_polymer_gro")
    add_panel(panels, summary, set_name, "Ring Polymer GRO", "polymer", ".gro", *draw_polymer_result(ring))

    polymer_path = set_dir / "polymer_lammps.dump"
    polymer_points = make_torus_knot(n_points=260, p=2, q=5, scale=1.0)
    write_lammps_dump(polymer_points, polymer_path, molecule_id=7)
    polymer_result = from_lammps_dump(polymer_path, molecule_id=7, closed=True, closure="direct", polymer_id="polymer_lammps")
    add_panel(panels, summary, set_name, "Polymer LAMMPS", "polymer", "LAMMPS dump", *draw_polymer_result(polymer_result))

    bottlebrush = write_and_load_curve(
        make_bottlebrush(),
        set_dir / "bottlebrush_polymer.xyz",
        curve_id="bottlebrush_polymer_xyz",
        closed=False,
    )
    add_panel(panels, summary, set_name, "Bottlebrush Polymer XYZ", "polymer coordinate chain", ".xyz", *draw_curve_result(bottlebrush, closed=False))

    cable = write_and_load_curve(
        make_helix(n_points=260, turns=7.0, radius=0.9, pitch=0.12),
        set_dir / "coiled_cable.dat",
        curve_id="coiled_cable_dat",
        closed=False,
    )
    add_panel(panels, summary, set_name, "Coiled Cable DAT", "cable harness", ".dat/.xyz", *draw_curve_result(cable, closed=False))

    return set_name, output_stem, panels, summary


def build_topology_set() -> tuple[str, str, list[dict], list[dict]]:
    set_name = "set4_topology_physics"
    output_stem = "task2_input_gallery_set4_topology_physics_with_source_insets"
    panels: list[dict] = []
    summary: list[dict] = []
    set_dir = DATA_DIR / set_name

    genus2 = write_and_load_mesh(make_genus2_surface_mesh(), set_dir / "genus2_surface.ply", mesh_id="genus2_surface")
    add_panel(panels, summary, set_name, "Genus-2 Surface PLY", "surface mesh", ".ply", *draw_surface_result(genus2))

    torus = write_and_load_mesh(make_torus_mesh(), set_dir / "torus_surface.ply", mesh_id="torus_surface")
    add_panel(panels, summary, set_name, "Torus Surface PLY", "surface mesh", ".ply", *draw_surface_result(torus))

    fig8 = write_and_load_curve(make_figure_eight(), set_dir / "figure_eight_knot.xyz", curve_id="figure_eight_knot")
    add_panel(panels, summary, set_name, "Figure-Eight XYZ", "coordinate knot", ".xyz", *draw_curve_result(fig8))

    hopf = write_and_load_spatial(make_hopf_payload(), set_dir, "hopf_link_csv")
    add_panel(panels, summary, set_name, "Hopf Link CSV", "spatial graph", "node/edge CSV", *draw_graph_result(hopf))

    rings = write_and_load_spatial(make_three_ring_payload(), set_dir, "three_ring_link_csv")
    add_panel(panels, summary, set_name, "Three-Ring Link CSV", "spatial graph", "node/edge CSV", *draw_graph_result(rings))

    gyroid_values, gyroid_spacing, gyroid_origin = make_gyroid_field()
    gyroid = write_and_load_volume(gyroid_values, gyroid_spacing, gyroid_origin, set_dir / "gyroid_volume.npz", field_id="gyroid_volume")
    draw, source = draw_surface_result(gyroid, color=VOLUME_COLOR)

    def gyroid_source(output_plotter):
        return source(output_plotter)

    add_panel(panels, summary, set_name, "Gyroid Volume NPZ", "volumetric field", ".npz", draw, gyroid_source)

    schwarz_values, schwarz_spacing, schwarz_origin = make_schwarz_p_field()
    schwarz = write_and_load_volume(schwarz_values, schwarz_spacing, schwarz_origin, set_dir / "schwarz_p_volume.npz", field_id="schwarz_p_volume")
    add_panel(panels, summary, set_name, "Schwarz-P Volume NPZ", "volumetric field", ".npz", *draw_surface_result(schwarz, color=VOLUME_COLOR))

    fermi = write_and_load_mesh(make_nodal_fermi_mesh(), set_dir / "nodal_line_fermi.vtp", mesh_id="nodal_line_fermi")
    add_panel(panels, summary, set_name, "Nodal-Line Fermi VTP", "Fermi surface", ".vtp", *draw_surface_result(fermi, color=FERMI_COLOR))

    return set_name, output_stem, panels, summary


def main() -> None:
    ensure_dirs()
    builders = [build_biology_set, build_engineering_set, build_topology_set]
    all_summaries = []
    for builder in builders:
        result = builder()
        if len(result) == 4:
            set_name, output_stem, panels, summary = result
        else:
            raise RuntimeError("Gallery set builder returned an unexpected result.")
        png_path, svg_path, pdf_path = assemble_gallery(panels, output_stem=output_stem)
        summary.extend(
            [
                {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
                {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
                {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
            ]
        )
        summary_path = DATA_DIR / f"{set_name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        all_summaries.append({"set_name": set_name, "output_stem": output_stem, "summary_path": str(summary_path)})
        print(f"Generated {set_name}")
        print(f"  PNG: {png_path}")
        print(f"  SVG: {svg_path}")
        print(f"  PDF: {pdf_path}")
        print(f"  Summary: {summary_path}")
        for item in summary:
            if "title" in item:
                print(f"  Panel: {item['title']} success={item['success']} issues={item['issues'] or 'none'}")
    (DATA_DIR / "additional_gallery_sets_summary.json").write_text(json.dumps(all_summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
