"""Publication-style Task 2 input gallery.

This script uses the public input adapters where available and renders a 3x3 figure
in the same visual spirit as the archived surface-skeletonization pipeline:
PyVista off-screen screenshots for each panel, then Matplotlib assembly.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv
from knotted_graph.inputs.mmcif import iter_atom_site_rows


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
ROOT = EXAMPLES_DIR.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
WITH_RAW_INSETS = os.environ.get("TASK2_GALLERY_RAW_INSETS", "0") == "1"
WITH_SOURCE_INSETS = os.environ.get("TASK2_GALLERY_SOURCE_INSETS", "0") == "1"
WITH_INSETS = WITH_RAW_INSETS or WITH_SOURCE_INSETS
FIGURE_STEM = (
    "task2_input_gallery_publication_style_with_source_insets"
    if WITH_SOURCE_INSETS
    else "task2_input_gallery_publication_style_with_raw_inputs"
    if WITH_RAW_INSETS
    else "task2_input_gallery_publication_style"
)
PANEL_DIR = FIGURE_DIR / (
    "publication_style_panels_with_source_insets"
    if WITH_SOURCE_INSETS
    else "publication_style_panels_with_raw_inputs"
    if WITH_RAW_INSETS
    else "publication_style_panels"
)
SUMMARY_PATH = DATA_DIR / (
    "publication_style_gallery_with_source_insets_summary.json"
    if WITH_SOURCE_INSETS
    else "publication_style_gallery_with_raw_inputs_summary.json"
    if WITH_RAW_INSETS
    else "publication_style_gallery_summary.json"
)
INSET_LABEL = "source" if WITH_SOURCE_INSETS else "input"
INSET_PATH_SUFFIX = "source_view" if WITH_SOURCE_INSETS else "raw_input"

for relative in [
    "coordinate_chains",
    "dna",
    "mmcif",
    "polymers",
    "proteins",
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
    from_lammps_dump,
    from_mmcif_backbone,
    from_nucleic_acid_backbone,
    from_protein_ca_backbone,
    write_lammps_dump,
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

EDGE_COLOR = "#0B5C89"
EDGE_COLOR_2 = "#2C79A7"
NODE_COLOR = "#A60628"
START_COLOR = "#A60628"
END_COLOR = "#A60628"
SURFACE_COLOR = "#0E6FA8"
VOLUME_COLOR = "#2F86B7"
FERMI_COLOR = "#1B75A6"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for old_panel in PANEL_DIR.glob("*.png"):
        old_panel.unlink()


def edge_points(graph) -> np.ndarray:
    data = next(iter(graph.edges(data=True)))[2]
    return np.asarray(data["pts"], dtype=float)


def crop_white(image: np.ndarray, threshold: int = 248, pad: int = 8) -> np.ndarray:
    mask = np.any(image < threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def point_span(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    return mins, maxs, center, span


def set_camera(plotter: pv.Plotter, points: np.ndarray) -> None:
    _, _, center, span = point_span(points)
    direction = np.array([1.15, -1.55, 1.08], dtype=float)
    direction /= np.linalg.norm(direction)
    plotter.camera_position = [
        tuple(center + direction * 4.0 * span),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 0.92 * span


def make_plotter(window_size: tuple[int, int] = (1800, 1600)) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    return plotter


def load_pdb_atom_points(path: Path, *, chain_id: str | None = None, model_id: int = 1) -> np.ndarray:
    """Load raw PDB atom coordinates for a compact input thumbnail."""
    points = []
    current_model_id = None
    has_model_records = False
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("MODEL"):
                has_model_records = True
                raw = line[10:14].strip() or line[5:].strip()
                try:
                    current_model_id = int(raw)
                except ValueError:
                    current_model_id = None
                continue
            if line.startswith("ENDMDL"):
                current_model_id = None
                continue
            if not line.startswith("ATOM"):
                continue
            if has_model_records and current_model_id != model_id:
                continue
            if chain_id is not None and (line[21].strip() or "?") != chain_id:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            try:
                points.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return np.asarray(points, dtype=float)


def load_mmcif_atom_points(path: Path, *, chain_id: str | None = None, model_id: int = 1) -> np.ndarray:
    """Load raw mmCIF atom coordinates for a compact input thumbnail."""
    points = []
    for row in iter_atom_site_rows(path):
        if row.get("_atom_site.group_PDB") != "ATOM":
            continue
        raw_model = row.get("_atom_site.pdbx_PDB_model_num", "1")
        try:
            if int(raw_model) != model_id:
                continue
        except ValueError:
            continue
        row_chain = row.get("_atom_site.auth_asym_id") or row.get("_atom_site.label_asym_id")
        if chain_id is not None and row_chain != chain_id:
            continue
        altloc = row.get("_atom_site.label_alt_id", ".")
        if altloc not in {".", "?", "A"}:
            continue
        try:
            points.append(
                (
                    float(row["_atom_site.Cartn_x"]),
                    float(row["_atom_site.Cartn_y"]),
                    float(row["_atom_site.Cartn_z"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return np.asarray(points, dtype=float)


def polyline_mesh(points: np.ndarray) -> pv.PolyData:
    pts = np.asarray(points, dtype=float)
    poly = pv.PolyData(pts)
    poly.lines = np.concatenate(([pts.shape[0]], np.arange(pts.shape[0])))
    return poly


def save_image_array(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    return path


def add_point_cloud(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    color: str = "#7f878c",
    opacity: float = 0.55,
    point_size: float = 7.0,
    max_points: int = 1400,
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return []
    if pts.shape[0] > max_points:
        indices = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[indices]
    cloud = pv.PolyData(pts)
    plotter.add_mesh(
        cloud,
        color=color,
        opacity=opacity,
        point_size=point_size,
        render_points_as_spheres=True,
    )
    return [cloud]


def add_raw_spatial_graph(plotter: pv.Plotter, graph) -> tuple[list[pv.DataSet], np.ndarray]:
    pts_all = graph_points(graph)
    _, _, _, span = point_span(pts_all)
    meshes: list[pv.DataSet] = []
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)
        tube = polyline_mesh(pts).tube(radius=0.006 * span, n_sides=12, capping=True)
        meshes.append(tube)
        plotter.add_mesh(tube, color="#6f777b", opacity=0.70, smooth_shading=True)
    for _, data in graph.nodes(data=True):
        pos = np.asarray(data["pos"], dtype=float)
        cube = pv.Cube(center=pos, x_length=0.11 * span, y_length=0.11 * span, z_length=0.11 * span)
        meshes.append(cube)
        plotter.add_mesh(cube, color="#b9bfc2", opacity=0.82, smooth_shading=False)
    return meshes, pts_all


def add_source_beads_and_bonds(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    bead_color: str = "#7d858a",
    bond_color: str = "#b1b8bc",
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    _, _, _, span = point_span(pts)
    line = polyline_mesh(pts).tube(radius=0.006 * span, n_sides=12, capping=True)
    beads = pv.PolyData(pts).glyph(
        geom=pv.Sphere(radius=0.018 * span, theta_resolution=12, phi_resolution=8),
        orient=False,
        scale=False,
    )
    plotter.add_mesh(line, color=bond_color, opacity=0.78, smooth_shading=True)
    plotter.add_mesh(beads, color=bead_color, opacity=0.84, smooth_shading=True, specular=0.25)
    return [line, beads]


def add_source_backbone_trace(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    color: str = EDGE_COLOR,
    radius_scale: float = 0.007,
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    _, _, _, span = point_span(pts)
    tube = polyline_mesh(pts).tube(radius=radius_scale * span, n_sides=16, capping=True)
    plotter.add_mesh(tube, color=color, opacity=0.88, smooth_shading=True, specular=0.22)
    return [tube]


def scalar_slices_image(values: np.ndarray, output_path: Path) -> np.ndarray:
    """Render three orthogonal scalar-field slices as a source-domain thumbnail."""
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
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, transparent=False)
    plt.close(fig)
    buffer.seek(0)
    image = np.asarray(Image.open(buffer).convert("RGB"))
    save_image_array(image, output_path)
    return image


def add_curve(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    color: str = EDGE_COLOR,
    add_endpoints: bool = True,
    closed: bool = False,
    direct_closure: bool = False,
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    _, _, _, span = point_span(pts)
    render_pts = np.vstack([pts, pts[0]]) if direct_closure and pts.shape[0] >= 2 else pts
    tube = polyline_mesh(render_pts).tube(radius=0.018 * span, n_sides=32, capping=True)
    meshes: list[pv.DataSet] = [tube]
    plotter.add_mesh(
        tube,
        color=color,
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        metallic=0.08,
    )
    if add_endpoints and pts.shape[0] >= 2:
        start = pv.Sphere(radius=0.045 * span, center=pts[0], theta_resolution=32, phi_resolution=16)
        meshes.append(start)
        plotter.add_mesh(start, color=START_COLOR, smooth_shading=True, specular=0.45)
        if not closed:
            end = pv.Sphere(radius=0.045 * span, center=pts[-1], theta_resolution=32, phi_resolution=16)
            meshes.append(end)
            plotter.add_mesh(end, color=END_COLOR, smooth_shading=True, specular=0.45)
    return meshes


def add_curve_with_radius(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    radius: float,
    color: str = EDGE_COLOR,
    add_endpoints: bool = False,
    closed: bool = False,
    direct_closure: bool = False,
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    render_pts = np.vstack([pts, pts[0]]) if direct_closure and pts.shape[0] >= 2 else pts
    tube = polyline_mesh(render_pts).tube(radius=radius, n_sides=32, capping=True)
    meshes: list[pv.DataSet] = [tube]
    plotter.add_mesh(
        tube,
        color=color,
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        metallic=0.08,
    )
    if add_endpoints and pts.shape[0] >= 2:
        start = pv.Sphere(radius=2.5 * radius, center=pts[0], theta_resolution=32, phi_resolution=16)
        meshes.append(start)
        plotter.add_mesh(start, color=START_COLOR, smooth_shading=True, specular=0.45)
        if not closed:
            end = pv.Sphere(radius=2.5 * radius, center=pts[-1], theta_resolution=32, phi_resolution=16)
            meshes.append(end)
            plotter.add_mesh(end, color=END_COLOR, smooth_shading=True, specular=0.45)
    return meshes


def graph_points(graph) -> np.ndarray:
    chunks = []
    for _, data in graph.nodes(data=True):
        if "pos" in data:
            chunks.append(np.asarray(data["pos"], dtype=float).reshape(1, 3))
    for _, _, data in graph.edges(data=True):
        chunks.append(np.asarray(data["pts"], dtype=float))
    return np.vstack(chunks)


def add_spatial_graph(plotter: pv.Plotter, graph) -> tuple[list[pv.DataSet], np.ndarray]:
    pts_all = graph_points(graph)
    _, _, _, span = point_span(pts_all)
    meshes: list[pv.DataSet] = []
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)
        tube = polyline_mesh(pts).tube(radius=0.016 * span, n_sides=28, capping=True)
        meshes.append(tube)
        plotter.add_mesh(tube, color=EDGE_COLOR, smooth_shading=True, specular=0.45)
    node_pts = np.asarray([data["pos"] for _, data in graph.nodes(data=True)], dtype=float)
    nodes = pv.PolyData(node_pts).glyph(
        geom=pv.Sphere(radius=0.043 * span, theta_resolution=32, phi_resolution=16),
        orient=False,
        scale=False,
    )
    meshes.append(nodes)
    plotter.add_mesh(nodes, color=NODE_COLOR, smooth_shading=True, specular=0.5)
    return meshes, pts_all


def add_surface(plotter: pv.Plotter, mesh: pv.PolyData, *, color: str) -> list[pv.DataSet]:
    mesh = mesh.triangulate().clean()
    if mesh.n_cells > 6000:
        mesh = mesh.decimate_pro(0.55, preserve_topology=True)
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


def render_panel(draw_fn, output_stem: str) -> tuple[np.ndarray, Path, list[str]]:
    plotter = make_plotter()
    meshes, points, issues = draw_fn(plotter)
    set_camera(plotter, points)
    image_path = PANEL_DIR / f"{output_stem}.png"
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    return crop_white(image, threshold=249, pad=10), image_path, issues


def render_raw_scene(draw_fn, output_stem: str) -> tuple[np.ndarray, Path]:
    plotter = make_plotter(window_size=(900, 760))
    meshes, points = draw_fn(plotter)
    set_camera(plotter, points)
    image_path = PANEL_DIR / f"{output_stem}_{INSET_PATH_SUFFIX}.png"
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    cropped = crop_white(image, threshold=249, pad=8)
    save_image_array(cropped, image_path)
    return cropped, image_path


def assemble_figure(panels: list[dict]) -> tuple[Path, Path, Path]:
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
        if WITH_SOURCE_INSETS and panel.get("raw_image") is not None:
            body = inner[1, 0].subgridspec(1, 2, width_ratios=[0.78, 0.22], wspace=0.02)
            image_ax = fig.add_subplot(body[0, 0])
            source_ax = fig.add_subplot(body[0, 1])
        else:
            image_ax = fig.add_subplot(inner[1, 0])
            source_ax = None
        title_ax.axis("off")
        image_ax.axis("off")
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
            fontsize=16,
            fontweight="semibold",
            ha="center",
            va="center",
        )
        image_ax.imshow(panel["image"])
        if source_ax is not None:
            source_ax.imshow(panel["raw_image"])
            source_ax.set_xticks([])
            source_ax.set_yticks([])
            source_ax.set_facecolor("white")
            for spine in source_ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#333333")
                spine.set_linewidth(0.75)
            source_ax.text(
                0.07,
                0.92,
                INSET_LABEL,
                transform=source_ax.transAxes,
                fontsize=8,
                fontweight="semibold",
                ha="left",
                va="top",
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
            )
            source_ax.set_box_aspect(1)
        elif WITH_RAW_INSETS and panel.get("raw_image") is not None:
            inset_ax = image_ax.inset_axes([0.61, 0.58, 0.34, 0.34])
            inset_ax.imshow(panel["raw_image"])
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_facecolor("white")
            for spine in inset_ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#333333")
                spine.set_linewidth(0.8)
            inset_ax.text(
                0.06,
                0.92,
                INSET_LABEL,
                transform=inset_ax.transAxes,
                fontsize=8,
                fontweight="semibold",
                ha="left",
                va="top",
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
            )
        image_ax.set_box_aspect(1)

    base = FIGURE_DIR / FIGURE_STEM
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(svg_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(pdf_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    plt.close(fig)
    return png_path, svg_path, pdf_path


def main() -> None:
    ensure_dirs()
    panels = []
    summary = []

    def add_panel(title: str, output_stem: str, draw_fn, *, input_format: str, domain: str, raw_fn=None):
        image, panel_path, issues = render_panel(draw_fn, output_stem)
        raw_image = None
        raw_path = None
        if WITH_INSETS and raw_fn is not None:
            raw_image, raw_path = raw_fn(output_stem)
        panels.append(
            {
                "title": title,
                "image": image,
                "raw_image": raw_image,
                "panel_path": panel_path,
                "raw_path": raw_path,
            }
        )
        summary.append(
            {
                "title": title,
                "domain": domain,
                "input_format": input_format,
                "panel_path": str(panel_path),
                "raw_input_path": str(raw_path) if raw_path is not None else None,
                "source_view_path": str(raw_path) if WITH_SOURCE_INSETS and raw_path is not None else None,
                "success": panel_path.exists() and panel_path.stat().st_size > 0,
                "issues": issues,
            }
        )

    def make_torus_knot(n_points: int = 220, p: int = 2, q: int = 3, scale: float = 1.0) -> np.ndarray:
        t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        major = 1.6 + 0.45 * np.cos(q * t)
        return scale * np.column_stack(
            [
                major * np.cos(p * t),
                major * np.sin(p * t),
                0.75 * np.sin(q * t),
            ]
        )

    def make_engineering_network_payload() -> dict:
        """Create a 3D component-interconnect network for the CSV panel."""
        nodes = {
            "pump": [-1.65, -0.45, 0.00],
            "controller": [-0.45, 0.85, 0.35],
            "heat_exchanger": [0.95, 0.55, -0.10],
            "battery": [1.55, -0.70, 0.25],
            "sensor": [-0.10, -1.05, -0.35],
        }

        def arc(source, target, lift, side, bend=0.0):
            start = np.asarray(nodes[source], dtype=float)
            end = np.asarray(nodes[target], dtype=float)
            t = np.linspace(0.0, 1.0, 70)
            base = (1.0 - t)[:, None] * start + t[:, None] * end
            tangent = end - start
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(normal) < 1e-8:
                normal = np.array([1.0, 0.0, 0.0])
            normal = normal / np.linalg.norm(normal)
            base += side * np.sin(np.pi * t)[:, None] * normal
            base += bend * np.sin(2.0 * np.pi * t)[:, None] * np.array([0.0, 0.0, 1.0])
            base[:, 2] += lift * np.sin(np.pi * t)
            return base.tolist()

        pairs = [
            ("pump", "controller", 0.80, 0.25, 0.08, "coolant_loop"),
            ("pump", "sensor", -0.35, -0.20, 0.04, "signal"),
            ("controller", "heat_exchanger", 0.25, -0.32, -0.10, "control_bus"),
            ("controller", "battery", -0.45, 0.40, 0.08, "power"),
            ("sensor", "heat_exchanger", 0.65, 0.30, -0.05, "return_pipe"),
            ("heat_exchanger", "battery", 0.25, -0.16, 0.05, "thermal_link"),
            ("pump", "battery", 1.10, -0.55, 0.12, "overhead_cable"),
        ]
        return {
            "graph_id": "engineering_network_csv",
            "nodes": nodes,
            "edges": [
                {
                    "id": edge_type,
                    "source": u,
                    "target": v,
                    "type": edge_type,
                    "points": arc(u, v, lift, side, bend),
                }
                for u, v, lift, side, bend, edge_type in pairs
            ],
        }

    def make_theta_paths(n_points: int = 160) -> list[np.ndarray]:
        t = np.linspace(0.0, 1.0, n_points)
        x = -1.18 + 2.36 * t
        top = np.column_stack([x, 0.82 * np.sin(np.pi * t), 0.22 * np.sin(2.0 * np.pi * t)])
        middle = np.column_stack([x, 0.0 * t, 0.70 * np.sin(np.pi * t)])
        bottom = np.column_stack([x, -0.82 * np.sin(np.pi * t), -0.22 * np.sin(2.0 * np.pi * t)])
        return [top, middle, bottom]

    def make_genus2_surface_mesh(n_grid: int = 62, tube_radius: float = 0.235) -> pv.PolyData:
        """Create a genus-2 surface as a tube neighborhood of a theta graph."""
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

    def make_gyroid_pocket_field(n: int = 82):
        axis = np.linspace(-2.8, 2.8, n)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        k = 2.1
        gyroid = (
            np.sin(k * x) * np.cos(k * y)
            + np.sin(k * y) * np.cos(k * z)
            + np.sin(k * z) * np.cos(k * x)
        )
        radius = np.sqrt(x * x + y * y + z * z)
        values = np.maximum(np.abs(gyroid) - 0.12, radius - 2.35)
        spacing = (float(axis[1] - axis[0]),) * 3
        origin = (float(axis[0]),) * 3
        return values, spacing, origin

    def make_nodal_line_fermi_surface() -> pv.PolyData:
        """Create a warped toroidal Fermi surface around a nodal ring."""
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
                x = (ring_radius + local_radius * np.cos(v)) * np.cos(u)
                y = (ring_radius + local_radius * np.cos(v)) * np.sin(u)
                z = 0.72 * local_radius * np.sin(v) + 0.06 * np.sin(3.0 * u)
                points.append([x, y, z])

        for i in range(n_major):
            for j in range(n_minor):
                a = i * n_minor + j
                b = ((i + 1) % n_major) * n_minor + j
                c = ((i + 1) % n_major) * n_minor + ((j + 1) % n_minor)
                d = i * n_minor + ((j + 1) % n_minor)
                faces.extend([4, a, b, c, d])

        return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces)).triangulate().clean()

    protein = from_protein_ca_backbone(
        "1J85",
        chain_id="A",
        data_dir=DATA_DIR / "proteins",
        save_coords=True,
    )

    def draw_protein(plotter):
        pts = edge_points(protein.graph)
        meshes = add_curve(
            plotter,
            pts,
            color=EDGE_COLOR,
            add_endpoints=True,
            closed=False,
            direct_closure=True,
        )
        return meshes, pts, protein.issues

    def raw_protein(output_stem):
        atom_points = load_pdb_atom_points(protein.pdb_path, chain_id=protein.chain_id, model_id=protein.model_id)

        def draw_raw(plotter):
            meshes = []
            meshes.extend(add_point_cloud(plotter, atom_points, color="#a8afb3", opacity=0.38, point_size=6.5))
            meshes.extend(add_source_backbone_trace(plotter, protein.coords, color=EDGE_COLOR, radius_scale=0.008))
            return meshes, np.vstack([atom_points, protein.coords])

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "Knotted Protein PDB",
        "a_knotted_protein_pdb",
        draw_protein,
        input_format=".pdb",
        domain="protein",
        raw_fn=raw_protein,
    )

    dna_a = from_nucleic_acid_backbone(
        "1BNA",
        chain_id="A",
        atom_name="P",
        data_dir=DATA_DIR / "dna",
        save_coords=True,
    )
    dna_b = from_nucleic_acid_backbone(
        "1BNA",
        chain_id="B",
        atom_name="P",
        data_dir=DATA_DIR / "dna",
        save_coords=True,
    )

    def draw_dna(plotter):
        pts_a = edge_points(dna_a.graph)
        pts_b = edge_points(dna_b.graph)
        pts = np.vstack([pts_a, pts_b])
        _, _, _, span = point_span(pts)
        radius = 0.018 * span
        meshes = []
        meshes.extend(
            add_curve_with_radius(
                plotter,
                pts_a,
                radius=radius,
                color=EDGE_COLOR,
                add_endpoints=True,
                direct_closure=True,
            )
        )
        meshes.extend(
            add_curve_with_radius(
                plotter,
                pts_b,
                radius=radius,
                color=EDGE_COLOR_2,
                add_endpoints=True,
                direct_closure=True,
            )
        )
        return meshes, pts, dna_a.issues + dna_b.issues

    def raw_dna(output_stem):
        atoms_a = load_pdb_atom_points(dna_a.pdb_path, chain_id=dna_a.chain_id, model_id=dna_a.model_id)
        atoms_b = load_pdb_atom_points(dna_b.pdb_path, chain_id=dna_b.chain_id, model_id=dna_b.model_id)
        atom_points = np.vstack([atoms_a, atoms_b])
        phosphate_points = np.vstack([dna_a.coords, dna_b.coords])

        def draw_raw(plotter):
            meshes = []
            meshes.extend(add_point_cloud(plotter, atom_points, color="#a8afb3", opacity=0.36, point_size=6.0))
            meshes.extend(add_source_backbone_trace(plotter, dna_a.coords, color=EDGE_COLOR, radius_scale=0.008))
            meshes.extend(add_source_backbone_trace(plotter, dna_b.coords, color=EDGE_COLOR_2, radius_scale=0.008))
            return meshes, np.vstack([atom_points, phosphate_points])

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "DNA Double Helix PDB",
        "b_dna_double_helix_pdb",
        draw_dna,
        input_format=".pdb",
        domain="DNA",
        raw_fn=raw_dna,
    )

    rna = from_mmcif_backbone(
        "1EHZ",
        chain_id="A",
        atom_name="P",
        data_dir=DATA_DIR / "mmcif",
        save_coords=True,
    )

    def draw_rna(plotter):
        pts = edge_points(rna.graph)
        meshes = add_curve(
            plotter,
            pts,
            color=EDGE_COLOR,
            add_endpoints=True,
            closed=False,
            direct_closure=True,
        )
        return meshes, pts, rna.issues

    def raw_rna(output_stem):
        atom_points = load_mmcif_atom_points(rna.cif_path, chain_id=rna.chain_id, model_id=rna.model_id)

        def draw_raw(plotter):
            meshes = []
            meshes.extend(add_point_cloud(plotter, atom_points, color="#a8afb3", opacity=0.36, point_size=6.0))
            meshes.extend(add_source_backbone_trace(plotter, rna.coords, color=EDGE_COLOR, radius_scale=0.008))
            return meshes, np.vstack([atom_points, rna.coords])

        return render_raw_scene(draw_raw, output_stem)

    add_panel("tRNA mmCIF", "c_trna_mmcif", draw_rna, input_format=".cif", domain="RNA", raw_fn=raw_rna)

    trefoil_dump = DATA_DIR / "polymers" / "trefoil_polymer_lammps.dump"
    write_lammps_dump(make_torus_knot(n_points=210, p=2, q=3, scale=1.15), trefoil_dump, molecule_id=11)
    polymer = from_lammps_dump(
        trefoil_dump,
        molecule_id=11,
        closed=True,
        closure="direct",
        polymer_id="trefoil_polymer_lammps",
    )

    def draw_polymer(plotter):
        pts = edge_points(polymer.graph)
        meshes = add_curve(plotter, pts, color=EDGE_COLOR, add_endpoints=False, closed=True)
        return meshes, pts, polymer.issues

    def raw_polymer(output_stem):
        def draw_raw(plotter):
            meshes = add_source_beads_and_bonds(plotter, polymer.coords)
            return meshes, polymer.coords

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "Trefoil Polymer LAMMPS",
        "d_trefoil_polymer_lammps",
        draw_polymer,
        input_format="LAMMPS dump",
        domain="polymer",
        raw_fn=raw_polymer,
    )

    cinquefoil_xyz = DATA_DIR / "coordinate_chains" / "cinquefoil_coordinate_chain.xyz"
    write_xyz_coords(make_torus_knot(n_points=260, p=2, q=5, scale=1.05), cinquefoil_xyz, comment="cinquefoil coordinate chain")
    coord = build_curve_from_xyz(
        cinquefoil_xyz,
        closed=True,
        curve_id="cinquefoil_coordinate_chain_xyz",
    )

    def draw_coordinate(plotter):
        pts = edge_points(coord.graph)
        meshes = add_curve(plotter, pts, color=EDGE_COLOR, add_endpoints=False, closed=True)
        return meshes, pts, coord.issues

    def raw_coordinate(output_stem):
        pts = edge_points(coord.graph)

        def draw_raw(plotter):
            meshes = add_source_beads_and_bonds(plotter, pts)
            return meshes, pts

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "Cinquefoil XYZ",
        "e_cinquefoil_xyz",
        draw_coordinate,
        input_format=".xyz",
        domain="coordinate chain",
        raw_fn=raw_coordinate,
    )

    engineering_nodes = DATA_DIR / "spatial_graphs" / "engineering_network_nodes.csv"
    engineering_edges = DATA_DIR / "spatial_graphs" / "engineering_network_edges.csv"
    write_spatial_graph_csv(make_engineering_network_payload(), engineering_nodes, engineering_edges)
    spatial = build_spatial_graph_from_csv(
        engineering_nodes,
        engineering_edges,
        graph_id="engineering_network_csv",
    )

    def draw_spatial(plotter):
        meshes, pts = add_spatial_graph(plotter, spatial.graph)
        return meshes, pts, spatial.issues

    def raw_spatial(output_stem):
        def draw_raw(plotter):
            return add_raw_spatial_graph(plotter, spatial.graph)

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "Engineering Network CSV",
        "f_engineering_network_csv",
        draw_spatial,
        input_format="node/edge CSV",
        domain="spatial graph",
        raw_fn=raw_spatial,
    )

    genus2_surface_ply = DATA_DIR / "surfaces" / "genus2_surface.ply"
    genus2_surface_ply.parent.mkdir(parents=True, exist_ok=True)
    make_genus2_surface_mesh().save(genus2_surface_ply)
    surface = build_surface_from_mesh_file(
        genus2_surface_ply,
        mesh_id="genus2_surface_ply",
    )

    def draw_surface(plotter):
        meshes = add_surface(plotter, surface.mesh, color=SURFACE_COLOR)
        skeleton_meshes = []
        for path in make_theta_paths():
            skeleton = polyline_mesh(path).tube(radius=0.018, n_sides=20, capping=True)
            plotter.add_mesh(skeleton, color="#111111", smooth_shading=True, specular=0.25)
            skeleton_meshes.append(skeleton)
        meshes.extend(skeleton_meshes)
        all_points = np.vstack([surface.mesh.points, *[mesh.points for mesh in skeleton_meshes]])
        return meshes, all_points, surface.issues

    def raw_surface(output_stem):
        def draw_raw(plotter):
            mesh = surface.mesh.triangulate().clean()
            plotter.add_mesh(mesh, color="#687177", style="wireframe", opacity=0.82, line_width=1.0)
            return [mesh], mesh.points

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "Genus-2 Surface Mesh PLY",
        "g_genus2_surface_ply",
        draw_surface,
        input_format=".ply",
        domain="surface mesh",
        raw_fn=raw_surface,
    )

    gyroid_values, gyroid_spacing, gyroid_origin = make_gyroid_pocket_field()
    gyroid_npz = DATA_DIR / "volumetric_fields" / "gyroid_pocket_scalar_volume.npz"
    write_npz_scalar_field(gyroid_values, gyroid_npz, spacing=gyroid_spacing, origin=gyroid_origin)
    volume = build_surface_from_scalar_field_file(
        gyroid_npz,
        level=0.0,
        field_id="gyroid_pocket_scalar_volume_npz",
    )

    def draw_volume(plotter):
        meshes = add_surface(plotter, volume.mesh, color=VOLUME_COLOR)
        return meshes, volume.mesh.points, volume.issues

    def raw_volume(output_stem):
        raw_path = PANEL_DIR / f"{output_stem}_{INSET_PATH_SUFFIX}.png"
        return scalar_slices_image(gyroid_values, raw_path), raw_path

    add_panel(
        "Gyroid Volume NPZ",
        "h_gyroid_volume_npz",
        draw_volume,
        input_format=".npz",
        domain="volumetric field",
        raw_fn=raw_volume,
    )

    fancy_fermi_vtp = DATA_DIR / "fermi_surfaces" / "nodal_line_fermi_surface.vtp"
    fancy_fermi_vtp.parent.mkdir(parents=True, exist_ok=True)
    make_nodal_line_fermi_surface().save(fancy_fermi_vtp)
    fermi = build_surface_from_mesh_file(
        fancy_fermi_vtp,
        mesh_id="nodal_line_fermi_surface",
    )

    def draw_fermi(plotter):
        meshes = add_surface(plotter, fermi.mesh, color=FERMI_COLOR)
        return meshes, fermi.mesh.points, fermi.issues

    def raw_fermi(output_stem):
        def draw_raw(plotter):
            mesh = fermi.mesh.triangulate().clean()
            plotter.add_mesh(mesh, color="#c3c9cc", opacity=0.28, smooth_shading=True)
            plotter.add_mesh(mesh, color="#687177", style="wireframe", opacity=0.78, line_width=0.8)
            return [mesh], mesh.points

        return render_raw_scene(draw_raw, output_stem)

    add_panel(
        "Nodal-Line Fermi VTP",
        "i_nodal_line_fermi_vtp",
        draw_fermi,
        input_format=".vtp",
        domain="Fermi surface",
        raw_fn=raw_fermi,
    )

    png_path, svg_path, pdf_path = assemble_figure(panels)
    summary.extend(
        [
            {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
            {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
            {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
        ]
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print("Publication-style Task 2 input gallery")
    print(f"Panel directory: {PANEL_DIR}")
    print(f"Final PNG: {png_path}")
    print(f"Final SVG: {svg_path}")
    print(f"Final PDF: {pdf_path}")
    print(f"Summary path: {SUMMARY_PATH}")
    print("")
    for item in summary:
        if "title" not in item:
            continue
        print(f"Panel: {item['title']}")
        print(f"Domain: {item['domain']}")
        print(f"Input format: {item['input_format']}")
        print(f"Panel path: {item['panel_path']}")
        print(f"Success: {item['success']}")
        if item["issues"]:
            print("Issues:")
            for issue in item["issues"]:
                print(f"- {issue}")
        else:
            print("Issues: none")
        print("")


if __name__ == "__main__":
    main()
