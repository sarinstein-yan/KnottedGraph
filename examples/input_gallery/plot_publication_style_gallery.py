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


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
ROOT = EXAMPLES_DIR.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PANEL_DIR = FIGURE_DIR / "publication_style_panels"
SUMMARY_PATH = DATA_DIR / "publication_style_gallery_summary.json"

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
SHADOW_COLOR = "#9aa3a8"


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


def make_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1600))
    plotter.set_background("white")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    return plotter


def polyline_mesh(points: np.ndarray) -> pv.PolyData:
    pts = np.asarray(points, dtype=float)
    poly = pv.PolyData(pts)
    poly.lines = np.concatenate(([pts.shape[0]], np.arange(pts.shape[0])))
    return poly


def add_projected_shadows(
    plotter: pv.Plotter,
    meshes: list[pv.DataSet],
    points: np.ndarray,
    *,
    opacity: float = 0.13,
) -> None:
    mins, _, _, span = point_span(points)
    pad = 0.08 * span
    origins = (
        np.array([mins[0] - pad, 0.0, 0.0]),
        np.array([0.0, mins[1] - pad, 0.0]),
        np.array([0.0, 0.0, mins[2] - pad]),
    )
    for mesh in meshes:
        if mesh is None:
            continue
        for normal, origin in zip(np.eye(3), origins):
            projected = mesh.project_points_to_plane(normal=normal, origin=origin)
            plotter.add_mesh(
                projected,
                color=SHADOW_COLOR,
                opacity=opacity,
                smooth_shading=True,
                specular=0.1,
            )


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
    add_projected_shadows(plotter, meshes, points)
    set_camera(plotter, points)
    image_path = PANEL_DIR / f"{output_stem}.png"
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    return crop_white(image, threshold=249, pad=10), image_path, issues


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
        image_ax = fig.add_subplot(inner[1, 0])
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
        image_ax.set_box_aspect(1)

    base = FIGURE_DIR / "task2_input_gallery_publication_style"
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

    def add_panel(title: str, output_stem: str, draw_fn, *, input_format: str, domain: str):
        image, panel_path, issues = render_panel(draw_fn, output_stem)
        panels.append({"title": title, "image": image, "panel_path": panel_path})
        summary.append(
            {
                "title": title,
                "domain": domain,
                "input_format": input_format,
                "panel_path": str(panel_path),
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

    def make_k4_payload() -> dict:
        nodes = {
            "A": [1.25, 1.05, 0.0],
            "B": [-1.25, 1.05, 0.25],
            "C": [-0.85, -1.15, -0.20],
            "D": [1.05, -1.05, 0.35],
        }

        def arc(source, target, lift, side):
            start = np.asarray(nodes[source], dtype=float)
            end = np.asarray(nodes[target], dtype=float)
            t = np.linspace(0.0, 1.0, 54)
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

        pairs = [
            ("A", "B", 0.35, 0.25),
            ("A", "C", -0.25, 0.30),
            ("A", "D", 0.20, -0.35),
            ("B", "C", 0.25, -0.25),
            ("B", "D", -0.35, 0.28),
            ("C", "D", 0.30, 0.18),
        ]
        return {
            "graph_id": "k4_spatial_network_csv",
            "nodes": nodes,
            "edges": [
                {"id": f"{u}{v}", "source": u, "target": v, "points": arc(u, v, lift, side)}
                for u, v, lift, side in pairs
            ],
        }

    def make_trefoil_tube_surface_mesh() -> pv.PolyData:
        """Create a closed trefoil tube surface for the PLY mesh panel."""
        pts = make_torus_knot(n_points=280, p=2, q=3, scale=1.0)
        closed_pts = np.vstack([pts, pts[0]])
        poly = polyline_mesh(closed_pts)
        return poly.tube(radius=0.13, n_sides=36, capping=True).triangulate().clean()

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

    def make_rashba_fermi_surface() -> pv.PolyData:
        """Create two spin-split torus-like Fermi-surface sheets."""
        n_major = 150
        n_minor = 54
        u_values = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
        v_values = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
        points = []
        faces = []

        def add_sheet(major_radius: float, minor_radius: float, z_shift: float, phase: float) -> None:
            offset = len(points)
            for u in u_values:
                local_major = major_radius + 0.10 * np.cos(2.0 * u + phase)
                for v in v_values:
                    local_minor = minor_radius * (
                        1.0
                        + 0.10 * np.cos(3.0 * u - phase)
                        + 0.05 * np.sin(2.0 * v + u)
                    )
                    x = (local_major + local_minor * np.cos(v)) * np.cos(u)
                    y = (local_major + local_minor * np.cos(v)) * np.sin(u)
                    z = 0.70 * local_minor * np.sin(v) + z_shift + 0.06 * np.sin(3.0 * u + phase)
                    points.append([x, y, z])
            for i in range(n_major):
                for j in range(n_minor):
                    a = offset + i * n_minor + j
                    b = offset + ((i + 1) % n_major) * n_minor + j
                    c = offset + ((i + 1) % n_major) * n_minor + ((j + 1) % n_minor)
                    d = offset + i * n_minor + ((j + 1) % n_minor)
                    faces.extend([4, a, b, c, d])

        add_sheet(1.05, 0.28, 0.16, 0.0)
        add_sheet(0.70, 0.20, -0.16, np.pi / 5.0)
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

    add_panel("Knotted Protein PDB", "a_knotted_protein_pdb", draw_protein, input_format=".pdb", domain="protein")

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

    add_panel("DNA Double Helix PDB", "b_dna_double_helix_pdb", draw_dna, input_format=".pdb", domain="DNA")

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

    add_panel("tRNA mmCIF", "c_trna_mmcif", draw_rna, input_format=".cif", domain="RNA")

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

    add_panel("Trefoil Polymer LAMMPS", "d_trefoil_polymer_lammps", draw_polymer, input_format="LAMMPS dump", domain="polymer")

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

    add_panel("Cinquefoil XYZ", "e_cinquefoil_xyz", draw_coordinate, input_format=".xyz", domain="coordinate chain")

    k4_nodes = DATA_DIR / "spatial_graphs" / "k4_spatial_network_nodes.csv"
    k4_edges = DATA_DIR / "spatial_graphs" / "k4_spatial_network_edges.csv"
    write_spatial_graph_csv(make_k4_payload(), k4_nodes, k4_edges)
    spatial = build_spatial_graph_from_csv(
        k4_nodes,
        k4_edges,
        graph_id="k4_spatial_network_csv",
    )

    def draw_spatial(plotter):
        meshes, pts = add_spatial_graph(plotter, spatial.graph)
        return meshes, pts, spatial.issues

    add_panel("K4 Spatial Graph CSV", "f_k4_spatial_graph_csv", draw_spatial, input_format="node/edge CSV", domain="spatial graph")

    trefoil_surface_ply = DATA_DIR / "surfaces" / "trefoil_tube_surface.ply"
    trefoil_surface_ply.parent.mkdir(parents=True, exist_ok=True)
    make_trefoil_tube_surface_mesh().save(trefoil_surface_ply)
    surface = build_surface_from_mesh_file(
        trefoil_surface_ply,
        mesh_id="trefoil_tube_surface_ply",
    )

    def draw_surface(plotter):
        meshes = add_surface(plotter, surface.mesh, color=SURFACE_COLOR)
        return meshes, surface.mesh.points, surface.issues

    add_panel("Trefoil Tube PLY", "g_trefoil_tube_ply", draw_surface, input_format=".ply", domain="surface mesh")

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

    add_panel("Gyroid Volume NPZ", "h_gyroid_volume_npz", draw_volume, input_format=".npz", domain="volumetric field")

    fancy_fermi_vtp = DATA_DIR / "fermi_surfaces" / "rashba_split_fermi_surface.vtp"
    fancy_fermi_vtp.parent.mkdir(parents=True, exist_ok=True)
    make_rashba_fermi_surface().save(fancy_fermi_vtp)
    fermi = build_surface_from_mesh_file(
        fancy_fermi_vtp,
        mesh_id="rashba_split_fermi_surface",
    )

    def draw_fermi(plotter):
        meshes = add_surface(plotter, fermi.mesh, color=FERMI_COLOR)
        return meshes, fermi.mesh.points, fermi.issues

    add_panel("Rashba Fermi VTP", "i_rashba_fermi_vtp", draw_fermi, input_format=".vtp", domain="Fermi surface")

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
