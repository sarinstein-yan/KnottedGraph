"""Additional dataset gallery for Task 2 input-format prototypes.

This script uses the public input adapters where available. It is a
validation gallery: swap in more datasets, render them, and make sure each
input channel still lands in a curve, spatial graph, or surface object.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv
from pyvista import examples as pv_examples

import knotted_graph as kg


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
SUMMARY_PATH = DATA_DIR / "gallery_summary.json"

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

from coordinate_curve_adapter import (
    build_curve_from_csv,
    build_curve_from_xyz,
    write_csv_coords,
    write_xyz_coords,
)
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


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def graph_edge_points(graph) -> np.ndarray:
    edge_data = next(iter(graph.edges(data=True)))[2]
    return np.asarray(edge_data["pts"], dtype=float)


def set_axes_equal(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_curve_png(graph, title: str, output_stem: str, closed: bool = False) -> Path:
    png_path = FIGURE_DIR / f"{output_stem}.png"
    pts = graph_edge_points(graph)
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0)
    ax.scatter(*pts[0], color="tab:green", s=45, label="start")
    if not closed:
        ax.scatter(*pts[-1], color="tab:red", s=45, label="end")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    set_axes_equal(ax, pts)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def write_graph_html(graph, title: str, output_stem: str) -> Path:
    html_path = FIGURE_DIR / f"{output_stem}_graph.html"
    fig = kg.plot_3D_graph_plotly(graph)
    fig.update_layout(title=title)
    fig.write_html(str(html_path))
    return html_path


def plot_spatial_graph_png(graph, title: str, output_stem: str) -> Path:
    png_path = FIGURE_DIR / f"{output_stem}.png"
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    edge_points = []
    for _, _, key, data in graph.edges(keys=True, data=True):
        pts = np.asarray(data["pts"], dtype=float)
        edge_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0, label=key)
    node_pts = np.asarray([data["pos"] for _, data in graph.nodes(data=True)])
    ax.scatter(node_pts[:, 0], node_pts[:, 1], node_pts[:, 2], color="tab:red", s=55)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=7)
    set_axes_equal(ax, np.vstack(edge_points))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def mesh_bounds_span(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray, float]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    mins = np.array([bounds[0], bounds[2], bounds[4]])
    maxs = np.array([bounds[1], bounds[3], bounds[5]])
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    return mins, maxs, span


def render_mesh(mesh: pv.PolyData, output_stem: str, color: str = "#2484c6") -> tuple[Path, Path, Path, list[str]]:
    png_path = FIGURE_DIR / f"{output_stem}.png"
    html_path = FIGURE_DIR / f"{output_stem}.html"
    svg_path = FIGURE_DIR / f"{output_stem}.svg"
    mins, maxs, span = mesh_bounds_span(mesh)
    center = 0.5 * (mins + maxs)

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=0.92,
        smooth_shading=True,
        specular=0.35,
        specular_power=18,
        name=output_stem,
    )
    plotter.add_bounding_box(color="black", line_width=1)
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        xtitle="X",
        ytitle="Y",
        ztitle="Z",
        color="black",
        font_size=12,
    )
    camera_distance = 2.6 * span
    plotter.camera_position = [
        center + np.array([1.2, -1.55, 1.05]) * camera_distance,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.3)

    issues = []
    try:
        plotter.export_html(str(html_path))
    except Exception as exc:  # pragma: no cover
        issues.append(f"HTML export failed: {exc}")
    try:
        plotter.screenshot(str(png_path))
    except Exception as exc:  # pragma: no cover
        issues.append(f"PNG screenshot failed: {exc}")
    try:
        plotter.save_graphic(str(svg_path))
    except Exception as exc:  # pragma: no cover
        issues.append(f"SVG export failed: {exc}")
    plotter.close()
    return png_path, html_path, svg_path, issues


def add_summary(
    summary: list[dict],
    *,
    name: str,
    domain: str,
    source_format: str,
    output_type: str,
    paths: list[Path],
    count: int | None = None,
    issues: list[str] | None = None,
) -> None:
    summary.append(
        {
            "name": name,
            "domain": domain,
            "source_format": source_format,
            "output_type": output_type,
            "count": count,
            "paths": [str(path) for path in paths],
            "success": all(path.exists() and path.stat().st_size > 0 for path in paths),
            "issues": issues or [],
        }
    )


def make_brownian_chain(n_points: int = 150) -> np.ndarray:
    rng = np.random.default_rng(20260520)
    steps = rng.normal(size=(n_points, 3))
    steps[:, 2] += 0.06
    coords = np.cumsum(steps, axis=0)
    coords -= coords.mean(axis=0)
    return coords


def make_torus_knot(n_points: int = 180) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    major = 2.2 + 0.65 * np.cos(3.0 * t)
    return np.column_stack(
        [
            major * np.cos(2.0 * t),
            major * np.sin(2.0 * t),
            0.65 * np.sin(3.0 * t),
        ]
    )


def make_polymer_melt_chain(n_points: int = 132) -> np.ndarray:
    t = np.linspace(0.0, 7.5 * np.pi, n_points)
    return np.column_stack(
        [
            np.cos(t) + 0.18 * np.cos(3.0 * t),
            np.sin(t) + 0.15 * np.sin(2.0 * t),
            np.linspace(-2.0, 2.0, n_points),
        ]
    )


def make_spatial_lattice_payload() -> dict:
    nodes = {
        "n0": [-1.0, -1.0, 0.0],
        "n1": [1.0, -1.0, 0.25],
        "n2": [1.0, 1.0, -0.15],
        "n3": [-1.0, 1.0, 0.2],
        "hub": [0.0, 0.0, 1.05],
    }

    def edge_points(source, target, lift):
        start = np.asarray(nodes[source], dtype=float)
        end = np.asarray(nodes[target], dtype=float)
        t = np.linspace(0.0, 1.0, 40)
        base = (1.0 - t)[:, None] * start + t[:, None] * end
        base[:, 2] += lift * np.sin(np.pi * t)
        return base.tolist()

    edges = []
    for index, (source, target, lift) in enumerate(
        [
            ("n0", "n1", 0.25),
            ("n1", "n2", -0.15),
            ("n2", "n3", 0.2),
            ("n3", "n0", -0.1),
            ("n0", "hub", 0.35),
            ("n1", "hub", -0.2),
            ("n2", "hub", 0.25),
            ("n3", "hub", -0.15),
        ]
    ):
        edges.append(
            {
                "id": f"edge_{index}",
                "source": source,
                "target": target,
                "points": edge_points(source, target, lift),
            }
        )
    return {
        "graph_id": "spatial_lattice_csv_gallery",
        "nodes": nodes,
        "edges": edges,
    }


def make_torus_levelset_field(n: int = 76):
    axis = np.linspace(-1.8, 1.8, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    major_radius = 0.95
    minor_radius = 0.32
    values = (np.sqrt(x * x + y * y) - major_radius) ** 2 + z * z - minor_radius**2
    spacing_value = float(axis[1] - axis[0])
    return values, (spacing_value, spacing_value, spacing_value), (float(axis[0]),) * 3


def save_bunny_dataset(path: Path) -> tuple[str, list[str]]:
    issues = []
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bunny = pv_examples.download_bunny_coarse().triangulate().clean()
        source = "pyvista_examples_stanford_bunny_coarse"
    except Exception as exc:  # pragma: no cover - depends on network/cache
        issues.append(f"PyVista bunny download failed; used sphere fallback: {exc}")
        bunny = pv.Sphere(theta_resolution=64, phi_resolution=32).triangulate().clean()
        source = "fallback_generated_sphere"
    bunny.save(path)
    return source, issues


def main() -> None:
    ensure_dirs()
    summary = []

    print("Additional supported-input dataset gallery")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")

    protein = from_protein_ca_backbone(
        "1UBQ",
        chain_id="A",
        data_dir=DATA_DIR / "proteins",
        save_coords=True,
    )
    protein_png = plot_curve_png(
        protein.graph,
        "1UBQ ubiquitin C-alpha trace from PDB",
        "protein_1ubq_pdb_ca",
    )
    protein_html = write_graph_html(protein.graph, "1UBQ PDB CA trace", "protein_1ubq_pdb_ca")
    add_summary(
        summary,
        name="1UBQ ubiquitin C-alpha trace",
        domain="protein",
        source_format="pdb",
        output_type="MultiGraph(pos/pts)",
        count=protein.coords.shape[0],
        paths=[protein_png, protein_html],
        issues=protein.issues,
    )

    dna = from_nucleic_acid_backbone(
        "1BNA",
        chain_id="B",
        atom_name="P",
        data_dir=DATA_DIR / "dna",
        save_coords=True,
    )
    dna_png = plot_curve_png(dna.graph, "1BNA chain B DNA phosphate trace", "dna_1bna_chainB_pdb_p")
    dna_html = write_graph_html(dna.graph, "1BNA chain B DNA P trace", "dna_1bna_chainB_pdb_p")
    add_summary(
        summary,
        name="1BNA chain B phosphate trace",
        domain="DNA",
        source_format="pdb",
        output_type="MultiGraph(pos/pts)",
        count=dna.coords.shape[0],
        paths=[dna_png, dna_html],
        issues=dna.issues,
    )

    mmcif_protein = from_mmcif_backbone(
        "1UBQ",
        chain_id="A",
        atom_name="CA",
        data_dir=DATA_DIR / "mmcif",
        save_coords=True,
    )
    mmcif_protein_png = plot_curve_png(
        mmcif_protein.graph,
        "1UBQ ubiquitin C-alpha trace from mmCIF",
        "protein_1ubq_mmcif_ca",
    )
    mmcif_protein_html = write_graph_html(
        mmcif_protein.graph,
        "1UBQ mmCIF CA trace",
        "protein_1ubq_mmcif_ca",
    )
    add_summary(
        summary,
        name="1UBQ ubiquitin C-alpha trace",
        domain="protein",
        source_format="mmcif",
        output_type="MultiGraph(pos/pts)",
        count=mmcif_protein.coords.shape[0],
        paths=[mmcif_protein_png, mmcif_protein_html],
        issues=mmcif_protein.issues,
    )

    mmcif_rna = from_mmcif_backbone(
        "1EHZ",
        chain_id="A",
        atom_name="P",
        data_dir=DATA_DIR / "mmcif",
        save_coords=True,
    )
    mmcif_rna_png = plot_curve_png(
        mmcif_rna.graph,
        "1EHZ tRNA phosphate trace from mmCIF",
        "rna_1ehz_mmcif_p",
    )
    mmcif_rna_html = write_graph_html(mmcif_rna.graph, "1EHZ mmCIF RNA P trace", "rna_1ehz_mmcif_p")
    add_summary(
        summary,
        name="1EHZ tRNA phosphate trace",
        domain="RNA",
        source_format="mmcif",
        output_type="MultiGraph(pos/pts)",
        count=mmcif_rna.coords.shape[0],
        paths=[mmcif_rna_png, mmcif_rna_html],
        issues=mmcif_rna.issues,
    )

    coordinate_csv = DATA_DIR / "coordinate_chains" / "brownian_polymer_chain.csv"
    coordinate_xyz = DATA_DIR / "coordinate_chains" / "closed_torus_knot.xyz"
    write_csv_coords(make_brownian_chain(), coordinate_csv)
    write_xyz_coords(make_torus_knot(), coordinate_xyz, comment="closed torus-knot-like ring")
    brownian = build_curve_from_csv(coordinate_csv, closed=False, curve_id="brownian_polymer_chain_csv")
    torus_knot = build_curve_from_xyz(coordinate_xyz, closed=True, curve_id="closed_torus_knot_xyz")
    for result, title in [
        (brownian, "Brownian polymer chain from CSV"),
        (torus_knot, "Closed torus-knot-like ring from XYZ"),
    ]:
        png = plot_curve_png(result.graph, title, result.curve_id, closed=result.closed)
        html = write_graph_html(result.graph, title, result.curve_id)
        add_summary(
            summary,
            name=result.curve_id,
            domain="coordinate_chain",
            source_format=result.source_format,
            output_type="MultiGraph(pos/pts)",
            count=result.coords.shape[0],
            paths=[png, html],
            issues=result.issues,
        )

    lammps_path = DATA_DIR / "polymers" / "polymer_melt_chain.dump"
    gro_path = DATA_DIR / "polymers" / "torus_knot_ring.gro"
    write_lammps_dump(make_polymer_melt_chain(), lammps_path, molecule_id=3)
    write_gro_coords(make_torus_knot(150), gro_path, residue_name="KNT", atom_name="BB")
    lammps = from_lammps_dump(
        lammps_path,
        molecule_id=3,
        closed=False,
        polymer_id="polymer_melt_chain_lammps",
    )
    gro = from_gromacs_gro(
        gro_path,
        atom_name="BB",
        residue_name="KNT",
        closed=True,
        closure="direct",
        polymer_id="torus_knot_ring_gro",
    )
    for result, title in [
        (lammps, "Polymer melt chain from LAMMPS dump"),
        (gro, "Closed polymer ring from GROMACS GRO"),
    ]:
        png = plot_curve_png(result.graph, title, result.polymer_id, closed=result.closed)
        html = write_graph_html(result.graph, title, result.polymer_id)
        add_summary(
            summary,
            name=result.polymer_id,
            domain="polymer",
            source_format=result.source_format,
            output_type="MultiGraph(pos/pts)",
            count=result.coords.shape[0],
            paths=[png, html],
            issues=result.issues,
        )

    nodes_csv = DATA_DIR / "spatial_graphs" / "spatial_lattice_nodes.csv"
    edges_csv = DATA_DIR / "spatial_graphs" / "spatial_lattice_edges.csv"
    write_spatial_graph_csv(make_spatial_lattice_payload(), nodes_csv, edges_csv)
    spatial = build_spatial_graph_from_csv(
        nodes_csv,
        edges_csv,
        graph_id="spatial_lattice_csv_gallery",
    )
    spatial_png = plot_spatial_graph_png(
        spatial.graph,
        "Spatial lattice from node/edge CSV",
        spatial.graph_id,
    )
    spatial_html = write_graph_html(spatial.graph, "Spatial lattice CSV graph", spatial.graph_id)
    add_summary(
        summary,
        name=spatial.graph_id,
        domain="spatial_graph",
        source_format="node_edge_csv",
        output_type="MultiGraph(pos/pts)",
        count=spatial.graph.number_of_edges(),
        paths=[spatial_png, spatial_html],
        issues=spatial.issues,
    )

    bunny_path = DATA_DIR / "surfaces" / "stanford_bunny_coarse.ply"
    bunny_source, bunny_source_issues = save_bunny_dataset(bunny_path)
    bunny = build_surface_from_mesh_file(bunny_path, mesh_id="stanford_bunny_coarse_ply")
    bunny_png, bunny_html, bunny_svg, bunny_render_issues = render_mesh(
        bunny.mesh,
        "stanford_bunny_coarse_ply",
        color="#12a47f",
    )
    add_summary(
        summary,
        name="Stanford bunny coarse mesh",
        domain="surface_mesh",
        source_format=f"ply:{bunny_source}",
        output_type="PyVista PolyData",
        count=bunny.mesh.n_cells,
        paths=[bunny_png, bunny_html, bunny_svg],
        issues=bunny_source_issues + bunny.issues + bunny_render_issues,
    )

    torus_values, torus_spacing, torus_origin = make_torus_levelset_field()
    torus_path = DATA_DIR / "volumetric_fields" / "torus_levelset_scalar_field.npz"
    write_npz_scalar_field(
        torus_values,
        torus_path,
        spacing=torus_spacing,
        origin=torus_origin,
    )
    torus_volume = build_surface_from_scalar_field_file(
        torus_path,
        level=0.0,
        field_id="torus_levelset_scalar_field_npz",
    )
    torus_mesh_path = DATA_DIR / "volumetric_fields" / "torus_levelset_scalar_field_isosurface.vtp"
    torus_volume.mesh.save(torus_mesh_path)
    torus_png, torus_html, torus_svg, torus_render_issues = render_mesh(
        torus_volume.mesh,
        "torus_levelset_scalar_field_npz",
        color="#b15fbd",
    )
    add_summary(
        summary,
        name="Torus level-set scalar field isosurface",
        domain="volumetric_field",
        source_format="npz",
        output_type="PyVista PolyData",
        count=torus_volume.mesh.n_cells,
        paths=[torus_mesh_path, torus_png, torus_html, torus_svg],
        issues=torus_volume.issues + torus_render_issues,
    )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    for item in summary:
        print(f"Name: {item['name']}")
        print(f"Domain: {item['domain']}")
        print(f"Source format: {item['source_format']}")
        print(f"Output type: {item['output_type']}")
        print(f"Count: {item['count']}")
        print(f"Success: {item['success']}")
        print("Paths:")
        for path in item["paths"]:
            print(f"- {path}")
        if item["issues"]:
            print("Issues:")
            for issue in item["issues"]:
                print(f"- {issue}")
        else:
            print("Issues: none")
        print("")
    print(f"Summary path: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
