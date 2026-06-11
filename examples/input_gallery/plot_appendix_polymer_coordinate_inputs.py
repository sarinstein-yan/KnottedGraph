"""Appendix polymer and coordinate-chain input figure for Task 2.

This grouped appendix figure shows polymer simulation snapshots and lightweight
coordinate-chain files. Each panel displays a source-domain bead/chain view
above the converted graph-compatible curve used by the input workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PANEL_DIR = FIGURE_DIR / "appendix_polymer_coordinate_panels"
SUMMARY_PATH = DATA_DIR / "appendix_polymer_coordinate_inputs_summary.json"
FIGURE_STEM = "appendix_polymer_coordinate_inputs"

sys.path.insert(0, str(HERE))

from knotted_graph.inputs import (  # noqa: E402
    from_coordinate_chain,
    from_gromacs_gro,
    from_lammps_dump,
    write_gro_coords,
    write_lammps_dump,
)
from knotted_graph.inputs.coordinate_chain import write_xyz_coords  # noqa: E402
from plot_main_text_input_figure import (  # noqa: E402
    add_curve,
    label_with_format,
    make_lammps_polymer_curve,
    make_torus_knot,
    render_scene,
    render_source,
)
from compact_appendix_layout import compact_panel_bboxes, draw_compact_panel  # noqa: E402
from plot_publication_style_gallery import (  # noqa: E402
    add_source_beads_and_bonds,
    edge_points,
)
from plot_publication_style_gallery_sets import (  # noqa: E402
    make_bottlebrush,
    make_figure_eight,
    make_helix,
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
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for old_panel in PANEL_DIR.glob("*.png"):
        old_panel.unlink()


def write_dat_coords(points: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# x y z"]
    lines.extend(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in np.asarray(points, dtype=float))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_csv_coords(points: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["x,y,z"]
    lines.extend(f"{x:.8f},{y:.8f},{z:.8f}" for x, y, z in np.asarray(points, dtype=float))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_json_coords(points: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"points": np.asarray(points, dtype=float).round(8).tolist()}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_tsv_coords(points: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["x\ty\tz"]
    lines.extend(f"{x:.8f}\t{y:.8f}\t{z:.8f}" for x, y, z in np.asarray(points, dtype=float))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_txt_coords(points: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# x y z"]
    lines.extend(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in np.asarray(points, dtype=float))
    path.write_text("\n".join(lines) + "\n")
    return path


def make_wavy_trace(n_points: int = 230) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points)
    x = -2.0 + 4.0 * t
    y = 0.55 * np.sin(5.0 * np.pi * t)
    z = 0.35 * np.cos(3.0 * np.pi * t) + 0.38 * t
    return np.column_stack([x, y, z])


def make_lissajous_loop(n_points: int = 260) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.column_stack(
        [
            1.25 * np.sin(3.0 * t + 0.30),
            0.92 * np.sin(2.0 * t),
            0.55 * np.sin(5.0 * t + 0.60),
        ]
    )


def make_numpy_ribbon_loop(n_points: int = 260) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = 1.02 + 0.25 * np.cos(3.0 * t)
    return np.column_stack(
        [
            radius * np.cos(t),
            0.74 * np.sin(2.0 * t),
            0.48 * np.sin(t) + 0.20 * np.sin(4.0 * t),
        ]
    )


def make_meander_chain(n_points: int = 240) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points)
    return np.column_stack(
        [
            -1.7 + 3.4 * t,
            0.48 * np.sin(8.0 * np.pi * t),
            0.30 * np.sin(2.0 * np.pi * t) + 0.22 * np.cos(6.0 * np.pi * t),
        ]
    )


def make_plain_text_zigzag(n_points: int = 210) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points)
    x = -1.75 + 3.50 * t
    triangular = 2.0 * np.abs(2.0 * ((4.0 * t) % 1.0) - 1.0) - 1.0
    y = 0.48 * triangular
    z = 0.30 * np.sin(2.0 * np.pi * t) + 0.42 * np.sin(np.pi * t)
    return np.column_stack([x, y, z])


def result_points(result) -> np.ndarray:
    return edge_points(result.graph)


def add_curve_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    result,
    input_format: str,
    domain: str,
    closed: bool,
) -> None:
    def draw_converted(plotter):
        pts = result_points(result)
        meshes = add_curve(plotter, pts, closed=closed, add_endpoints=not closed)
        return meshes, pts, result.issues

    def draw_source(plotter):
        meshes = add_source_beads_and_bonds(plotter, result.coords)
        return meshes, result.coords

    converted, converted_path, issues = render_scene(draw_converted, PANEL_DIR / f"{stem}_converted.png")
    source, source_path = render_source(draw_source, PANEL_DIR / f"{stem}_source.png")
    panels.append({"title": title, "source_image": source, "converted_image": converted})
    summary.append(
        {
            "title": title,
            "domain": domain,
            "input_format": input_format,
            "source_path": str(result.source_path),
            "source_view_path": str(source_path),
            "converted_view_path": str(converted_path),
            "point_count": int(result.coords.shape[0]),
            "graph_node_count": result.graph.number_of_nodes(),
            "graph_edge_count": result.graph.number_of_edges(),
            "closed": bool(closed),
            "success": source_path.exists() and converted_path.exists(),
            "issues": issues,
            "yamada_status": "pending downstream audit",
        }
    )


def assemble_figure(panels: list[dict]) -> tuple[Path, Path, Path]:
    labels = [f"({chr(ord('a') + i)})" for i in range(len(panels))]
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig = plt.figure(figsize=(18.2, 4.35 * rows), facecolor="white")
    bboxes = compact_panel_bboxes(len(panels), rows=rows, cols=cols, gap_x=0.004, gap_y=0.006)
    for i, (panel, bbox) in enumerate(zip(panels, bboxes)):
        draw_compact_panel(
            fig,
            bbox,
            label=labels[i],
            title=panel["title"],
            source_image=panel["source_image"],
            result_image=panel["converted_image"],
            result_label="converted graph",
        )

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
    panels: list[dict] = []
    summary: list[dict] = []
    data_dir = DATA_DIR / "appendix_polymer_coordinate"

    ring_path = data_dir / "ring_polymer.gro"
    write_gro_coords(make_torus_knot(n_points=230, p=1, q=2, scale=0.88), ring_path, title="ring polymer appendix source snapshot")
    ring = from_gromacs_gro(ring_path, closed=True, closure="direct", polymer_id="ring_polymer_gro")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Ring Polymer", "GRO"),
        stem="a_ring_polymer_gro",
        result=ring,
        input_format=".gro",
        domain="GROMACS polymer snapshot",
        closed=True,
    )

    polymer_path = data_dir / "polymer_lammps.dump"
    write_lammps_dump(make_lammps_polymer_curve(n_points=280, scale=1.15), polymer_path, molecule_id=7)
    polymer = from_lammps_dump(polymer_path, molecule_id=7, closed=True, closure="direct", polymer_id="polymer_lammps")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Polymer", "LAMMPS"),
        stem="b_polymer_lammps",
        result=polymer,
        input_format="LAMMPS dump",
        domain="LAMMPS polymer snapshot",
        closed=True,
    )

    trefoil_path = data_dir / "trefoil_polymer.dump"
    write_lammps_dump(make_torus_knot(n_points=250, p=2, q=3, scale=1.0), trefoil_path, molecule_id=3)
    trefoil = from_lammps_dump(trefoil_path, molecule_id=3, closed=True, closure="direct", polymer_id="trefoil_polymer")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Trefoil Polymer", "LAMMPS"),
        stem="c_trefoil_polymer_lammps",
        result=trefoil,
        input_format="LAMMPS dump",
        domain="LAMMPS polymer snapshot",
        closed=True,
    )

    bottlebrush_path = data_dir / "bottlebrush_polymer.xyz"
    write_xyz_coords(make_bottlebrush(), bottlebrush_path, comment="bottlebrush polymer coordinate chain")
    bottlebrush = from_coordinate_chain(bottlebrush_path, closed=False, input_id="bottlebrush_polymer_xyz")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Bottlebrush Polymer", "XYZ"),
        stem="d_bottlebrush_polymer_xyz",
        result=bottlebrush,
        input_format=".xyz",
        domain="polymer coordinate chain",
        closed=False,
    )

    cable_path = data_dir / "coiled_cable.dat"
    write_dat_coords(make_helix(n_points=260, turns=7.0, radius=0.9, pitch=0.12), cable_path)
    cable = from_coordinate_chain(cable_path, source_format="dat", closed=False, input_id="coiled_cable_dat")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Coiled Cable", "DAT"),
        stem="e_coiled_cable_dat",
        result=cable,
        input_format=".dat",
        domain="cable harness coordinate chain",
        closed=False,
    )

    cinquefoil_path = data_dir / "cinquefoil_knot.xyz"
    write_xyz_coords(make_torus_knot(n_points=260, p=2, q=5, scale=1.05), cinquefoil_path, comment="cinquefoil coordinate knot")
    cinquefoil = from_coordinate_chain(cinquefoil_path, closed=True, closure="direct", input_id="cinquefoil_knot_xyz")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Cinquefoil Knot", "XYZ"),
        stem="f_cinquefoil_knot_xyz",
        result=cinquefoil,
        input_format=".xyz",
        domain="coordinate knot",
        closed=True,
    )

    figure_eight_path = data_dir / "figure_eight_knot.xyz"
    write_xyz_coords(make_figure_eight(), figure_eight_path, comment="figure-eight coordinate knot")
    figure_eight = from_coordinate_chain(figure_eight_path, closed=True, closure="direct", input_id="figure_eight_knot_xyz")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Figure-Eight Knot", "XYZ"),
        stem="g_figure_eight_knot_xyz",
        result=figure_eight,
        input_format=".xyz",
        domain="coordinate knot",
        closed=True,
    )

    csv_path = data_dir / "wavy_sensor_trace.csv"
    write_csv_coords(make_wavy_trace(), csv_path)
    csv_trace = from_coordinate_chain(csv_path, closed=False, input_id="wavy_sensor_trace_csv")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Sensor Trace", "CSV"),
        stem="h_sensor_trace_csv",
        result=csv_trace,
        input_format=".csv",
        domain="coordinate-chain CSV",
        closed=False,
    )

    json_path = data_dir / "lissajous_loop.json"
    write_json_coords(make_lissajous_loop(), json_path)
    json_loop = from_coordinate_chain(json_path, closed=True, closure="direct", input_id="lissajous_loop_json")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Lissajous Loop", "JSON"),
        stem="i_lissajous_loop_json",
        result=json_loop,
        input_format=".json",
        domain="coordinate-chain JSON",
        closed=True,
    )

    npy_path = data_dir / "fiber_coil.npy"
    np.save(npy_path, make_numpy_ribbon_loop())
    npy_coil = from_coordinate_chain(npy_path, closed=True, closure="direct", input_id="fiber_coil_npy")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Ribbon Loop", "NPY"),
        stem="j_fiber_coil_npy",
        result=npy_coil,
        input_format=".npy",
        domain="NumPy coordinate array",
        closed=True,
    )

    tsv_path = data_dir / "meander_path.tsv"
    write_tsv_coords(make_meander_chain(), tsv_path)
    tsv_trace = from_coordinate_chain(tsv_path, closed=False, input_id="meander_path_tsv")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Meander Path", "TSV"),
        stem="k_meander_path_tsv",
        result=tsv_trace,
        input_format=".tsv",
        domain="coordinate-chain TSV",
        closed=False,
    )

    txt_path = data_dir / "plain_text_cable.txt"
    write_txt_coords(make_plain_text_zigzag(), txt_path)
    txt_trace = from_coordinate_chain(txt_path, source_format="txt", closed=False, input_id="plain_text_cable_txt")
    add_curve_panel(
        panels,
        summary,
        title=label_with_format("Plain Text Cable", "TXT"),
        stem="l_plain_text_cable_txt",
        result=txt_trace,
        input_format=".txt",
        domain="plain-text coordinate chain",
        closed=False,
    )

    png_path, svg_path, pdf_path = assemble_figure(panels)
    summary.extend(
        [
            {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
            {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
            {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
        ]
    )
    SUMMARY_PATH.write_text(json.dumps({"figure": "Appendix S2 polymer and coordinate-chain inputs", "panels": summary}, indent=2) + "\n")

    print("Appendix S2 polymer and coordinate-chain input figure")
    print(f"Panel directory: {PANEL_DIR}")
    print(f"Final PNG: {png_path}")
    print(f"Final SVG: {svg_path}")
    print(f"Final PDF: {pdf_path}")
    print(f"Summary path: {SUMMARY_PATH}")
    for item in summary:
        if "title" in item:
            print(
                f"Panel: {item['title']} success={item['success']} "
                f"points={item['point_count']} closed={item['closed']} "
                f"issues={item['issues'] or 'none'}"
            )


if __name__ == "__main__":
    main()
