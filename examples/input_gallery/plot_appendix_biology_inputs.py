"""Appendix biology input figure for Task 2.

This grouped appendix figure shows multiple biological examples rather than one
example per input type.  Each panel displays a source-domain molecular view
above the converted backbone graph used by the input workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PANEL_DIR = FIGURE_DIR / "appendix_biology_panels"
SUMMARY_PATH = DATA_DIR / "appendix_biology_inputs_summary.json"
FIGURE_STEM = "appendix_biology_inputs"

sys.path.insert(0, str(HERE))

from knotted_graph.inputs import (  # noqa: E402
    from_mmcif_backbone,
    from_nucleic_acid_backbone,
    from_protein_ca_backbone,
)
from plot_main_text_input_figure import (  # noqa: E402
    BIOLOGY_SEGMENT_COLORS,
    SOURCE_GREY,
    add_endpoint_nodes,
    add_segmented_curve,
    add_tube,
    render_scene,
    render_source,
    write_image,
)
from compact_appendix_layout import compact_panel_bboxes, draw_compact_panel  # noqa: E402
from plot_publication_style_gallery import (  # noqa: E402
    EDGE_COLOR,
    EDGE_COLOR_2,
    NODE_COLOR,
    add_point_cloud,
    crop_white,
    edge_points,
    load_mmcif_atom_points,
    load_pdb_atom_points,
    make_plotter,
    point_span,
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

CHAIN_COLORS = ["#2563A8", "#B33F62", "#4B9D3A", "#D99B1B", "#1B8A8F", "#7A4FA3"]
CHAIN_SOURCE_COLORS = ["#98AFCB", "#D1A0AE", "#A7C79B", "#DCC187", "#99C9C9", "#B6A2CF"]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for old_panel in PANEL_DIR.glob("*.png"):
        old_panel.unlink()


def render_converted(draw_fn, image_path: Path) -> tuple[np.ndarray, Path, list[str]]:
    return render_scene(draw_fn, image_path)


def render_source_view(draw_fn, image_path: Path) -> tuple[np.ndarray, Path]:
    return render_source(draw_fn, image_path)


def stack_points(chunks: list[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(chunk, dtype=float) for chunk in chunks if np.asarray(chunk).size]
    if not usable:
        return np.zeros((1, 3), dtype=float)
    return np.vstack(usable)


def add_single_trace_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    result,
    input_format: str,
    domain: str,
    source_kind: str,
) -> None:
    coords = np.asarray(result.coords, dtype=float)

    def draw_converted(plotter):
        pts = edge_points(result.graph)
        meshes = add_segmented_curve(plotter, pts, direct_closure=True, add_endpoints=True)
        return meshes, pts, result.issues

    def draw_source(plotter):
        if source_kind == "pdb":
            atom_points = load_pdb_atom_points(result.pdb_path, chain_id=result.chain_id, model_id=result.model_id)
        elif source_kind == "mmcif":
            atom_points = load_mmcif_atom_points(result.cif_path, chain_id=result.chain_id, model_id=result.model_id)
        else:
            raise ValueError(f"Unsupported source kind: {source_kind}")
        meshes = []
        meshes.extend(add_point_cloud(plotter, atom_points, color=SOURCE_GREY, opacity=0.30, point_size=5.5))
        meshes.extend(add_segmented_curve(plotter, coords, direct_closure=False, add_endpoints=False))
        return meshes, stack_points([atom_points, coords])

    converted, converted_path, issues = render_converted(draw_converted, PANEL_DIR / f"{stem}_converted.png")
    source, source_path = render_source_view(draw_source, PANEL_DIR / f"{stem}_source.png")
    panels.append({"title": title, "source_image": source, "converted_image": converted})
    summary.append(
        {
            "title": title,
            "domain": domain,
            "input_format": input_format,
            "source_view_path": str(source_path),
            "converted_view_path": str(converted_path),
            "success": source_path.exists() and converted_path.exists(),
            "issues": issues,
            "yamada_status": "pending downstream audit",
        }
    )


def add_multichain_pdb_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    pdb_id: str,
    chains: list[str],
    data_dir: Path,
    domain: str,
) -> None:
    results = [
        from_protein_ca_backbone(pdb_id, chain_id=chain, data_dir=data_dir, save_coords=True)
        for chain in chains
    ]

    def draw_converted(plotter):
        all_points = []
        meshes: list[pv.DataSet] = []
        for index, result in enumerate(results):
            pts = edge_points(result.graph)
            all_points.append(pts)
            _, _, _, span = point_span(stack_points([r.coords for r in results]))
            radius = 0.014 * span
            render_pts = np.vstack([pts, pts[0]])
            meshes.append(add_tube(plotter, render_pts, radius=radius, color=CHAIN_COLORS[index % len(CHAIN_COLORS)]))
            meshes.extend(add_endpoint_nodes(plotter, pts, radius=2.5 * radius))
        return meshes, stack_points(all_points), [issue for result in results for issue in result.issues]

    def draw_source(plotter):
        all_points = []
        meshes: list[pv.DataSet] = []
        for index, result in enumerate(results):
            atom_points = load_pdb_atom_points(result.pdb_path, chain_id=result.chain_id, model_id=result.model_id)
            all_points.extend([atom_points, result.coords])
            meshes.extend(
                add_point_cloud(
                    plotter,
                    atom_points,
                    color=CHAIN_SOURCE_COLORS[index % len(CHAIN_SOURCE_COLORS)],
                    opacity=0.28,
                    point_size=5.0,
                )
            )
            _, _, _, span = point_span(stack_points([r.coords for r in results]))
            meshes.append(
                add_tube(
                    plotter,
                    result.coords,
                    radius=0.008 * span,
                    color=CHAIN_COLORS[index % len(CHAIN_COLORS)],
                    opacity=0.92,
                    n_sides=18,
                )
            )
        return meshes, stack_points(all_points)

    converted, converted_path, issues = render_converted(draw_converted, PANEL_DIR / f"{stem}_converted.png")
    source, source_path = render_source_view(draw_source, PANEL_DIR / f"{stem}_source.png")
    panels.append({"title": title, "source_image": source, "converted_image": converted})
    summary.append(
        {
            "title": title,
            "domain": domain,
            "input_format": ".pdb",
            "source_view_path": str(source_path),
            "converted_view_path": str(converted_path),
            "success": source_path.exists() and converted_path.exists(),
            "issues": issues,
            "chains": chains,
            "yamada_status": "pending downstream audit",
        }
    )


def add_dna_duplex_panel(panels: list[dict], summary: list[dict]) -> None:
    dna_a = from_nucleic_acid_backbone("1BNA", chain_id="A", atom_name="P", data_dir=DATA_DIR / "dna", save_coords=True)
    dna_b = from_nucleic_acid_backbone("1BNA", chain_id="B", atom_name="P", data_dir=DATA_DIR / "dna", save_coords=True)
    results = [dna_a, dna_b]

    def draw_converted(plotter):
        all_points = stack_points([edge_points(result.graph) for result in results])
        _, _, _, span = point_span(all_points)
        radius = 0.015 * span
        meshes: list[pv.DataSet] = []
        for index, result in enumerate(results):
            pts = edge_points(result.graph)
            meshes.append(add_tube(plotter, np.vstack([pts, pts[0]]), radius=radius, color=[EDGE_COLOR, EDGE_COLOR_2][index]))
            meshes.extend(add_endpoint_nodes(plotter, pts, radius=2.5 * radius))
        return meshes, all_points, dna_a.issues + dna_b.issues

    def draw_source(plotter):
        atoms_a = load_pdb_atom_points(dna_a.pdb_path, chain_id=dna_a.chain_id, model_id=dna_a.model_id)
        atoms_b = load_pdb_atom_points(dna_b.pdb_path, chain_id=dna_b.chain_id, model_id=dna_b.model_id)
        meshes = []
        meshes.extend(add_point_cloud(plotter, atoms_a, color="#98AFCB", opacity=0.30, point_size=5.0))
        meshes.extend(add_point_cloud(plotter, atoms_b, color="#D1A0AE", opacity=0.30, point_size=5.0))
        meshes.extend(add_segmented_curve(plotter, dna_a.coords, colors=[EDGE_COLOR] * 5, direct_closure=False, add_endpoints=False))
        meshes.extend(add_segmented_curve(plotter, dna_b.coords, colors=[EDGE_COLOR_2] * 5, direct_closure=False, add_endpoints=False))
        return meshes, stack_points([atoms_a, atoms_b, dna_a.coords, dna_b.coords])

    converted, converted_path, issues = render_converted(draw_converted, PANEL_DIR / "e_b_dna_duplex_pdb_converted.png")
    source, source_path = render_source_view(draw_source, PANEL_DIR / "e_b_dna_duplex_pdb_source.png")
    panels.append({"title": "B-DNA Duplex PDB", "source_image": source, "converted_image": converted})
    summary.append(
        {
            "title": "B-DNA Duplex PDB",
            "domain": "DNA",
            "input_format": ".pdb",
            "source_view_path": str(source_path),
            "converted_view_path": str(converted_path),
            "success": source_path.exists() and converted_path.exists(),
            "issues": issues,
            "chains": ["A", "B"],
            "yamada_status": "pending downstream audit",
        }
    )


def assemble_figure(panels: list[dict]) -> tuple[Path, Path, Path]:
    labels = [f"({chr(ord('a') + i)})" for i in range(len(panels))]
    fig = plt.figure(figsize=(18.2, 8.7), facecolor="white")
    bboxes = compact_panel_bboxes(len(panels), rows=2, cols=4, gap_x=0.004, gap_y=0.006)
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

    add_single_trace_panel(
        panels,
        summary,
        title="Crambin PDB",
        stem="a_crambin_pdb",
        result=from_protein_ca_backbone("1CRN", chain_id="A", data_dir=EXAMPLES_DIR / "proteins" / "data", save_coords=True),
        input_format=".pdb",
        domain="protein",
        source_kind="pdb",
    )
    add_single_trace_panel(
        panels,
        summary,
        title="Ubiquitin PDB",
        stem="b_ubiquitin_pdb",
        result=from_protein_ca_backbone("1UBQ", chain_id="A", data_dir=DATA_DIR / "proteins", save_coords=True),
        input_format=".pdb",
        domain="protein",
        source_kind="pdb",
    )
    add_single_trace_panel(
        panels,
        summary,
        title="Protein Backbone PDB",
        stem="c_protein_backbone_pdb",
        result=from_protein_ca_backbone("1J85", chain_id="A", data_dir=DATA_DIR / "proteins", save_coords=True),
        input_format=".pdb",
        domain="protein",
        source_kind="pdb",
    )
    add_multichain_pdb_panel(
        panels,
        summary,
        title="Hemoglobin PDB",
        stem="d_hemoglobin_pdb",
        pdb_id="4HHB",
        chains=["A", "B", "C", "D"],
        data_dir=EXAMPLES_DIR / "proteins" / "data",
        domain="protein complex",
    )
    add_dna_duplex_panel(panels, summary)
    add_single_trace_panel(
        panels,
        summary,
        title="tRNA mmCIF",
        stem="f_trna_mmcif",
        result=from_mmcif_backbone("1EHZ", chain_id="A", atom_name="P", data_dir=DATA_DIR / "mmcif", save_coords=True),
        input_format=".cif",
        domain="RNA",
        source_kind="mmcif",
    )
    add_single_trace_panel(
        panels,
        summary,
        title="Ubiquitin mmCIF",
        stem="g_ubiquitin_mmcif",
        result=from_mmcif_backbone("1UBQ", chain_id="A", atom_name="CA", data_dir=DATA_DIR / "mmcif", save_coords=True),
        input_format=".cif",
        domain="protein",
        source_kind="mmcif",
    )

    png_path, svg_path, pdf_path = assemble_figure(panels)
    summary.extend(
        [
            {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
            {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
            {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
        ]
    )
    SUMMARY_PATH.write_text(json.dumps({"figure": "Appendix S1 biology inputs", "panels": summary}, indent=2) + "\n")

    print("Appendix S1 biology input figure")
    print(f"Panel directory: {PANEL_DIR}")
    print(f"Final PNG: {png_path}")
    print(f"Final SVG: {svg_path}")
    print(f"Final PDF: {pdf_path}")
    print(f"Summary path: {SUMMARY_PATH}")
    for item in summary:
        if "title" in item:
            print(f"Panel: {item['title']} success={item['success']} issues={item['issues'] or 'none'}")


if __name__ == "__main__":
    main()
