"""Appendix spatial-graph input figure for Task 2.

This grouped appendix figure shows several node/edge CSV examples for
engineering and abstract spatial-network settings.  Each panel displays a
source-domain schematic above the converted ``MultiGraph(pos/pts)`` rendering.
"""

from __future__ import annotations

import csv
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
PANEL_DIR = FIGURE_DIR / "appendix_spatial_graph_panels"
SUMMARY_PATH = DATA_DIR / "appendix_spatial_graph_inputs_summary.json"
FIGURE_STEM = "appendix_spatial_graph_inputs"

sys.path.insert(0, str(HERE))

from knotted_graph.inputs import from_spatial_graph_csv  # noqa: E402
from plot_main_text_input_figure import (  # noqa: E402
    make_engineering_network_payload,
    render_scene,
    render_source,
)
from compact_appendix_layout import compact_panel_bboxes, draw_compact_panel  # noqa: E402
from plot_publication_style_gallery import (  # noqa: E402
    add_raw_spatial_graph,
    add_spatial_graph,
    graph_points,
)
from plot_publication_style_gallery_sets import (  # noqa: E402
    make_basic_network,
    make_hopf_payload,
    make_three_ring_payload,
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


def node_items(payload: dict) -> list[dict]:
    nodes = payload["nodes"]
    if isinstance(nodes, dict):
        return [
            {
                "node_id": str(node_id),
                "pos": pos,
                "label": str(node_id).replace("_", " ").title(),
                "type": "node",
            }
            for node_id, pos in nodes.items()
        ]
    normalized = []
    for node in nodes:
        node_id = node.get("node_id", node.get("id"))
        pos = node.get("pos")
        if node_id is None or pos is None:
            raise ValueError("node payload entries require node_id/id and pos")
        normalized.append(
            {
                "node_id": str(node_id),
                "pos": pos,
                "label": node.get("label", str(node_id).replace("_", " ").title()),
                "type": node.get("type", "node"),
            }
        )
    return normalized


def edge_items(payload: dict) -> list[dict]:
    normalized = []
    seen: dict[str, int] = {}
    for index, edge in enumerate(payload["edges"]):
        raw_edge_id = str(edge.get("edge_id", edge.get("id", edge.get("key", f"edge_{index}"))))
        seen[raw_edge_id] = seen.get(raw_edge_id, 0) + 1
        edge_id = raw_edge_id if seen[raw_edge_id] == 1 else f"{raw_edge_id}_{seen[raw_edge_id]}"
        points = edge.get("points_json", edge.get("points", edge.get("pts")))
        if points is None:
            raise ValueError(f"edge {edge_id!r} is missing curved points")
        normalized.append(
            {
                "edge_id": str(edge_id),
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "label": edge.get("label", str(edge_id).replace("_", " ").title()),
                "type": edge.get("type", str(edge_id)),
                "points_json": np.asarray(points, dtype=float).tolist(),
            }
        )
    return normalized


def normalize_payload(payload: dict, *, graph_id: str) -> dict:
    return {
        "graph_id": graph_id,
        "nodes": node_items(payload),
        "edges": edge_items(payload),
    }


def write_csv_payload(payload: dict, nodes_path: Path, edges_path: Path) -> None:
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "x", "y", "z", "label", "type"])
        writer.writeheader()
        for node in payload["nodes"]:
            x, y, z = np.asarray(node["pos"], dtype=float)
            writer.writerow(
                {
                    "node_id": node["node_id"],
                    "x": f"{x:.8f}",
                    "y": f"{y:.8f}",
                    "z": f"{z:.8f}",
                    "label": node["label"],
                    "type": node["type"],
                }
            )
    with edges_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["edge_id", "source", "target", "label", "type", "points_json"])
        writer.writeheader()
        for edge in payload["edges"]:
            writer.writerow(
                {
                    "edge_id": edge["edge_id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": edge["label"],
                    "type": edge["type"],
                    "points_json": json.dumps(edge["points_json"]),
                }
            )


def curved_edge_count(graph) -> int:
    count = 0
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)
        if pts.shape[0] > 2:
            count += 1
    return count


def add_spatial_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    payload: dict,
    domain: str,
) -> None:
    set_dir = DATA_DIR / "appendix_spatial_graphs"
    normalized = normalize_payload(payload, graph_id=stem)
    nodes_path = set_dir / f"{stem}_nodes.csv"
    edges_path = set_dir / f"{stem}_edges.csv"
    write_csv_payload(normalized, nodes_path, edges_path)
    result = from_spatial_graph_csv(nodes_path, edges_path, graph_id=stem)

    def draw_converted(plotter):
        meshes, points = add_spatial_graph(plotter, result.graph)
        return meshes, points, result.issues

    def draw_source(plotter):
        return add_raw_spatial_graph(plotter, result.graph)

    converted, converted_path, issues = render_scene(draw_converted, PANEL_DIR / f"{stem}_converted.png")
    source, source_path = render_source(draw_source, PANEL_DIR / f"{stem}_source.png")
    panels.append({"title": title, "source_image": source, "converted_image": converted})
    summary.append(
        {
            "title": title,
            "domain": domain,
            "input_format": "node/edge CSV",
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "source_view_path": str(source_path),
            "converted_view_path": str(converted_path),
            "node_count": result.graph.number_of_nodes(),
            "edge_count": result.graph.number_of_edges(),
            "curved_edge_count": curved_edge_count(result.graph),
            "success": source_path.exists() and converted_path.exists(),
            "issues": issues,
            "yamada_status": "pending downstream audit",
        }
    )


def assemble_figure(panels: list[dict]) -> tuple[Path, Path, Path]:
    labels = [f"({chr(ord('a') + i)})" for i in range(len(panels))]
    fig = plt.figure(figsize=(18.2, 8.9), facecolor="white")
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
    examples = [
        ("Engineering Network CSV", "engineering_network_csv", make_engineering_network_payload(), "engineering component network"),
        ("Pipe Manifold CSV", "pipe_manifold_csv", make_basic_network("pipe_manifold_csv", "pipe"), "pipe network"),
        ("Circuit Harness CSV", "circuit_harness_csv", make_basic_network("circuit_harness_csv", "circuit"), "electric circuit / cable harness"),
        ("Cooling Network CSV", "cooling_network_csv", make_basic_network("cooling_network_csv", "cooling"), "cooling system"),
        ("Vascular Branch CSV", "vascular_branch_csv", make_basic_network("vascular_branch_csv", "vascular"), "branching transport network"),
        ("Lattice Truss CSV", "lattice_truss_csv", make_basic_network("lattice_truss_csv", "truss"), "mechanical truss"),
        ("Hopf Link CSV", "hopf_link_csv", make_hopf_payload(), "abstract spatial graph"),
        ("Three-Ring Link CSV", "three_ring_link_csv", make_three_ring_payload(), "abstract spatial graph"),
    ]
    for title, stem, payload, domain in examples:
        add_spatial_panel(panels, summary, title=title, stem=stem, payload=payload, domain=domain)

    png_path, svg_path, pdf_path = assemble_figure(panels)
    summary.extend(
        [
            {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
            {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
            {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
        ]
    )
    SUMMARY_PATH.write_text(json.dumps({"figure": "Appendix S3 spatial graph inputs", "panels": summary}, indent=2) + "\n")

    print("Appendix S3 spatial graph input figure")
    print(f"Panel directory: {PANEL_DIR}")
    print(f"Final PNG: {png_path}")
    print(f"Final SVG: {svg_path}")
    print(f"Final PDF: {pdf_path}")
    print(f"Summary path: {SUMMARY_PATH}")
    for item in summary:
        if "title" in item:
            print(
                f"Panel: {item['title']} success={item['success']} "
                f"nodes={item['node_count']} edges={item['edge_count']} "
                f"curved_edges={item['curved_edge_count']} issues={item['issues'] or 'none'}"
            )


if __name__ == "__main__":
    main()
