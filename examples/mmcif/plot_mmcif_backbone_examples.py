"""Smoke test RCSB mmCIF backbone inputs for Task 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg

from knotted_graph.inputs import MMCIFBackboneInputResult, from_mmcif_backbone
from knotted_graph.inputs.pdb import format_chain_counts


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"


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


def plot_matplotlib_curve(result: MMCIFBackboneInputResult, output_stem: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / f"{output_stem}.png"
    pts = result.coords

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0)
    ax.scatter(*pts[0], color="tab:green", s=45, label="start")
    ax.scatter(*pts[-1], color="tab:red", s=45, label="end")
    ax.set_title(f"{result.pdb_id} chain {result.chain_id} {result.atom_name} trace from mmCIF")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    set_axes_equal(ax, pts)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_plotly_graph(result: MMCIFBackboneInputResult, output_stem: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    html_path = FIGURE_DIR / f"{output_stem}_graph.html"
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(title=f"{result.pdb_id} {result.atom_name} mmCIF trace as MultiGraph(pos/pts)")
    fig.write_html(str(html_path))
    return html_path


def print_result_summary(
    result: MMCIFBackboneInputResult,
    png_path: Path,
    html_path: Path,
) -> None:
    print(f"PDB ID: {result.pdb_id}")
    print(f"Source URL: {result.source_url}")
    print(f"Downloaded now: {result.downloaded}")
    print(f"mmCIF path: {result.cif_path}")
    print(f"Available chains for {result.atom_name}: {format_chain_counts(result.available_chains)}")
    print(f"Selected chain: {result.chain_id}")
    print(f"Selected model: {result.model_id}")
    print(f"Atom name: {result.atom_name}")
    print(f"Extracted atom count: {result.coords.shape[0]}")
    print(f"Coords npy path: {result.coords_npy_path}")
    print(f"Coords saved successfully: {result.coords_saved}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Graph input_kind: {result.graph.graph.get('input_kind')}")
    print(f"PNG path: {png_path}")
    print(f"PNG created successfully: {png_path.exists() and png_path.stat().st_size > 0}")
    print(f"Graph HTML path: {html_path}")
    print(f"Graph HTML created successfully: {html_path.exists() and html_path.stat().st_size > 0}")
    if result.issues:
        print("Issues:")
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("Issues: none")
    print("")


def main() -> None:
    examples = [
        {
            "pdb_id": "1CRN",
            "chain_id": "A",
            "atom_name": "CA",
            "output_stem": "1crn_chainA_ca_mmcif",
            "description": "protein C-alpha trace",
        },
        {
            "pdb_id": "1EHZ",
            "chain_id": "A",
            "atom_name": "P",
            "output_stem": "1ehz_chainA_rna_phosphate_mmcif",
            "description": "RNA phosphate trace",
        },
    ]

    print("mmCIF backbone input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")

    for example in examples:
        print(f"Example: {example['description']}")
        result = from_mmcif_backbone(
            example["pdb_id"],
            chain_id=example["chain_id"],
            atom_name=example["atom_name"],
            data_dir=DATA_DIR,
            save_coords=True,
        )
        png_path = plot_matplotlib_curve(result, example["output_stem"])
        html_path = write_plotly_graph(result, example["output_stem"])
        print_result_summary(result, png_path, html_path)


if __name__ == "__main__":
    main()
