"""Smoke test DNA PDB input using a phosphate backbone trace."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg

from knotted_graph.inputs import PDBBackboneInputResult, from_nucleic_acid_backbone
from knotted_graph.inputs.pdb import format_chain_counts


PDB_ID = "1BNA"
CHAIN_ID = "A"
MODEL_ID = 1
ATOM_NAME = "P"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PNG_PATH = FIGURE_DIR / "1bna_chainA_phosphate_backbone.png"
HTML_PATH = FIGURE_DIR / "1bna_chainA_phosphate_backbone_graph.html"


def set_axes_equal(ax) -> None:
    x_limits = np.asarray(ax.get_xlim3d(), dtype=float)
    y_limits = np.asarray(ax.get_ylim3d(), dtype=float)
    z_limits = np.asarray(ax.get_zlim3d(), dtype=float)
    centers = np.array([x_limits.mean(), y_limits.mean(), z_limits.mean()])
    radius = 0.5 * max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits))
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_backbone(result: PDBBackboneInputResult) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    coords = result.coords
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "-o", lw=2.0, ms=4.0)
    ax.scatter(*coords[0], color="tab:green", s=55, label="start")
    ax.scatter(*coords[-1], color="tab:red", s=55, label="end")
    ax.set_title(f"{result.pdb_id} chain {result.chain_id} DNA {result.atom_name} trace")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)


def write_plotly_graph(result: PDBBackboneInputResult) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(title=f"{result.pdb_id} chain {result.chain_id} DNA {result.atom_name} trace")
    fig.write_html(str(HTML_PATH))


def print_summary(result: PDBBackboneInputResult) -> None:
    chain_counts = Counter(record["chain_id"] for record in result.records)
    chains = ", ".join(f"{chain}:{count}" for chain, count in sorted(chain_counts.items()))
    edge_data = next(iter(result.graph.edges(data=True)))[2]
    print(f"PDB ID: {result.pdb_id}")
    print(f"Source URL: {result.source_url}")
    print(f"PDB path: {result.pdb_path}")
    print(f"Downloaded new file: {result.downloaded}")
    print(f"Available {result.atom_name} chains: {format_chain_counts(result.available_chains)}")
    print(f"Selected chain: {result.chain_id}")
    print(f"Selected model: {result.model_id}")
    print(f"Backbone atom: {result.atom_name}")
    print(f"Atoms extracted: {result.coords.shape[0]}")
    print(f"Chains among atoms: {chains}")
    print(f"Coordinates NPY path: {result.coords_npy_path}")
    print(f"Coordinates saved successfully: {result.coords_saved}")
    print(f"Saved coordinates shape: {result.saved_coords_shape}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Graph edge pts shape: {np.asarray(edge_data['pts']).shape}")
    print(f"PNG path: {PNG_PATH}")
    print(f"PNG created successfully: {PNG_PATH.exists() and PNG_PATH.stat().st_size > 0}")
    print(f"Graph HTML path: {HTML_PATH}")
    print(f"Graph HTML created successfully: {HTML_PATH.exists() and HTML_PATH.stat().st_size > 0}")
    if result.issues:
        print("Issues:")
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("Issues: none")


def main() -> None:
    result = from_nucleic_acid_backbone(
        PDB_ID,
        chain_id=CHAIN_ID,
        atom_name=ATOM_NAME,
        model_id=MODEL_ID,
        data_dir=DATA_DIR,
        save_coords=True,
    )
    plot_backbone(result)
    write_plotly_graph(result)
    print_summary(result)


if __name__ == "__main__":
    main()
