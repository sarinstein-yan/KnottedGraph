"""Download 1CRN and plot its C-alpha backbone as a 3D curve.

This is a protein-input smoke test using the public ``knotted_graph.inputs``
API.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg

from knotted_graph.inputs import PDBBackboneInputResult, from_protein_ca_backbone
from knotted_graph.inputs.pdb import format_chain_counts


PDB_ID = "1CRN"
SELECTED_CHAIN_ID = "A"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
FIGURE_PATH = FIGURE_DIR / f"{PDB_ID.lower()}_backbone.png"
GRAPH_HTML_PATH = FIGURE_DIR / f"{PDB_ID.lower()}_backbone_graph.html"


def set_axes_equal(ax) -> None:
    """Make a 3D matplotlib axis use equal data scale on x/y/z."""
    x_limits = np.asarray(ax.get_xlim3d(), dtype=float)
    y_limits = np.asarray(ax.get_ylim3d(), dtype=float)
    z_limits = np.asarray(ax.get_zlim3d(), dtype=float)

    centers = np.array([x_limits.mean(), y_limits.mean(), z_limits.mean()])
    radius = 0.5 * max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits))

    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_backbone(coords: np.ndarray, figure_path: Path = FIGURE_PATH) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords with shape (N, 3), got {coords.shape}.")
    if coords.shape[0] < 2:
        raise ValueError("Need at least two C-alpha atoms to plot a backbone curve.")

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "-o", lw=1.8, ms=3.0)
    ax.scatter(*coords[0], color="tab:green", s=45, label="first CA")
    ax.scatter(*coords[-1], color="tab:red", s=45, label="last CA")

    ax.set_title("1CRN crambin C-alpha backbone")
    ax.set_xlabel("x (Angstrom)")
    ax.set_ylabel("y (Angstrom)")
    ax.set_zlabel("z (Angstrom)")
    ax.legend(loc="upper left")
    set_axes_equal(ax)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def write_plotly_graph(result: PDBBackboneInputResult, html_path: Path = GRAPH_HTML_PATH) -> None:
    """Use the existing knotted_graph Plotly graph visualizer."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(
        title=f"{result.pdb_id} C-alpha backbone as MultiGraph(pos/pts)"
    )
    fig.write_html(str(html_path))


def print_summary(result: PDBBackboneInputResult) -> None:
    chain_counts = Counter(record["chain_id"] for record in result.records)
    chains = ", ".join(f"{chain}:{count}" for chain, count in sorted(chain_counts.items()))
    edge_data = next(iter(result.graph.edges(data=True)))[2]

    print(f"PDB ID: {result.pdb_id}")
    print(f"Source URL: {result.source_url}")
    print(f"PDB path: {result.pdb_path}")
    print(f"Downloaded new file: {result.downloaded}")
    print(
        "File downloaded successfully: "
        f"{result.pdb_path.exists() and result.pdb_path.stat().st_size > 0}"
    )
    print(f"Available C-alpha chains: {format_chain_counts(result.available_chains)}")
    print(f"Selected chain: {result.chain_id}")
    print(f"Selected model: {result.model_id}")
    print(f"C-alpha atoms extracted: {result.coords.shape[0]}")
    print(f"Chains among C-alpha atoms: {chains}")
    print(f"Coordinates NPY path: {result.coords_npy_path}")
    print(f"Coordinates saved successfully: {result.coords_saved}")
    print(f"Saved coordinates shape: {result.saved_coords_shape}")
    print(f"First coordinate: {result.coords[0].tolist()}")
    print(f"Last coordinate: {result.coords[-1].tolist()}")
    print(f"Figure path: {FIGURE_PATH}")
    print(f"Plot created successfully: {FIGURE_PATH.exists() and FIGURE_PATH.stat().st_size > 0}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Graph model_id: {result.graph.graph.get('model_id')}")
    print(
        "Graph node 'pos' attributes present: "
        f"{all('pos' in data for _, data in result.graph.nodes(data=True))}"
    )
    print(
        "Graph edge 'pts' attributes present: "
        f"{all('pts' in data for _, _, data in result.graph.edges(data=True))}"
    )
    print(f"Backbone edge pts shape: {np.asarray(edge_data['pts']).shape}")
    print(f"Graph HTML path: {GRAPH_HTML_PATH}")
    print(
        "Graph HTML created successfully: "
        f"{GRAPH_HTML_PATH.exists() and GRAPH_HTML_PATH.stat().st_size > 0}"
    )
    if result.issues:
        print("Issues:")
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("Issues: none")


def main() -> None:
    result = from_protein_ca_backbone(
        PDB_ID,
        chain_id=SELECTED_CHAIN_ID,
        data_dir=DATA_DIR,
        save_coords=True,
    )
    plot_backbone(result.coords)
    write_plotly_graph(result)
    print_summary(result)


if __name__ == "__main__":
    main()
