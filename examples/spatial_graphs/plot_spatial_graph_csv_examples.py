"""Smoke test node/edge CSV input for abstract spatial graphs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg

from spatial_graph_adapter import (
    DATA_DIR,
    SpatialGraphResult,
    build_spatial_graph_from_csv,
    write_spatial_graph_csv,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
NODES_CSV = DATA_DIR / "vascular_bifurcation_nodes.csv"
EDGES_CSV = DATA_DIR / "vascular_bifurcation_edges.csv"


def curved_segment(start, end, bend, n_points: int = 36) -> list[list[float]]:
    t = np.linspace(0.0, 1.0, n_points)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    base = (1.0 - t)[:, None] * start + t[:, None] * end
    base += np.sin(np.pi * t)[:, None] * np.asarray(bend, dtype=float)
    return base.tolist()


def make_vascular_bifurcation_payload() -> dict:
    inlet = [0.0, -1.6, 0.0]
    junction = [0.0, 0.0, 0.35]
    left = [-1.2, 1.25, 0.0]
    right = [1.25, 1.15, 0.15]
    bridge = [0.0, 1.55, 0.95]
    return {
        "graph_id": "vascular_bifurcation_csv",
        "metadata": {
            "description": "node/edge CSV spatial network with curved embedded edges",
            "example_kind": "csv_spatial_graph",
        },
        "nodes": {
            "inlet": inlet,
            "junction": junction,
            "left_outlet": left,
            "right_outlet": right,
            "bridge": bridge,
        },
        "edges": [
            {
                "id": "inlet_to_junction",
                "source": "inlet",
                "target": "junction",
                "points": curved_segment(inlet, junction, [0.15, -0.1, 0.35]),
            },
            {
                "id": "junction_to_left",
                "source": "junction",
                "target": "left_outlet",
                "points": curved_segment(junction, left, [-0.25, 0.15, 0.2]),
            },
            {
                "id": "junction_to_right",
                "source": "junction",
                "target": "right_outlet",
                "points": curved_segment(junction, right, [0.2, 0.15, -0.1]),
            },
            {
                "id": "left_to_bridge",
                "source": "left_outlet",
                "target": "bridge",
                "points": curved_segment(left, bridge, [0.0, 0.15, 0.35]),
            },
            {
                "id": "bridge_to_right",
                "source": "bridge",
                "target": "right_outlet",
                "points": curved_segment(bridge, right, [0.0, -0.15, 0.2]),
            },
        ],
    }


def write_example_inputs() -> None:
    write_spatial_graph_csv(make_vascular_bifurcation_payload(), NODES_CSV, EDGES_CSV)


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


def plot_matplotlib_graph(result: SpatialGraphResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / f"{result.graph_id}.png"
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    edge_points = []
    for _, _, key, data in result.graph.edges(keys=True, data=True):
        pts = np.asarray(data["pts"], dtype=float)
        edge_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0, label=key)
    node_pts = np.asarray([data["pos"] for _, data in result.graph.nodes(data=True)])
    ax.scatter(node_pts[:, 0], node_pts[:, 1], node_pts[:, 2], color="tab:red", s=55)
    ax.set_title(f"{result.graph_id} node/edge CSV spatial graph")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=7)
    set_axes_equal(ax, np.vstack(edge_points))
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_plotly_graph(result: SpatialGraphResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    html_path = FIGURE_DIR / f"{result.graph_id}_graph.html"
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(title=f"{result.graph_id} as MultiGraph(pos/pts)")
    fig.write_html(str(html_path))
    return html_path


def print_result_summary(
    result: SpatialGraphResult,
    png_path: Path,
    html_path: Path,
) -> None:
    print(f"Graph ID: {result.graph_id}")
    print(f"Node CSV: {NODES_CSV}")
    print(f"Edge CSV: {EDGES_CSV}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Input kind: {result.graph.graph.get('input_kind')}")
    print(f"Source format: {result.graph.graph.get('source_format')}")
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


def main() -> None:
    write_example_inputs()
    result = build_spatial_graph_from_csv(
        NODES_CSV,
        EDGES_CSV,
        graph_id="vascular_bifurcation_csv",
    )
    png_path = plot_matplotlib_graph(result)
    html_path = write_plotly_graph(result)

    print("Node/edge CSV spatial graph input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")
    print_result_summary(result, png_path, html_path)


if __name__ == "__main__":
    main()
