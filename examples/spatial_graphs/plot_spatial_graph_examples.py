"""Smoke test abstract spatial-graph JSON input for Task 2."""

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
    build_spatial_graph_from_json,
    write_spatial_graph_json,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
THETA_GRAPH_JSON = DATA_DIR / "theta_graph.json"


def make_arc(start, end, lift: float, side: float, n_points: int = 64) -> list[list[float]]:
    t = np.linspace(0.0, 1.0, n_points)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    base = (1.0 - t)[:, None] * start + t[:, None] * end
    base[:, 1] += side * np.sin(np.pi * t)
    base[:, 2] += lift * np.sin(np.pi * t)
    return base.tolist()


def make_theta_graph_payload() -> dict:
    left = [-1.5, 0.0, 0.0]
    right = [1.5, 0.0, 0.0]
    return {
        "graph_id": "theta_graph_json",
        "metadata": {
            "description": "two vertices connected by three embedded arcs",
            "example_kind": "abstract_spatial_graph",
        },
        "nodes": {
            "u": left,
            "v": right,
        },
        "edges": [
            {
                "id": "upper_arc",
                "source": "u",
                "target": "v",
                "points": make_arc(left, right, lift=0.9, side=0.75),
            },
            {
                "id": "middle_arc",
                "source": "u",
                "target": "v",
                "points": make_arc(left, right, lift=-0.25, side=-0.15),
            },
            {
                "id": "lower_arc",
                "source": "u",
                "target": "v",
                "points": make_arc(left, right, lift=0.45, side=-0.85),
            },
        ],
    }


def write_example_inputs() -> None:
    write_spatial_graph_json(make_theta_graph_payload(), THETA_GRAPH_JSON)


def set_axes_equal(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_matplotlib_graph(result: SpatialGraphResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / f"{result.graph_id}.png"

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    all_edge_points = []

    for _, _, key, data in result.graph.edges(keys=True, data=True):
        pts = np.asarray(data["pts"], dtype=float)
        all_edge_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0, label=key)

    node_pts = np.asarray([data["pos"] for _, data in result.graph.nodes(data=True)])
    ax.scatter(node_pts[:, 0], node_pts[:, 1], node_pts[:, 2], color="tab:red", s=55)
    ax.set_title(f"{result.graph_id} abstract spatial graph")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    set_axes_equal(ax, np.vstack(all_edge_points))

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
    figure_path: Path,
    html_path: Path,
) -> None:
    print(f"Graph ID: {result.graph_id}")
    print(f"Source path: {result.source_path}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Input kind: {result.graph.graph.get('input_kind')}")
    print(f"PNG path: {figure_path}")
    print(f"PNG created successfully: {figure_path.exists() and figure_path.stat().st_size > 0}")
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
    result = build_spatial_graph_from_json(THETA_GRAPH_JSON)
    figure_path = plot_matplotlib_graph(result)
    html_path = write_plotly_graph(result)

    print("Abstract spatial-graph input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")
    print_result_summary(result, figure_path, html_path)


if __name__ == "__main__":
    main()
