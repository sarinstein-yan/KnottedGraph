"""Smoke test electric circuit input as an embedded spatial network."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg


HERE = Path(__file__).resolve().parent
REPO_EXAMPLES = HERE.parent
sys.path.insert(0, str(REPO_EXAMPLES / "spatial_graphs"))

from spatial_graph_adapter import build_spatial_graph_from_json, write_spatial_graph_json


DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
CIRCUIT_JSON = DATA_DIR / "rc_filter_spatial_circuit.json"
PNG_PATH = FIGURE_DIR / "rc_filter_spatial_circuit.png"
HTML_PATH = FIGURE_DIR / "rc_filter_spatial_circuit_graph.html"


def wire(points):
    return [list(point) for point in points]


def make_circuit_payload() -> dict:
    nodes = {
        "vin": [-2.0, 0.0, 0.0],
        "r1_left": [-1.25, 0.0, 0.0],
        "r1_right": [0.0, 0.0, 0.0],
        "vout": [1.1, 0.0, 0.0],
        "ground": [1.1, -1.5, 0.0],
        "return": [-2.0, -1.5, 0.0],
        "jumper": [0.0, 0.75, 0.65],
    }
    return {
        "graph_id": "rc_filter_spatial_circuit",
        "metadata": {
            "description": "simple embedded RC-like circuit with a raised jumper wire",
            "example_kind": "electric_circuit_spatial_network",
        },
        "nodes": nodes,
        "edges": [
            {
                "id": "input_wire",
                "component": "wire",
                "source": "vin",
                "target": "r1_left",
            },
            {
                "id": "resistor_R1",
                "component": "resistor",
                "source": "r1_left",
                "target": "r1_right",
                "points": wire(
                    [
                        nodes["r1_left"],
                        [-1.05, 0.18, 0.0],
                        [-0.85, -0.18, 0.0],
                        [-0.65, 0.18, 0.0],
                        [-0.45, -0.18, 0.0],
                        [-0.25, 0.18, 0.0],
                        nodes["r1_right"],
                    ]
                ),
            },
            {
                "id": "output_wire",
                "component": "wire",
                "source": "r1_right",
                "target": "vout",
            },
            {
                "id": "capacitor_C1",
                "component": "capacitor",
                "source": "vout",
                "target": "ground",
                "points": wire(
                    [
                        nodes["vout"],
                        [1.1, -0.45, 0.0],
                        [0.85, -0.55, 0.0],
                        [1.35, -0.55, 0.0],
                        [1.35, -0.75, 0.0],
                        [0.85, -0.75, 0.0],
                        [1.1, -0.9, 0.0],
                        nodes["ground"],
                    ]
                ),
            },
            {
                "id": "return_wire",
                "component": "wire",
                "source": "ground",
                "target": "return",
                "points": wire([nodes["ground"], [0.0, -1.5, 0.0], nodes["return"]]),
            },
            {
                "id": "source_return",
                "component": "wire",
                "source": "return",
                "target": "vin",
            },
            {
                "id": "raised_jumper",
                "component": "jumper",
                "source": "r1_right",
                "target": "jumper",
                "points": wire(
                    [
                        nodes["r1_right"],
                        [0.0, 0.25, 0.3],
                        [0.0, 0.55, 0.6],
                        nodes["jumper"],
                    ]
                ),
            },
            {
                "id": "jumper_to_vout",
                "component": "wire",
                "source": "jumper",
                "target": "vout",
                "points": wire(
                    [
                        nodes["jumper"],
                        [0.45, 0.65, 0.55],
                        [0.85, 0.35, 0.25],
                        nodes["vout"],
                    ]
                ),
            },
        ],
    }


def write_example_input() -> None:
    write_spatial_graph_json(make_circuit_payload(), CIRCUIT_JSON)


def set_axes_equal(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_circuit(result) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    for _, _, key, data in result.graph.edges(keys=True, data=True):
        pts = np.asarray(data["pts"], dtype=float)
        all_pts.append(pts)
        component = data.get("component", "edge")
        lw = 2.8 if component in {"resistor", "capacitor"} else 1.8
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=lw, label=key)
    node_pts = np.asarray([data["pos"] for _, data in result.graph.nodes(data=True)])
    ax.scatter(node_pts[:, 0], node_pts[:, 1], node_pts[:, 2], color="tab:red", s=45)
    ax.set_title("RC filter as embedded spatial network")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=7)
    set_axes_equal(ax, np.vstack(all_pts))
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)


def write_plotly_graph(result) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(title="RC filter as MultiGraph(pos/pts)")
    fig.write_html(str(HTML_PATH))


def main() -> None:
    write_example_input()
    result = build_spatial_graph_from_json(CIRCUIT_JSON)
    plot_circuit(result)
    write_plotly_graph(result)

    print("Electric circuit spatial-network smoke test")
    print(f"Graph ID: {result.graph_id}")
    print(f"Source path: {result.source_path}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Input kind: {result.graph.graph.get('input_kind')}")
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


if __name__ == "__main__":
    main()
