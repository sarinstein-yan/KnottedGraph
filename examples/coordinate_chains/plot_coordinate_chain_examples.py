"""Smoke test generic coordinate-chain inputs for Task 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg

from coordinate_curve_adapter import (
    DATA_DIR,
    CoordinateCurveResult,
    build_curve_from_csv,
    build_curve_from_json,
    build_curve_from_npy,
    build_curve_from_table,
    build_curve_from_xyz,
    save_npy_coords,
    write_csv_coords,
    write_json_coords,
    write_table_coords,
    write_xyz_coords,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"

OPEN_HELIX_CSV = DATA_DIR / "open_helix_polymer.csv"
OPEN_HELIX_JSON = DATA_DIR / "open_helix_polymer.json"
OPEN_HELIX_TSV = DATA_DIR / "open_helix_polymer.tsv"
OPEN_HELIX_DAT = DATA_DIR / "open_helix_polymer.dat"
CLOSED_TREFOIL_XYZ = DATA_DIR / "closed_trefoil_ring.xyz"
CLOSED_TREFOIL_NPY = DATA_DIR / "closed_trefoil_ring.npy"


def make_open_helix(n_points: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 6.0 * np.pi, n_points)
    return np.column_stack([np.cos(t), np.sin(t), 0.12 * t])


def make_closed_trefoil(n_points: int = 160) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.column_stack(
        [
            np.sin(t) + 2.0 * np.sin(2.0 * t),
            np.cos(t) - 2.0 * np.cos(2.0 * t),
            -np.sin(3.0 * t),
        ]
    )


def write_example_inputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    open_helix = make_open_helix()
    write_csv_coords(open_helix, OPEN_HELIX_CSV)
    write_json_coords(
        open_helix,
        OPEN_HELIX_JSON,
        closed=False,
        curve_id="open_helix_polymer",
    )
    write_table_coords(open_helix, OPEN_HELIX_TSV)
    write_table_coords(open_helix, OPEN_HELIX_DAT, delimiter=" ", header="x y z")
    trefoil = make_closed_trefoil()
    write_xyz_coords(trefoil, CLOSED_TREFOIL_XYZ, comment="closed trefoil ring")
    save_npy_coords(trefoil, CLOSED_TREFOIL_NPY)


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


def edge_points(result: CoordinateCurveResult) -> np.ndarray:
    edge_data = next(iter(result.graph.edges(data=True)))[2]
    return np.asarray(edge_data["pts"], dtype=float)


def plot_matplotlib_curve(result: CoordinateCurveResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / f"{result.curve_id}.png"
    pts = edge_points(result)

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0)
    ax.scatter(*pts[0], color="tab:green", s=45, label="start")
    if result.closed:
        ax.set_title(f"{result.curve_id} closed coordinate curve")
    else:
        ax.scatter(*pts[-1], color="tab:red", s=45, label="end")
        ax.set_title(f"{result.curve_id} open coordinate curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    set_axes_equal(ax)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_plotly_graph(result: CoordinateCurveResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    html_path = FIGURE_DIR / f"{result.curve_id}_graph.html"
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(title=f"{result.curve_id} as MultiGraph(pos/pts)")
    fig.write_html(str(html_path))
    return html_path


def print_result_summary(
    result: CoordinateCurveResult,
    figure_path: Path,
    html_path: Path,
) -> None:
    pts = edge_points(result)
    print(f"Curve ID: {result.curve_id}")
    print(f"Source format: {result.source_format}")
    print(f"Source path: {result.source_path}")
    print(f"Closed curve: {result.closed}")
    print(f"Input coordinate shape: {result.coords.shape}")
    print(f"Graph edge pts shape: {pts.shape}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Graph is_closed: {result.graph.graph.get('is_closed')}")
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
    print("")


def main() -> None:
    write_example_inputs()

    results = [
        build_curve_from_csv(
            OPEN_HELIX_CSV,
            closed=False,
            curve_id="open_helix_polymer_csv",
        ),
        build_curve_from_json(
            OPEN_HELIX_JSON,
            closed=False,
            curve_id="open_helix_polymer_json",
        ),
        build_curve_from_table(
            OPEN_HELIX_TSV,
            closed=False,
            curve_id="open_helix_polymer_tsv",
            source_format="tsv",
            delimiter="\t",
        ),
        build_curve_from_table(
            OPEN_HELIX_DAT,
            closed=False,
            curve_id="open_helix_polymer_dat",
            source_format="dat",
        ),
        build_curve_from_xyz(
            CLOSED_TREFOIL_XYZ,
            closed=True,
            curve_id="closed_trefoil_ring_xyz",
        ),
        build_curve_from_npy(
            CLOSED_TREFOIL_NPY,
            closed=True,
            curve_id="closed_trefoil_ring_npy",
        ),
    ]

    print("Coordinate-chain input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")

    for result in results:
        figure_path = plot_matplotlib_curve(result)
        html_path = write_plotly_graph(result)
        print_result_summary(result, figure_path, html_path)


if __name__ == "__main__":
    main()
