"""Smoke test polymer simulation snapshot inputs for Task 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import knotted_graph as kg

from knotted_graph.inputs import (
    PolymerInputResult,
    from_gromacs_gro,
    from_lammps_dump,
    write_gro_coords,
    write_lammps_dump,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

LAMMPS_DUMP = DATA_DIR / "linear_polymer_lammps.dump"
GRO_SNAPSHOT = DATA_DIR / "ring_polymer.gro"


def make_linear_polymer(n_points: int = 120) -> np.ndarray:
    t = np.linspace(0.0, 8.0 * np.pi, n_points)
    slow = np.linspace(-2.0, 2.0, n_points)
    return np.column_stack(
        [
            0.8 * np.cos(t) + 0.15 * slow,
            0.8 * np.sin(t),
            slow,
        ]
    )


def make_ring_polymer(n_points: int = 144) -> np.ndarray:
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
    write_lammps_dump(make_linear_polymer(), LAMMPS_DUMP, molecule_id=7)
    write_gro_coords(make_ring_polymer(), GRO_SNAPSHOT, residue_name="RNG", atom_name="BB")


def edge_points(result: PolymerInputResult) -> np.ndarray:
    edge_data = next(iter(result.graph.edges(data=True)))[2]
    return np.asarray(edge_data["pts"], dtype=float)


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


def plot_matplotlib_curve(result: PolymerInputResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / f"{result.polymer_id}.png"
    pts = edge_points(result)

    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0)
    ax.scatter(*pts[0], color="tab:green", s=45, label="start")
    if result.closed:
        ax.set_title(f"{result.polymer_id} closed polymer curve")
    else:
        ax.scatter(*pts[-1], color="tab:red", s=45, label="end")
        ax.set_title(f"{result.polymer_id} open polymer curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    set_axes_equal(ax, pts)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_plotly_graph(result: PolymerInputResult) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    html_path = FIGURE_DIR / f"{result.polymer_id}_graph.html"
    fig = kg.plot_3D_graph_plotly(result.graph)
    fig.update_layout(title=f"{result.polymer_id} as MultiGraph(pos/pts)")
    fig.write_html(str(html_path))
    return html_path


def print_result_summary(
    result: PolymerInputResult,
    png_path: Path,
    html_path: Path,
) -> None:
    pts = edge_points(result)
    print(f"Polymer ID: {result.polymer_id}")
    print(f"Source format: {result.source_format}")
    print(f"Source path: {result.source_path}")
    print(f"Closed curve: {result.closed}")
    print(f"Input coordinate shape: {result.coords.shape}")
    print(f"Graph edge pts shape: {pts.shape}")
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
    write_example_inputs()
    results = [
        from_lammps_dump(
            LAMMPS_DUMP,
            molecule_id=7,
            closed=False,
            polymer_id="linear_polymer_lammps_dump",
        ),
        from_gromacs_gro(
            GRO_SNAPSHOT,
            atom_name="BB",
            residue_name="RNG",
            closed=True,
            closure="direct",
            polymer_id="ring_polymer_gromacs_gro",
        ),
    ]

    print("Polymer simulation snapshot input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")

    for result in results:
        png_path = plot_matplotlib_curve(result)
        html_path = write_plotly_graph(result)
        print_result_summary(result, png_path, html_path)


if __name__ == "__main__":
    main()
