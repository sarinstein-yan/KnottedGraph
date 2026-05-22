"""Render a protein C-alpha backbone in a PyVista tube/silhouette style."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv

from knotted_graph.inputs import PDBBackboneInputResult, from_protein_ca_backbone


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"


@dataclass
class RenderResult:
    html_path: Path
    png_path: Path
    svg_path: Path
    issues: list[str]


def default_output_prefix(pdb_id: str) -> str:
    return f"{pdb_id.lower()}_backbone"


def graph_backbone_points(result: PDBBackboneInputResult) -> np.ndarray:
    """Extract the one backbone edge's ``pts`` array from the exploratory graph."""
    edges = list(result.graph.edges(keys=True, data=True))
    if len(edges) != 1:
        raise RuntimeError(f"Expected one backbone edge, got {len(edges)}.")
    _, _, _, edge_data = edges[0]
    pts = np.asarray(edge_data["pts"], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Expected edge pts with shape (N, 3), got {pts.shape}.")
    return pts


def polyline_from_points(points: np.ndarray) -> pv.PolyData:
    """Create one connected polyline through the original C-alpha coordinates."""
    polyline = pv.PolyData(points)
    polyline.lines = np.concatenate(([points.shape[0]], np.arange(points.shape[0])))
    return polyline


def scene_scale(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    return mins, maxs, span


def silhouette_origins(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins, _, span = scene_scale(points)
    pad = 0.08 * span
    return (
        np.array([mins[0] - pad, 0.0, 0.0]),
        np.array([0.0, mins[1] - pad, 0.0]),
        np.array([0.0, 0.0, mins[2] - pad]),
    )


def add_projected_silhouettes(
    plotter: pv.Plotter,
    meshes: list[pv.DataSet],
    origins: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Add x/y/z plane projections in the style of the existing assets."""
    for mesh in meshes:
        for normal, origin in zip(np.eye(3), origins):
            projected = mesh.project_points_to_plane(normal=normal, origin=origin)
            plotter.add_mesh(
                projected,
                color="#5f5f5f",
                opacity=0.16,
                smooth_shading=True,
                specular=0.15,
            )


def render_backbone_scene(
    result: PDBBackboneInputResult,
    output_prefix: str | None = None,
) -> RenderResult:
    """Render the backbone tube figure and return non-fatal export issues."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix or default_output_prefix(result.pdb_id)
    html_path = FIGURE_DIR / f"{output_prefix}_tube.html"
    png_path = FIGURE_DIR / f"{output_prefix}_tube.png"
    svg_path = FIGURE_DIR / f"{output_prefix}_tube.svg"

    points = graph_backbone_points(result)
    mins, maxs, span = scene_scale(points)
    center = 0.5 * (mins + maxs)
    radius = 0.018 * span
    node_radius = 0.045 * span

    backbone_polyline = polyline_from_points(points)
    backbone_tube = backbone_polyline.tube(radius=radius, n_sides=28, capping=True)

    n_terminus_sphere = pv.Sphere(
        radius=node_radius,
        center=points[0],
        theta_resolution=32,
        phi_resolution=16,
    )
    c_terminus_sphere = pv.Sphere(
        radius=node_radius,
        center=points[-1],
        theta_resolution=32,
        phi_resolution=16,
    )

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("white")

    add_projected_silhouettes(
        plotter,
        meshes=[backbone_tube, n_terminus_sphere, c_terminus_sphere],
        origins=silhouette_origins(points),
    )
    plotter.add_mesh(
        backbone_tube,
        color="#2f72c4",
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        metallic=0.05,
        name=f"{result.pdb_id} CA backbone",
    )
    plotter.add_mesh(
        n_terminus_sphere,
        color="#14a85d",
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        name="N terminus",
    )
    plotter.add_mesh(
        c_terminus_sphere,
        color="#d92525",
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        name="C terminus",
    )
    plotter.add_bounding_box(color="black", line_width=1)
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        xtitle="X",
        ytitle="Y",
        ztitle="Z",
        color="black",
        font_size=12,
    )

    camera_distance = 2.35 * span
    plotter.camera_position = [
        center + np.array([1.25, -1.55, 1.15]) * camera_distance,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.45)

    export_issues = []
    try:
        plotter.export_html(str(html_path))
    except Exception as exc:  # pragma: no cover - depends on VTK/trame backend
        export_issues.append(f"HTML export failed: {exc}")

    try:
        plotter.screenshot(str(png_path))
    except Exception as exc:  # pragma: no cover - depends on headless rendering
        export_issues.append(f"PNG screenshot failed: {exc}")

    try:
        plotter.save_graphic(str(svg_path))
    except Exception as exc:  # pragma: no cover - GL2PS may be unavailable
        export_issues.append(f"SVG export failed: {exc}")

    plotter.close()
    return RenderResult(
        html_path=html_path,
        png_path=png_path,
        svg_path=svg_path,
        issues=export_issues,
    )


def run(
    pdb_id: str,
    chain_id: str | None = None,
    model_id: int = 1,
    output_prefix: str | None = None,
) -> tuple[PDBBackboneInputResult, RenderResult]:
    result = from_protein_ca_backbone(
        pdb_id,
        chain_id=chain_id,
        model_id=model_id,
        data_dir=DATA_DIR,
        save_coords=True,
    )
    render_result = render_backbone_scene(result, output_prefix=output_prefix)
    print_summary(result, render_result)
    return result, render_result


def print_summary(result: PDBBackboneInputResult, render_result: RenderResult) -> None:
    pts = graph_backbone_points(result)

    print(f"PDB ID: {result.pdb_id}")
    print(f"Source URL: {result.source_url}")
    print(f"PDB path: {result.pdb_path}")
    print(f"Downloaded new file: {result.downloaded}")
    print(f"Selected chain: {result.chain_id}")
    print(f"Selected model: {result.model_id}")
    print(f"C-alpha atoms rendered: {pts.shape[0]}")
    print(f"Coordinates NPY path: {result.coords_npy_path}")
    print(f"Coordinates saved successfully: {result.coords_saved}")
    print(f"Saved coordinates shape: {result.saved_coords_shape}")
    print(f"Graph nodes: {result.graph.number_of_nodes()}")
    print(f"Graph edges: {result.graph.number_of_edges()}")
    print(f"Graph model_id: {result.graph.graph.get('model_id')}")
    print(f"Open curve closure applied: {result.graph.graph.get('is_closed', False)}")
    print(f"Tube HTML path: {render_result.html_path}")
    print(
        "Tube HTML created successfully: "
        f"{render_result.html_path.exists() and render_result.html_path.stat().st_size > 0}"
    )
    print(f"Tube PNG path: {render_result.png_path}")
    print(
        "Tube PNG created successfully: "
        f"{render_result.png_path.exists() and render_result.png_path.stat().st_size > 0}"
    )
    print(f"Tube SVG path: {render_result.svg_path}")
    print(
        "Tube SVG created successfully: "
        f"{render_result.svg_path.exists() and render_result.svg_path.stat().st_size > 0}"
    )

    issues = result.issues + render_result.issues
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Issues: none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an RCSB PDB C-alpha backbone as a PyVista tube figure."
    )
    parser.add_argument("--pdb-id", required=True, help="RCSB PDB ID, e.g. 1CRN.")
    parser.add_argument(
        "--chain-id",
        default=None,
        help="Protein chain ID. Required when the PDB has multiple CA chains.",
    )
    parser.add_argument(
        "--model-id",
        default=1,
        type=int,
        help="PDB MODEL number to use for NMR/multi-model files. Defaults to 1.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix under examples/proteins/figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.pdb_id,
        chain_id=args.chain_id,
        model_id=args.model_id,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
