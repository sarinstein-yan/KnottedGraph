"""Smoke test Fermi surface input as a precomputed surface mesh."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv


HERE = Path(__file__).resolve().parent
REPO_EXAMPLES = HERE.parent
sys.path.insert(0, str(REPO_EXAMPLES / "surfaces"))

from mesh_surface_adapter import build_surface_from_mesh_file, mesh_bounds_span, mesh_summary


DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
FERMI_SURFACE_VTP = DATA_DIR / "tight_binding_fermi_surface.vtp"
PNG_PATH = FIGURE_DIR / "tight_binding_fermi_surface.png"
HTML_PATH = FIGURE_DIR / "tight_binding_fermi_surface.html"
SVG_PATH = FIGURE_DIR / "tight_binding_fermi_surface.svg"


def tight_binding_energy(kx, ky, kz, mu: float = 2.4):
    return np.cos(kx) + np.cos(ky) + np.cos(kz) - mu


def make_fermi_surface_mesh(grid_size: int = 72) -> pv.PolyData:
    k = np.linspace(-np.pi, np.pi, grid_size)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    values = tight_binding_energy(kx, ky, kz)
    grid = pv.ImageData(
        dimensions=values.shape,
        spacing=(
            float(k[1] - k[0]),
            float(k[1] - k[0]),
            float(k[1] - k[0]),
        ),
        origin=(float(k[0]), float(k[0]), float(k[0])),
    )
    grid.point_data["energy_minus_ef"] = values.ravel(order="F")
    surface = grid.contour(isosurfaces=[0.0], scalars="energy_minus_ef")
    return surface.triangulate().clean()


def write_example_input() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_fermi_surface_mesh().save(FERMI_SURFACE_VTP)


def silhouette_origins(mesh: pv.PolyData):
    mins, _, span = mesh_bounds_span(mesh)
    pad = 0.08 * span
    return (
        np.array([mins[0] - pad, 0.0, 0.0]),
        np.array([0.0, mins[1] - pad, 0.0]),
        np.array([0.0, 0.0, mins[2] - pad]),
    )


def render_surface(mesh: pv.PolyData) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    mins, maxs, span = mesh_bounds_span(mesh)
    center = 0.5 * (mins + maxs)
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("white")

    for normal, origin in zip(np.eye(3), silhouette_origins(mesh)):
        projected = mesh.project_points_to_plane(normal=normal, origin=origin)
        plotter.add_mesh(projected, color="#606060", opacity=0.12, smooth_shading=True)

    plotter.add_mesh(
        mesh,
        color="#b15fbd",
        opacity=0.92,
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        metallic=0.05,
        name="tight_binding_fermi_surface",
    )
    plotter.add_bounding_box(color="black", line_width=1)
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        xtitle="kx",
        ytitle="ky",
        ztitle="kz",
        color="black",
        font_size=12,
    )
    camera_distance = 2.5 * span
    plotter.camera_position = [
        center + np.array([1.15, -1.45, 1.15]) * camera_distance,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.25)

    issues = []
    try:
        plotter.export_html(str(HTML_PATH))
    except Exception as exc:  # pragma: no cover
        issues.append(f"HTML export failed: {exc}")
    try:
        plotter.screenshot(str(PNG_PATH))
    except Exception as exc:  # pragma: no cover
        issues.append(f"PNG screenshot failed: {exc}")
    try:
        plotter.save_graphic(str(SVG_PATH))
    except Exception as exc:  # pragma: no cover
        issues.append(f"SVG export failed: {exc}")

    plotter.close()
    return issues


def main() -> None:
    write_example_input()
    result = build_surface_from_mesh_file(FERMI_SURFACE_VTP, mesh_id="tight_binding_fermi_surface")
    render_issues = render_surface(result.mesh)
    summary = mesh_summary(result)

    print("Fermi surface mesh input smoke test")
    print(f"Source path: {result.source_path}")
    print(f"Source format: {result.source_format}")
    print(f"Mesh points: {summary['n_points']}")
    print(f"Mesh cells: {summary['n_cells']}")
    print(f"Open boundary edges: {summary['n_open_edges']}")
    print(f"Bounds min: {summary['bounds_min']}")
    print(f"Bounds max: {summary['bounds_max']}")
    print(f"PNG path: {PNG_PATH}")
    print(f"PNG created successfully: {PNG_PATH.exists() and PNG_PATH.stat().st_size > 0}")
    print(f"HTML path: {HTML_PATH}")
    print(f"HTML created successfully: {HTML_PATH.exists() and HTML_PATH.stat().st_size > 0}")
    print(f"SVG path: {SVG_PATH}")
    print(f"SVG created successfully: {SVG_PATH.exists() and SVG_PATH.stat().st_size > 0}")
    issues = result.issues + render_issues
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Issues: none")


if __name__ == "__main__":
    main()
