"""Smoke test arbitrary surface/mesh inputs for Task 2."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv

from mesh_surface_adapter import (
    DATA_DIR,
    MeshSurfaceResult,
    build_surface_from_mesh_file,
    mesh_bounds_span,
    mesh_summary,
    write_off_mesh,
    write_obj_mesh,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"

SPHERE_PLY = DATA_DIR / "sphere_surface.ply"
TORUS_OBJ = DATA_DIR / "torus_surface.obj"
ELLIPSOID_STL = DATA_DIR / "ellipsoid_surface.stl"
CUBE_OFF = DATA_DIR / "cube_surface.off"


def make_sphere_mesh() -> pv.PolyData:
    return pv.Sphere(theta_resolution=48, phi_resolution=24).triangulate().clean()


def make_torus_mesh() -> pv.PolyData:
    torus = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.32)
    return torus.triangulate().clean()


def make_ellipsoid_mesh() -> pv.PolyData:
    sphere = pv.Sphere(theta_resolution=48, phi_resolution=24)
    return sphere.scale([1.0, 0.55, 1.35], inplace=False).triangulate().clean()


def make_cube_mesh() -> pv.PolyData:
    return pv.Cube().triangulate().clean()


def write_example_inputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_sphere_mesh().save(SPHERE_PLY)
    write_obj_mesh(make_torus_mesh(), TORUS_OBJ)
    make_ellipsoid_mesh().save(ELLIPSOID_STL)
    write_off_mesh(make_cube_mesh(), CUBE_OFF)


def silhouette_origins(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins, _, span = mesh_bounds_span(mesh)
    pad = 0.08 * span
    return (
        np.array([mins[0] - pad, 0.0, 0.0]),
        np.array([0.0, mins[1] - pad, 0.0]),
        np.array([0.0, 0.0, mins[2] - pad]),
    )


def add_projected_silhouettes(plotter: pv.Plotter, mesh: pv.PolyData) -> None:
    for normal, origin in zip(np.eye(3), silhouette_origins(mesh)):
        projected = mesh.project_points_to_plane(normal=normal, origin=origin)
        plotter.add_mesh(
            projected,
            color="#606060",
            opacity=0.12,
            smooth_shading=True,
        )


def render_mesh(result: MeshSurfaceResult) -> tuple[Path, Path, Path, list[str]]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{result.mesh_id}.png"
    html_path = FIGURE_DIR / f"{result.mesh_id}.html"
    svg_path = FIGURE_DIR / f"{result.mesh_id}.svg"

    mesh = result.mesh
    mins, maxs, span = mesh_bounds_span(mesh)
    center = 0.5 * (mins + maxs)

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("white")
    add_projected_silhouettes(plotter, mesh)
    plotter.add_mesh(
        mesh,
        color="#12a47f",
        opacity=0.9,
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        metallic=0.05,
        name=result.mesh_id,
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
    camera_distance = 2.6 * span
    plotter.camera_position = [
        center + np.array([1.2, -1.55, 1.05]) * camera_distance,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.35)

    issues = []
    try:
        plotter.export_html(str(html_path))
    except Exception as exc:  # pragma: no cover - depends on VTK/trame backend
        issues.append(f"HTML export failed: {exc}")
    try:
        plotter.screenshot(str(png_path))
    except Exception as exc:  # pragma: no cover - depends on headless rendering
        issues.append(f"PNG screenshot failed: {exc}")
    try:
        plotter.save_graphic(str(svg_path))
    except Exception as exc:  # pragma: no cover - depends on GL2PS backend
        issues.append(f"SVG export failed: {exc}")

    plotter.close()
    return png_path, html_path, svg_path, issues


def print_result_summary(
    result: MeshSurfaceResult,
    png_path: Path,
    html_path: Path,
    svg_path: Path,
    render_issues: list[str],
) -> None:
    summary = mesh_summary(result)
    print(f"Mesh ID: {summary['mesh_id']}")
    print(f"Source format: {summary['source_format']}")
    print(f"Source path: {summary['source_path']}")
    print(f"Mesh points: {summary['n_points']}")
    print(f"Mesh cells: {summary['n_cells']}")
    print(f"Open boundary edges: {summary['n_open_edges']}")
    print(f"Bounds min: {summary['bounds_min']}")
    print(f"Bounds max: {summary['bounds_max']}")
    print(f"PNG path: {png_path}")
    print(f"PNG created successfully: {png_path.exists() and png_path.stat().st_size > 0}")
    print(f"HTML path: {html_path}")
    print(f"HTML created successfully: {html_path.exists() and html_path.stat().st_size > 0}")
    print(f"SVG path: {svg_path}")
    print(f"SVG created successfully: {svg_path.exists() and svg_path.stat().st_size > 0}")

    issues = result.issues + render_issues
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Issues: none")
    print("")


def main() -> None:
    write_example_inputs()
    results = [
        build_surface_from_mesh_file(SPHERE_PLY, mesh_id="sphere_surface_ply"),
        build_surface_from_mesh_file(TORUS_OBJ, mesh_id="torus_surface_obj"),
        build_surface_from_mesh_file(ELLIPSOID_STL, mesh_id="ellipsoid_surface_stl"),
        build_surface_from_mesh_file(CUBE_OFF, mesh_id="cube_surface_off"),
    ]

    print("Arbitrary surface/mesh input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")

    for result in results:
        png_path, html_path, svg_path, render_issues = render_mesh(result)
        print_result_summary(result, png_path, html_path, svg_path, render_issues)


if __name__ == "__main__":
    main()
