"""Smoke test volumetric scalar-field inputs for Task 2."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv

from volumetric_field_adapter import (
    DATA_DIR,
    VolumetricSurfaceResult,
    build_surface_from_scalar_field_file,
    write_npz_scalar_field,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"

SPHERE_LEVELSET_NPZ = DATA_DIR / "sphere_levelset_volume.npz"
ELLIPSOID_LEVELSET_NPY = DATA_DIR / "ellipsoid_levelset_volume.npy"


def make_centered_grid(n: int = 72, extent: float = 1.35):
    axis = np.linspace(-extent, extent, n)
    return np.meshgrid(axis, axis, axis, indexing="ij"), axis


def make_sphere_levelset(n: int = 72) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    (x, y, z), axis = make_centered_grid(n=n, extent=1.35)
    radius = 0.82
    values = np.sqrt(x * x + y * y + z * z) - radius
    spacing_value = float(axis[1] - axis[0])
    origin = (float(axis[0]), float(axis[0]), float(axis[0]))
    spacing = (spacing_value, spacing_value, spacing_value)
    return values, spacing, origin


def make_ellipsoid_levelset(n: int = 64) -> np.ndarray:
    axis = np.linspace(-1.25, 1.25, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return (x / 0.85) ** 2 + (y / 0.55) ** 2 + (z / 1.05) ** 2 - 1.0


def write_example_inputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    values, spacing, origin = make_sphere_levelset()
    write_npz_scalar_field(values, SPHERE_LEVELSET_NPZ, spacing=spacing, origin=origin)
    np.save(ELLIPSOID_LEVELSET_NPY, make_ellipsoid_levelset())


def mesh_bounds_span(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray, float]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    mins = np.array([bounds[0], bounds[2], bounds[4]])
    maxs = np.array([bounds[1], bounds[3], bounds[5]])
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    return mins, maxs, span


def add_projected_silhouettes(plotter: pv.Plotter, mesh: pv.PolyData) -> None:
    mins, _, span = mesh_bounds_span(mesh)
    pad = 0.08 * span
    origins = (
        np.array([mins[0] - pad, 0.0, 0.0]),
        np.array([0.0, mins[1] - pad, 0.0]),
        np.array([0.0, 0.0, mins[2] - pad]),
    )
    for normal, origin in zip(np.eye(3), origins):
        projected = mesh.project_points_to_plane(normal=normal, origin=origin)
        plotter.add_mesh(projected, color="#606060", opacity=0.12, smooth_shading=True)


def render_surface(result: VolumetricSurfaceResult) -> tuple[Path, Path, Path, Path, list[str]]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{result.field_id}.png"
    html_path = FIGURE_DIR / f"{result.field_id}.html"
    svg_path = FIGURE_DIR / f"{result.field_id}.svg"
    mesh_path = DATA_DIR / f"{result.field_id}_isosurface.vtp"

    mesh = result.mesh
    mesh.save(mesh_path)
    mins, maxs, span = mesh_bounds_span(mesh)
    center = 0.5 * (mins + maxs)

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("white")
    add_projected_silhouettes(plotter, mesh)
    plotter.add_mesh(
        mesh,
        color="#2484c6",
        opacity=0.92,
        smooth_shading=True,
        specular=0.35,
        specular_power=18,
        name=result.field_id,
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
    except Exception as exc:  # pragma: no cover
        issues.append(f"HTML export failed: {exc}")
    try:
        plotter.screenshot(str(png_path))
    except Exception as exc:  # pragma: no cover
        issues.append(f"PNG screenshot failed: {exc}")
    try:
        plotter.save_graphic(str(svg_path))
    except Exception as exc:  # pragma: no cover
        issues.append(f"SVG export failed: {exc}")
    plotter.close()
    return png_path, html_path, svg_path, mesh_path, issues


def print_result_summary(
    result: VolumetricSurfaceResult,
    png_path: Path,
    html_path: Path,
    svg_path: Path,
    mesh_path: Path,
    render_issues: list[str],
) -> None:
    print(f"Field ID: {result.field_id}")
    print(f"Source format: {result.source_format}")
    print(f"Source path: {result.source_path}")
    print(f"Scalar shape: {result.values_shape}")
    print(f"Level: {result.level}")
    print(f"Spacing: {result.spacing}")
    print(f"Origin: {result.origin}")
    print(f"Mesh points: {result.mesh.n_points}")
    print(f"Mesh cells: {result.mesh.n_cells}")
    print(f"Open boundary edges: {result.mesh.n_open_edges}")
    print(f"Isosurface mesh path: {mesh_path}")
    print(f"Mesh saved successfully: {mesh_path.exists() and mesh_path.stat().st_size > 0}")
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
        build_surface_from_scalar_field_file(
            SPHERE_LEVELSET_NPZ,
            level=0.0,
            field_id="sphere_levelset_npz",
        ),
        build_surface_from_scalar_field_file(
            ELLIPSOID_LEVELSET_NPY,
            level=0.0,
            field_id="ellipsoid_levelset_npy",
        ),
    ]

    print("Volumetric scalar-field input smoke test")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print("")

    for result in results:
        png_path, html_path, svg_path, mesh_path, render_issues = render_surface(result)
        print_result_summary(result, png_path, html_path, svg_path, mesh_path, render_issues)


if __name__ == "__main__":
    main()
