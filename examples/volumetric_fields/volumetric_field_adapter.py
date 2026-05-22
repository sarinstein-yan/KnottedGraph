"""Volumetric scalar-field adapter for Task 2 prototypes.

This converts a 3D scalar field into an isosurface mesh so scalar volumes can
enter the same surface/mesh visualization path as OBJ/PLY/STL inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pyvista as pv


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

REPO_EXAMPLES = HERE.parent
sys.path.insert(0, str(REPO_EXAMPLES / "surfaces"))

from mesh_surface_adapter import validate_surface_mesh


@dataclass
class VolumetricSurfaceResult:
    """Container for an isosurface extracted from a scalar volume."""

    field_id: str
    source_path: Path
    source_format: str
    level: float
    values_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    mesh: pv.PolyData
    issues: list[str]


def validate_scalar_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D scalar field, got shape {arr.shape}.")
    if min(arr.shape) < 2:
        raise ValueError(f"Scalar field dimensions must all be >= 2, got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Scalar field contains NaN or infinite values.")
    return arr


def _triple(value, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"Expected a length-3 value, got shape {arr.shape}.")
    return tuple(float(x) for x in arr)


def load_scalar_field_file(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """Load `.npy` or `.npz` scalar-field input.

    `.npy` stores only the 3D values. `.npz` may store `values`, `spacing`, and
    `origin`, which is the more user-friendly option for physical coordinates.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return validate_scalar_values(np.load(path)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    if suffix == ".npz":
        with np.load(path) as data:
            if "values" not in data:
                raise ValueError(".npz scalar field must contain a 'values' array.")
            values = validate_scalar_values(data["values"])
            spacing = _triple(data["spacing"] if "spacing" in data else None, default=(1.0, 1.0, 1.0))
            origin = _triple(data["origin"] if "origin" in data else None, default=(0.0, 0.0, 0.0))
            return values, spacing, origin
    raise ValueError("Supported scalar-field suffixes are .npy and .npz.")


def scalar_field_to_isosurface(
    values: np.ndarray,
    *,
    level: float,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pv.PolyData:
    values = validate_scalar_values(values)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if not (min_value <= level <= max_value):
        raise ValueError(
            f"level {level} is outside the scalar range [{min_value}, {max_value}]."
        )
    grid = pv.ImageData(dimensions=values.shape, spacing=spacing, origin=origin)
    grid.point_data["scalar"] = values.ravel(order="F")
    surface = grid.contour(isosurfaces=[level], scalars="scalar")
    return surface.triangulate().clean()


def build_surface_from_scalar_field_file(
    path: Path,
    *,
    level: float,
    field_id: str | None = None,
) -> VolumetricSurfaceResult:
    values, spacing, origin = load_scalar_field_file(path)
    mesh = scalar_field_to_isosurface(values, level=level, spacing=spacing, origin=origin)
    issues = validate_surface_mesh(mesh)
    return VolumetricSurfaceResult(
        field_id=field_id or path.stem,
        source_path=path,
        source_format=path.suffix.lower().lstrip("."),
        level=level,
        values_shape=tuple(int(x) for x in values.shape),
        spacing=spacing,
        origin=origin,
        mesh=mesh,
        issues=issues,
    )


def write_npz_scalar_field(
    values: np.ndarray,
    path: Path,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    values = validate_scalar_values(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, values=values, spacing=np.asarray(spacing), origin=np.asarray(origin))
