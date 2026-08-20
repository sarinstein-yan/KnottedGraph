"""Surface-mesh input adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


SUPPORTED_MESH_SUFFIXES = {".obj", ".off", ".ply", ".stl", ".vtk", ".vtp"}


@dataclass
class SurfaceInputResult:
    """Parsed surface mesh and validation details."""

    mesh_id: str
    source_path: Path
    source_format: str
    mesh: pv.PolyData
    metadata: dict
    issues: list[str]


def _as_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _non_comment_off_lines(path: Path) -> list[str]:
    lines = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def read_off_mesh(path) -> pv.PolyData:
    """Read a simple OFF surface mesh into ``pyvista.PolyData``."""
    path = _as_path(path)
    lines = _non_comment_off_lines(path)
    if not lines or lines[0] != "OFF":
        raise ValueError(f"OFF file must start with OFF header: {path}")
    if len(lines) < 2:
        raise ValueError(f"OFF file is missing counts line: {path}")

    counts = lines[1].split()
    if len(counts) < 2:
        raise ValueError("OFF counts line must contain vertex and face counts.")
    try:
        n_vertices = int(counts[0])
        n_faces = int(counts[1])
    except ValueError as exc:
        raise ValueError("OFF counts line must contain integer counts.") from exc

    vertex_lines = lines[2 : 2 + n_vertices]
    face_lines = lines[2 + n_vertices : 2 + n_vertices + n_faces]
    if len(vertex_lines) != n_vertices or len(face_lines) != n_faces:
        raise ValueError("OFF file ended before all vertices/faces were read.")

    try:
        points = np.asarray(
            [[float(value) for value in line.split()[:3]] for line in vertex_lines],
            dtype=float,
        )
    except ValueError as exc:
        raise ValueError("OFF vertex coordinates must be numeric.") from exc
    if points.shape != (n_vertices, 3):
        raise ValueError("OFF vertex rows must contain x y z coordinates.")

    faces: list[int] = []
    for line in face_lines:
        tokens = line.split()
        if not tokens:
            continue
        try:
            face_size = int(tokens[0])
            indices = [int(value) for value in tokens[1 : 1 + face_size]]
        except ValueError as exc:
            raise ValueError(f"OFF face has non-integer index: {line!r}") from exc
        if len(indices) != face_size:
            raise ValueError(f"OFF face has wrong index count: {line!r}")
        faces.extend([face_size, *indices])

    return pv.PolyData(points, np.asarray(faces, dtype=np.int64))


def load_surface_mesh(path, *, triangulate: bool = True, clean: bool = True) -> pv.PolyData:
    """Load a supported mesh file as ``pyvista.PolyData``."""
    path = _as_path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(
            f"Unsupported mesh suffix {suffix!r}. Supported: {sorted(SUPPORTED_MESH_SUFFIXES)}"
        )

    if suffix == ".off":
        mesh = read_off_mesh(path)
    else:
        mesh = pv.read(path)

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine().extract_geometry()
    elif not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_geometry()

    if clean:
        mesh = mesh.clean()
    if triangulate:
        mesh = mesh.triangulate()
    return mesh


def validate_surface_mesh(mesh: pv.PolyData) -> list[str]:
    """Return validation issues for a ``pyvista.PolyData`` surface."""
    issues: list[str] = []
    if not isinstance(mesh, pv.PolyData):
        return [f"mesh is not a PyVista PolyData: {type(mesh)!r}"]
    if mesh.n_points == 0:
        issues.append("mesh has no points")
    if mesh.n_cells == 0:
        issues.append("mesh has no cells")
    if mesh.n_points and not np.isfinite(mesh.points).all():
        issues.append("mesh points contain NaN or infinite values")
    if mesh.n_cells and mesh.n_open_edges > 0:
        issues.append(f"mesh has {mesh.n_open_edges} open boundary edges")
    return issues


def from_surface_mesh(
    path,
    *,
    mesh_id: str | None = None,
    triangulate: bool = True,
    clean: bool = True,
    metadata: dict | None = None,
) -> SurfaceInputResult:
    """Load a surface mesh for KnottedGraph-compatible workflows."""
    source_path = _as_path(path)
    mesh = load_surface_mesh(source_path, triangulate=triangulate, clean=clean)
    issues = validate_surface_mesh(mesh)
    meta = dict(metadata or {})
    resolved_mesh_id = mesh_id or source_path.stem
    mesh.field_data["input_kind"] = np.array(["surface_mesh"])
    mesh.field_data["mesh_id"] = np.array([resolved_mesh_id])
    mesh.field_data["source_format"] = np.array([source_path.suffix.lower().lstrip(".")])
    return SurfaceInputResult(
        mesh_id=resolved_mesh_id,
        source_path=source_path,
        source_format=source_path.suffix.lower().lstrip("."),
        mesh=mesh,
        metadata=meta,
        issues=issues,
    )


__all__ = [
    "SUPPORTED_MESH_SUFFIXES",
    "SurfaceInputResult",
    "from_surface_mesh",
    "load_surface_mesh",
    "read_off_mesh",
    "validate_surface_mesh",
]
