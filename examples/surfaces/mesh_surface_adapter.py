"""Arbitrary surface/mesh adapter for Task 2 prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SUPPORTED_MESH_SUFFIXES = {".obj", ".off", ".ply", ".stl", ".vtk", ".vtp"}


@dataclass
class MeshSurfaceResult:
    """Container for a loaded surface mesh."""

    mesh_id: str
    source_path: Path
    source_format: str
    mesh: pv.PolyData
    issues: list[str]


def load_surface_mesh(path: Path, *, triangulate: bool = True) -> pv.PolyData:
    """Load a mesh file into a PyVista ``PolyData`` surface."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(
            f"Unsupported mesh suffix {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_MESH_SUFFIXES)}"
        )

    if suffix == ".off":
        mesh = read_off_mesh(path)
    else:
        mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine().extract_geometry()
    elif not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_geometry()

    mesh = mesh.clean()
    if triangulate:
        mesh = mesh.triangulate()
    return mesh


def validate_surface_mesh(mesh: pv.PolyData) -> list[str]:
    """Return validation issues for a surface mesh."""
    issues = []
    if not isinstance(mesh, pv.PolyData):
        return [f"mesh is not a PyVista PolyData: {type(mesh)!r}"]
    if mesh.n_points == 0:
        issues.append("mesh has no points")
    if mesh.n_cells == 0:
        issues.append("mesh has no cells")
    if not np.isfinite(mesh.points).all():
        issues.append("mesh points contain NaN or infinite values")
    if mesh.n_open_edges > 0:
        issues.append(f"mesh has {mesh.n_open_edges} open boundary edges")
    return issues


def build_surface_from_mesh_file(path: Path, *, mesh_id: str | None = None) -> MeshSurfaceResult:
    mesh = load_surface_mesh(path)
    issues = validate_surface_mesh(mesh)
    mesh_id = mesh_id or path.stem
    return MeshSurfaceResult(
        mesh_id=mesh_id,
        source_path=path,
        source_format=path.suffix.lower().lstrip("."),
        mesh=mesh,
        issues=issues,
    )


def mesh_bounds_span(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray, float]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    mins = np.array([bounds[0], bounds[2], bounds[4]])
    maxs = np.array([bounds[1], bounds[3], bounds[5]])
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    return mins, maxs, span


def mesh_summary(result: MeshSurfaceResult) -> dict:
    mins, maxs, span = mesh_bounds_span(result.mesh)
    return {
        "mesh_id": result.mesh_id,
        "source_format": result.source_format,
        "source_path": str(result.source_path),
        "n_points": result.mesh.n_points,
        "n_cells": result.mesh.n_cells,
        "n_open_edges": result.mesh.n_open_edges,
        "bounds_min": mins.tolist(),
        "bounds_max": maxs.tolist(),
        "span": span,
    }


def _triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    mesh = mesh.triangulate()
    faces = mesh.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected a triangulated mesh.")
    return faces[:, 1:]


def write_obj_mesh(mesh: pv.PolyData, path: Path) -> None:
    """Write a small OBJ file from a triangulated PyVista mesh."""
    mesh = mesh.triangulate().clean()
    faces = _triangle_faces(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Task 2 mesh surface prototype"]
    for x, y, z in mesh.points:
        lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n")


def _non_comment_off_lines(path: Path) -> list[str]:
    lines = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def read_off_mesh(path: Path) -> pv.PolyData:
    """Read a simple OFF surface mesh."""
    lines = _non_comment_off_lines(path)
    if not lines or lines[0] != "OFF":
        raise ValueError(f"OFF file must start with OFF header: {path}")
    if len(lines) < 2:
        raise ValueError(f"OFF file is missing counts line: {path}")
    counts = lines[1].split()
    if len(counts) < 2:
        raise ValueError("OFF counts line must contain vertex and face counts.")
    n_vertices = int(counts[0])
    n_faces = int(counts[1])
    vertex_lines = lines[2 : 2 + n_vertices]
    face_lines = lines[2 + n_vertices : 2 + n_vertices + n_faces]
    if len(vertex_lines) != n_vertices or len(face_lines) != n_faces:
        raise ValueError("OFF file ended before all vertices/faces were read.")

    points = np.asarray(
        [[float(value) for value in line.split()[:3]] for line in vertex_lines],
        dtype=float,
    )
    faces = []
    for line in face_lines:
        tokens = line.split()
        face_size = int(tokens[0])
        indices = [int(value) for value in tokens[1 : 1 + face_size]]
        if len(indices) != face_size:
            raise ValueError(f"OFF face has wrong index count: {line!r}")
        faces.extend([face_size, *indices])
    return pv.PolyData(points, np.asarray(faces, dtype=np.int64))


def write_off_mesh(mesh: pv.PolyData, path: Path) -> None:
    """Write a small OFF file from a PyVista surface mesh."""
    mesh = mesh.clean()
    faces = mesh.faces.reshape(-1, mesh.faces[0] + 1) if mesh.n_cells == 1 else None
    if faces is None:
        offset = 0
        parsed_faces = []
        raw_faces = mesh.faces
        while offset < raw_faces.size:
            face_size = int(raw_faces[offset])
            indices = raw_faces[offset + 1 : offset + 1 + face_size]
            parsed_faces.append(indices)
            offset += face_size + 1
    else:
        parsed_faces = [face[1:] for face in faces]

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["OFF", f"{mesh.n_points} {len(parsed_faces)} 0"]
    for x, y, z in mesh.points:
        lines.append(f"{x:.8f} {y:.8f} {z:.8f}")
    for face in parsed_faces:
        values = " ".join(str(int(index)) for index in face)
        lines.append(f"{len(face)} {values}")
    path.write_text("\n".join(lines) + "\n")
