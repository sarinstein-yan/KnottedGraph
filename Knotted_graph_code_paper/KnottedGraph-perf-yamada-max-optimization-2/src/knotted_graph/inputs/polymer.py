"""Polymer simulation snapshot input adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .coordinate_chain import coordinates_to_multigraph, validate_coords, validate_curve_graph


@dataclass
class PolymerInputResult:
    """Container for one polymer chain extracted from a simulation file."""

    polymer_id: str
    source_path: Path
    source_format: str
    closed: bool
    closure_method: str | None
    coords: np.ndarray
    graph: nx.MultiGraph
    metadata: dict
    issues: list[str]


def _as_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def write_lammps_dump(
    coords,
    path,
    *,
    molecule_id: int = 1,
    atom_type: int = 1,
    timestep: int = 0,
) -> None:
    """Write a minimal LAMMPS dump with columns ``id mol type x y z``."""
    arr = validate_coords(coords)
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mins = arr.min(axis=0) - 1.0
    maxs = arr.max(axis=0) + 1.0
    lines = [
        "ITEM: TIMESTEP",
        str(timestep),
        "ITEM: NUMBER OF ATOMS",
        str(arr.shape[0]),
        "ITEM: BOX BOUNDS pp pp pp",
        f"{mins[0]:.8f} {maxs[0]:.8f}",
        f"{mins[1]:.8f} {maxs[1]:.8f}",
        f"{mins[2]:.8f} {maxs[2]:.8f}",
        "ITEM: ATOMS id mol type x y z",
    ]
    for atom_id, (x, y, z) in enumerate(arr, start=1):
        lines.append(f"{atom_id} {molecule_id} {atom_type} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n")


def load_lammps_dump_coords(
    path,
    *,
    molecule_id: int | None = 1,
    sort_column: str = "id",
) -> np.ndarray:
    """Load one chain from the first frame of a LAMMPS dump file."""
    path = _as_path(path)
    lines = path.read_text().splitlines()
    atom_header_index = None
    columns = None
    for index, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            atom_header_index = index
            columns = line.split()[2:]
            break
    if atom_header_index is None or columns is None:
        raise ValueError("LAMMPS dump is missing an 'ITEM: ATOMS' section.")

    required = {"x", "y", "z"}
    if not required.issubset(columns):
        raise ValueError(
            "Only unscaled x/y/z LAMMPS coordinates are supported. "
            f"Found columns: {columns}"
        )
    column_index = {name: idx for idx, name in enumerate(columns)}
    if sort_column not in column_index:
        raise ValueError(f"sort_column {sort_column!r} is not present in LAMMPS dump.")

    rows = []
    for raw_line in lines[atom_header_index + 1 :]:
        if raw_line.startswith("ITEM:"):
            break
        if not raw_line.strip():
            continue
        tokens = raw_line.split()
        if len(tokens) < len(columns):
            raise ValueError(f"could not parse atom row: {raw_line!r}")
        if molecule_id is not None and "mol" in column_index:
            try:
                row_molecule_id = int(tokens[column_index["mol"]])
            except ValueError as exc:
                raise ValueError(f"invalid molecule id in atom row: {raw_line!r}") from exc
            if row_molecule_id != molecule_id:
                continue
        rows.append(tokens)

    if not rows:
        raise ValueError(f"No LAMMPS atoms found for molecule_id={molecule_id!r}.")

    try:
        rows.sort(key=lambda row: int(float(row[column_index[sort_column]])))
        coords = [
            (
                float(row[column_index["x"]]),
                float(row[column_index["y"]]),
                float(row[column_index["z"]]),
            )
            for row in rows
        ]
    except ValueError as exc:
        raise ValueError("LAMMPS atom rows contain non-numeric values.") from exc
    return validate_coords(coords)


def write_gro_coords(
    coords,
    path,
    *,
    residue_name: str = "POL",
    atom_name: str = "BB",
    title: str = "polymer coordinate curve",
    input_unit_scale: float = 10.0,
) -> None:
    """Write a minimal GROMACS `.gro` file."""
    arr = validate_coords(coords)
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, f"{arr.shape[0]:5d}"]
    gro_coords = arr / input_unit_scale
    for atom_id, (x, y, z) in enumerate(gro_coords, start=1):
        residue_id = 1 + (atom_id - 1) // 99999
        lines.append(
            f"{residue_id % 100000:5d}{residue_name[:5]:<5}"
            f"{atom_name[:5]:>5}{atom_id % 100000:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append("   10.00000   10.00000   10.00000")
    path.write_text("\n".join(lines) + "\n")


def _parse_gro_coord_line(line: str) -> tuple[str, str, int, tuple[float, float, float]]:
    if len(line) >= 44:
        try:
            residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_id = int(line[15:20])
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            return residue_name, atom_name, atom_id, (x, y, z)
        except ValueError:
            pass

    tokens = line.split()
    if len(tokens) < 6:
        raise ValueError(f"could not parse .gro coordinate row: {line!r}")
    residue_name = tokens[0][-3:]
    atom_name = tokens[1]
    atom_id = int(tokens[2])
    return residue_name, atom_name, atom_id, tuple(float(value) for value in tokens[3:6])


def load_gromacs_gro_coords(
    path,
    *,
    atom_name: str | None = None,
    residue_name: str | None = None,
    output_unit_scale: float = 10.0,
) -> np.ndarray:
    """Load ordered coordinates from a GROMACS `.gro` snapshot."""
    path = _as_path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(".gro file is too short.")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(".gro file second line must contain the atom count.") from exc
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f".gro atom count says {atom_count}, found {len(atom_lines)} rows.")

    rows = []
    for line in atom_lines:
        row_residue, row_atom, atom_id, coord = _parse_gro_coord_line(line)
        if atom_name is not None and row_atom != atom_name:
            continue
        if residue_name is not None and row_residue != residue_name:
            continue
        rows.append((atom_id, coord))

    if not rows:
        raise ValueError("No .gro atoms matched the requested filters.")
    rows.sort(key=lambda item: item[0])
    coords = np.asarray([coord for _, coord in rows], dtype=float) * output_unit_scale
    return validate_coords(coords)


def _polymer_result(
    coords: np.ndarray,
    *,
    path: Path,
    source_format: str,
    polymer_id: str,
    closed: bool,
    closure: str | None,
    metadata: dict,
) -> PolymerInputResult:
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        closure=closure,
        input_id=polymer_id,
        source_format=source_format,
        source_path=path,
        metadata=metadata,
    )
    graph.graph["input_kind"] = "polymer_snapshot"
    issues = validate_curve_graph(graph)
    return PolymerInputResult(
        polymer_id=polymer_id,
        source_path=path,
        source_format=source_format,
        closed=bool(closed or graph.graph.get("graph_is_closed")),
        closure_method=closure,
        coords=coords,
        graph=graph,
        metadata=metadata,
        issues=issues,
    )


def from_lammps_dump(
    path,
    *,
    molecule_id: int | None = 1,
    sort_column: str = "id",
    closed: bool = False,
    closure: str | None = None,
    polymer_id: str | None = None,
    metadata: dict | None = None,
) -> PolymerInputResult:
    """Load one polymer coordinate chain from a LAMMPS dump file."""
    source_path = _as_path(path)
    coords = load_lammps_dump_coords(
        source_path,
        molecule_id=molecule_id,
        sort_column=sort_column,
    )
    resolved_polymer_id = polymer_id or source_path.stem
    meta = dict(metadata or {})
    meta["molecule_id"] = molecule_id
    meta["sort_column"] = sort_column
    return _polymer_result(
        coords,
        path=source_path,
        source_format="lammps_dump",
        polymer_id=resolved_polymer_id,
        closed=closed,
        closure=closure,
        metadata=meta,
    )


def from_gromacs_gro(
    path,
    *,
    atom_name: str | None = None,
    residue_name: str | None = None,
    output_unit_scale: float = 10.0,
    closed: bool = False,
    closure: str | None = None,
    polymer_id: str | None = None,
    metadata: dict | None = None,
) -> PolymerInputResult:
    """Load one polymer coordinate chain from a GROMACS `.gro` file."""
    source_path = _as_path(path)
    coords = load_gromacs_gro_coords(
        source_path,
        atom_name=atom_name,
        residue_name=residue_name,
        output_unit_scale=output_unit_scale,
    )
    resolved_polymer_id = polymer_id or source_path.stem
    meta = dict(metadata or {})
    meta["atom_name"] = atom_name
    meta["residue_name"] = residue_name
    meta["output_unit_scale"] = output_unit_scale
    return _polymer_result(
        coords,
        path=source_path,
        source_format="gro",
        polymer_id=resolved_polymer_id,
        closed=closed,
        closure=closure,
        metadata=meta,
    )


__all__ = [
    "PolymerInputResult",
    "from_gromacs_gro",
    "from_lammps_dump",
    "load_gromacs_gro_coords",
    "load_lammps_dump_coords",
    "write_gro_coords",
    "write_lammps_dump",
]
