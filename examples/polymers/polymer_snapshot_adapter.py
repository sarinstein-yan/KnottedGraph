"""Compatibility wrapper for the core polymer input API."""

from __future__ import annotations

from pathlib import Path

from knotted_graph.inputs.polymer import (
    PolymerInputResult as PolymerSnapshotResult,
    from_gromacs_gro,
    from_lammps_dump,
    load_gromacs_gro_coords as load_gro_coords,
    load_lammps_dump_coords,
    write_gro_coords,
    write_lammps_dump,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def build_polymer_from_lammps_dump(
    path: Path,
    *,
    molecule_id: int | None = 1,
    closed: bool = False,
    polymer_id: str | None = None,
) -> PolymerSnapshotResult:
    return from_lammps_dump(
        path,
        molecule_id=molecule_id,
        closed=closed,
        polymer_id=polymer_id,
    )


def build_polymer_from_gro(
    path: Path,
    *,
    atom_name: str | None = None,
    residue_name: str | None = None,
    closed: bool = False,
    polymer_id: str | None = None,
) -> PolymerSnapshotResult:
    return from_gromacs_gro(
        path,
        atom_name=atom_name,
        residue_name=residue_name,
        closed=closed,
        closure="direct" if closed else None,
        polymer_id=polymer_id,
    )
