"""Compatibility wrapper for the core nucleic-acid PDB input API."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from knotted_graph.inputs.pdb import (
    NUCLEIC_RESIDUES,
    PDBBackboneInputResult as NucleicBackboneResult,
    available_pdb_backbone_chains,
    coords_npy_path_for as _core_coords_npy_path_for,
    format_chain_counts,
    from_nucleic_acid_backbone,
    normalize_pdb_id,
    parse_pdb_backbone,
    pdb_path_for as _core_pdb_path_for,
    rcsb_pdb_url,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def pdb_path_for(pdb_id: str, data_dir: Path = DATA_DIR) -> Path:
    return _core_pdb_path_for(pdb_id, data_dir)


def coords_npy_path_for(
    pdb_id: str,
    atom_name: str,
    data_dir: Path = DATA_DIR,
) -> Path:
    return _core_coords_npy_path_for(pdb_id, atom_name, data_dir)


def download_pdb_if_needed(pdb_id: str, out_path: Path | None = None, url: str | None = None) -> bool:
    pdb_id = normalize_pdb_id(pdb_id)
    out_path = out_path or pdb_path_for(pdb_id, DATA_DIR)
    url = url or rcsb_pdb_url(pdb_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    request = Request(url, headers={"User-Agent": "knotted-graph-dna-smoke-test"})
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(out_path)
    return True


def available_backbone_chains(pdb_path: Path, *, atom_name: str, model_id: int = 1):
    return available_pdb_backbone_chains(
        pdb_path,
        atom_name=atom_name,
        model_id=model_id,
        residue_names=NUCLEIC_RESIDUES,
    )


def select_chain_id(chain_id, chain_counts):
    from knotted_graph.inputs.pdb import select_chain_id as _select_chain_id

    return _select_chain_id(chain_id, chain_counts)


def parse_nucleic_backbone(
    pdb_path: Path,
    *,
    atom_name: str,
    chain_id: str | None = None,
    model_id: int = 1,
):
    chain_counts = available_backbone_chains(pdb_path, atom_name=atom_name, model_id=model_id)
    selected_chain = select_chain_id(chain_id, chain_counts)
    return parse_pdb_backbone(
        pdb_path,
        atom_name=atom_name,
        chain_id=selected_chain,
        model_id=model_id,
        residue_names=NUCLEIC_RESIDUES,
    )


def save_coords(coords: np.ndarray, out_path: Path) -> tuple[bool, tuple[int, ...]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coords)
    loaded = np.load(out_path)
    return out_path.exists() and np.array_equal(loaded, coords), loaded.shape


def backbone_to_multigraph(coords, records, *, pdb_id: str, atom_name: str, model_id: int = 1):
    from knotted_graph.inputs import from_coordinate_chain

    chain_id = records[0]["chain_id"] if records else "?"
    result = from_coordinate_chain(
        coords,
        input_id=f"{normalize_pdb_id(pdb_id)}_{chain_id}_{atom_name}_pdb",
        metadata={
            "source": "RCSB PDB",
            "pdb_id": normalize_pdb_id(pdb_id),
            "chain_id": chain_id,
            "model_id": model_id,
            "atom_name": atom_name,
            "records": records,
        },
    )
    result.graph.graph["input_kind"] = "nucleic_acid_backbone"
    return result.graph


def validate_backbone_graph(graph, expected_points: int):
    from knotted_graph.inputs.coordinate_chain import validate_curve_graph

    return validate_curve_graph(graph)


def build_nucleic_backbone(
    pdb_id: str,
    *,
    chain_id: str | None = None,
    atom_name: str = "P",
    model_id: int = 1,
    data_dir: Path = DATA_DIR,
) -> NucleicBackboneResult:
    return from_nucleic_acid_backbone(
        pdb_id,
        chain_id=chain_id,
        atom_name=atom_name,
        model_id=model_id,
        data_dir=data_dir,
        save_coords=True,
    )
