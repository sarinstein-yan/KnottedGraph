"""Compatibility wrapper for the core mmCIF input API."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from knotted_graph.inputs.mmcif import (
    MMCIFBackboneInputResult as MMCIFBackboneResult,
    available_mmcif_atom_chains as available_atom_chains,
    cif_path_for as _core_cif_path_for,
    coords_npy_path_for as _core_coords_npy_path_for,
    format_chain_counts,
    from_mmcif_backbone,
    iter_atom_site_rows,
    parse_mmcif_backbone,
    rcsb_mmcif_url,
    select_chain_id,
)
from knotted_graph.inputs.pdb import normalize_pdb_id


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def cif_path_for(pdb_id: str, data_dir: Path = DATA_DIR) -> Path:
    return _core_cif_path_for(pdb_id, data_dir)


def coords_npy_path_for(pdb_id: str, atom_name: str, data_dir: Path = DATA_DIR) -> Path:
    return _core_coords_npy_path_for(pdb_id, atom_name, data_dir)


def download_cif_if_needed(pdb_id: str, out_path: Path | None = None, url: str | None = None) -> bool:
    pdb_id = normalize_pdb_id(pdb_id)
    out_path = out_path or cif_path_for(pdb_id, DATA_DIR)
    url = url or rcsb_mmcif_url(pdb_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    request = Request(url, headers={"User-Agent": "knotted-graph-mmcif-smoke-test"})
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(out_path)
    return True


def save_coords(coords: np.ndarray, out_path: Path) -> tuple[bool, tuple[int, ...]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coords)
    loaded = np.load(out_path)
    return out_path.exists() and np.array_equal(loaded, coords), loaded.shape


def build_mmcif_backbone(
    pdb_id: str,
    *,
    chain_id: str | None = None,
    atom_name: str = "CA",
    model_id: int = 1,
    data_dir: Path = DATA_DIR,
) -> MMCIFBackboneResult:
    return from_mmcif_backbone(
        pdb_id,
        chain_id=chain_id,
        atom_name=atom_name,
        model_id=model_id,
        data_dir=data_dir,
        save_coords=True,
    )
