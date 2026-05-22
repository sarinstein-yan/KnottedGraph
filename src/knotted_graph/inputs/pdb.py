"""PDB backbone input adapters for proteins and nucleic acids."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np

from .coordinate_chain import coordinates_to_multigraph, validate_curve_graph


RCSB_PDB_DOWNLOAD_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
NUCLEIC_RESIDUES = {
    "A",
    "C",
    "G",
    "U",
    "T",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
}


@dataclass
class PDBBackboneInputResult:
    """Container for a backbone atom trace extracted from a PDB file."""

    pdb_id: str
    chain_id: str
    model_id: int
    atom_name: str
    source_url: str | None
    pdb_path: Path
    coords_npy_path: Path | None
    downloaded: bool
    available_chains: Counter
    coords: np.ndarray
    records: list[dict]
    coords_saved: bool
    saved_coords_shape: tuple[int, ...]
    graph: nx.MultiGraph
    closed: bool
    closure_method: str | None
    metadata: dict
    issues: list[str]


def normalize_pdb_id(pdb_id: str) -> str:
    normalized = pdb_id.strip().upper()
    if not normalized:
        raise ValueError("pdb_id must not be empty.")
    return normalized


def rcsb_pdb_url(pdb_id: str) -> str:
    return RCSB_PDB_DOWNLOAD_TEMPLATE.format(pdb_id=normalize_pdb_id(pdb_id))


def format_chain_counts(chain_counts: Counter) -> str:
    if not chain_counts:
        return "none"
    return ", ".join(f"{chain}:{count}" for chain, count in sorted(chain_counts.items()))


def pdb_path_for(pdb_id: str, data_dir) -> Path:
    return Path(data_dir) / f"{normalize_pdb_id(pdb_id)}.pdb"


def coords_npy_path_for(pdb_id: str, atom_name: str, data_dir) -> Path:
    safe_atom = atom_name.replace("'", "prime").replace("*", "star")
    return Path(data_dir) / f"{normalize_pdb_id(pdb_id)}_{safe_atom}_coords.npy"


def _looks_like_pdb_id(value: str) -> bool:
    path = Path(value)
    return path.suffix == "" and len(value.strip()) == 4


def _download_pdb_if_needed(pdb_id: str, out_path: Path, url: str) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return False

    request = Request(url, headers={"User-Agent": "knotted-graph-input-adapter"})
    with urlopen(request, timeout=60) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"Download failed with HTTP status {response.status}.")
        text = response.read().decode("utf-8")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(out_path)
    return True


def _resolve_pdb_source(
    source,
    *,
    pdb_id: str | None,
    data_dir,
    download: bool,
) -> tuple[str, Path, str | None, bool]:
    if isinstance(source, Path):
        path = source
        resolved_pdb_id = normalize_pdb_id(pdb_id or path.stem)
        return resolved_pdb_id, path, None, False

    source_text = str(source)
    path = Path(source_text)
    if path.suffix.lower() == ".pdb" or path.exists():
        resolved_pdb_id = normalize_pdb_id(pdb_id or path.stem)
        return resolved_pdb_id, path, None, False

    if not _looks_like_pdb_id(source_text):
        raise ValueError(
            "PDB source must be a PDB ID such as '1CRN' or a path to a .pdb file."
        )

    resolved_pdb_id = normalize_pdb_id(pdb_id or source_text)
    out_dir = Path(data_dir) if data_dir is not None else Path.cwd()
    path = pdb_path_for(resolved_pdb_id, out_dir)
    url = rcsb_pdb_url(resolved_pdb_id)
    downloaded = _download_pdb_if_needed(resolved_pdb_id, path, url) if download else False
    if not path.exists():
        raise FileNotFoundError(f"PDB file does not exist: {path}")
    return resolved_pdb_id, path, url, downloaded


def _parse_model_id(line: str) -> int | None:
    if not line.startswith("MODEL"):
        return None
    raw = line[10:14].strip() or line[5:].strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _line_is_in_model(
    model_id: int,
    current_model_id: int | None,
    has_model_records: bool,
) -> bool:
    if not has_model_records:
        return True
    return current_model_id == model_id


def _chain_id(line: str) -> str:
    return line[21].strip() or "?"


def _atom_matches(
    line: str,
    *,
    atom_name: str,
    residue_names: set[str] | None,
) -> bool:
    if not line.startswith("ATOM"):
        return False
    if residue_names is not None and line[17:20].strip() not in residue_names:
        return False
    if line[12:16].strip() != atom_name:
        return False
    altloc = line[16].strip()
    return altloc in {"", "A"}


def available_pdb_backbone_chains(
    pdb_path,
    *,
    atom_name: str = "CA",
    model_id: int = 1,
    residue_names: set[str] | None = None,
) -> Counter:
    """Count matching backbone atoms per chain in one PDB model."""
    path = Path(pdb_path)
    counts = Counter()
    current_model_id = None
    has_model_records = False

    with path.open() as handle:
        for line in handle:
            parsed_model_id = _parse_model_id(line)
            if parsed_model_id is not None:
                has_model_records = True
                current_model_id = parsed_model_id
                continue
            if line.startswith("ENDMDL"):
                current_model_id = None
                continue
            if not _line_is_in_model(model_id, current_model_id, has_model_records):
                continue
            if _atom_matches(line, atom_name=atom_name, residue_names=residue_names):
                counts[_chain_id(line)] += 1
    return counts


def select_chain_id(chain_id: str | None, chain_counts: Counter) -> str:
    if chain_id is not None:
        selected = chain_id.strip() or "?"
        if selected not in chain_counts:
            raise ValueError(
                f"chain_id {selected!r} is not present. Available chains: "
                f"{format_chain_counts(chain_counts)}"
            )
        return selected
    if len(chain_counts) == 1:
        return next(iter(chain_counts))
    if not chain_counts:
        raise ValueError("No matching backbone atoms were found in the PDB file.")
    raise ValueError(
        "Multiple chains are present; pass chain_id explicitly. "
        f"Available chains: {format_chain_counts(chain_counts)}"
    )


def parse_pdb_backbone(
    pdb_path,
    *,
    atom_name: str,
    chain_id: str,
    model_id: int = 1,
    residue_names: set[str] | None = None,
) -> tuple[np.ndarray, list[dict], list[str]]:
    """Parse one ordered backbone atom trace from a PDB file."""
    path = Path(pdb_path)
    coords = []
    records = []
    issues = []
    skipped_altloc = 0
    current_model_id = None
    has_model_records = False

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed_model_id = _parse_model_id(line)
            if parsed_model_id is not None:
                has_model_records = True
                current_model_id = parsed_model_id
                continue
            if line.startswith("ENDMDL"):
                current_model_id = None
                continue
            if not _line_is_in_model(model_id, current_model_id, has_model_records):
                continue
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != atom_name:
                continue
            residue_name = line[17:20].strip()
            if residue_names is not None and residue_name not in residue_names:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                skipped_altloc += 1
                continue
            current_chain_id = _chain_id(line)
            if current_chain_id != chain_id:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as exc:
                issues.append(f"line {line_number}: could not parse coordinates ({exc})")
                continue

            records.append(
                {
                    "chain_id": current_chain_id,
                    "residue_name": residue_name,
                    "resseq": line[22:26].strip(),
                    "icode": line[26].strip(),
                    "atom_name": atom_name,
                    "model_id": model_id if has_model_records else 1,
                    "line_number": line_number,
                }
            )
            coords.append((x, y, z))

    if skipped_altloc:
        issues.append(f"skipped {skipped_altloc} non-primary alternate-location atoms")

    return np.asarray(coords, dtype=float), records, issues


def _save_coords(coords: np.ndarray, out_path: Path | None) -> tuple[bool, tuple[int, ...]]:
    if out_path is None:
        return False, ()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coords)
    loaded = np.load(out_path)
    return out_path.exists() and np.array_equal(loaded, coords), loaded.shape


def from_pdb_backbone(
    source,
    *,
    pdb_id: str | None = None,
    chain_id: str | None = None,
    atom_name: str = "CA",
    model_id: int = 1,
    residue_names: set[str] | None = None,
    data_dir=None,
    download: bool = True,
    save_coords: bool = False,
    coords_npy_path=None,
    input_id: str | None = None,
    input_kind: str = "pdb_backbone",
    closed: bool = False,
    closure: str | None = None,
    metadata: dict | None = None,
) -> PDBBackboneInputResult:
    """Load a protein, DNA, or RNA backbone atom trace from a PDB file.

    ``source`` may be an RCSB PDB ID or a local PDB path. The returned graph
    follows the package's ``MultiGraph(pos/pts)`` convention.
    """
    if model_id < 1:
        raise ValueError("model_id must be a positive integer.")
    atom_name = atom_name.strip()
    if not atom_name:
        raise ValueError("atom_name must not be empty.")

    resolved_pdb_id, pdb_path, source_url, downloaded = _resolve_pdb_source(
        source,
        pdb_id=pdb_id,
        data_dir=data_dir,
        download=download,
    )
    chain_counts = available_pdb_backbone_chains(
        pdb_path,
        atom_name=atom_name,
        model_id=model_id,
        residue_names=residue_names,
    )
    selected_chain_id = select_chain_id(chain_id, chain_counts)
    coords, records, issues = parse_pdb_backbone(
        pdb_path,
        atom_name=atom_name,
        chain_id=selected_chain_id,
        model_id=model_id,
        residue_names=residue_names,
    )
    if coords.shape[0] < 2:
        raise RuntimeError(
            f"Only {coords.shape[0]} {atom_name!r} atoms found for "
            f"{resolved_pdb_id} chain {selected_chain_id}, model {model_id}."
        )

    if coords_npy_path is not None:
        resolved_coords_path = Path(coords_npy_path)
    elif save_coords:
        out_dir = Path(data_dir) if data_dir is not None else pdb_path.parent
        resolved_coords_path = coords_npy_path_for(resolved_pdb_id, atom_name, out_dir)
    else:
        resolved_coords_path = None
    coords_saved, saved_coords_shape = _save_coords(coords, resolved_coords_path)

    meta = dict(metadata or {})
    meta.update(
        {
            "source": "RCSB PDB" if source_url else "PDB",
            "pdb_id": resolved_pdb_id,
            "chain_id": selected_chain_id,
            "model_id": model_id,
            "atom_name": atom_name,
            "records": records,
        }
    )
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        closure=closure,
        input_id=input_id or f"{resolved_pdb_id}_{selected_chain_id}_{atom_name}_pdb",
        source_format="pdb",
        source_path=pdb_path,
        metadata=meta,
    )
    graph.graph["input_kind"] = input_kind
    for _, _, _, edge_data in graph.edges(keys=True, data=True):
        edge_data.update(
            {
                "pdb_id": resolved_pdb_id,
                "chain_id": selected_chain_id,
                "model_id": model_id,
                "atom_name": atom_name,
                "residue_records": records,
            }
        )
    issues.extend(validate_curve_graph(graph))

    return PDBBackboneInputResult(
        pdb_id=resolved_pdb_id,
        chain_id=selected_chain_id,
        model_id=model_id,
        atom_name=atom_name,
        source_url=source_url,
        pdb_path=pdb_path,
        coords_npy_path=resolved_coords_path,
        downloaded=downloaded,
        available_chains=chain_counts,
        coords=coords,
        records=records,
        coords_saved=coords_saved,
        saved_coords_shape=saved_coords_shape,
        graph=graph,
        closed=bool(closed or graph.graph.get("graph_is_closed")),
        closure_method=closure,
        metadata=dict(metadata or {}),
        issues=issues,
    )


def from_protein_ca_backbone(source, **kwargs) -> PDBBackboneInputResult:
    """Load a protein C-alpha backbone from a PDB ID or path."""
    kwargs.setdefault("atom_name", "CA")
    kwargs.setdefault("input_kind", "protein_ca_backbone")
    return from_pdb_backbone(source, **kwargs)


def from_nucleic_acid_backbone(source, **kwargs) -> PDBBackboneInputResult:
    """Load a DNA/RNA backbone atom trace from a PDB ID or path."""
    kwargs.setdefault("atom_name", "P")
    kwargs.setdefault("residue_names", NUCLEIC_RESIDUES)
    kwargs.setdefault("input_kind", "nucleic_acid_backbone")
    return from_pdb_backbone(source, **kwargs)


__all__ = [
    "NUCLEIC_RESIDUES",
    "PDBBackboneInputResult",
    "available_pdb_backbone_chains",
    "format_chain_counts",
    "from_nucleic_acid_backbone",
    "from_pdb_backbone",
    "from_protein_ca_backbone",
    "normalize_pdb_id",
    "parse_pdb_backbone",
    "pdb_path_for",
    "coords_npy_path_for",
    "rcsb_pdb_url",
    "select_chain_id",
]
