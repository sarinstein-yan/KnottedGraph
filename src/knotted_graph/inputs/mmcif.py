"""mmCIF backbone input adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shlex
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np

from .coordinate_chain import coordinates_to_multigraph, validate_curve_graph
from .pdb import format_chain_counts, normalize_pdb_id


RCSB_MMCIF_DOWNLOAD_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.cif"


@dataclass
class MMCIFBackboneInputResult:
    """Container for an atom trace extracted from an RCSB mmCIF file."""

    pdb_id: str
    chain_id: str
    model_id: int
    atom_name: str
    source_url: str | None
    cif_path: Path
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


def rcsb_mmcif_url(pdb_id: str) -> str:
    return RCSB_MMCIF_DOWNLOAD_TEMPLATE.format(pdb_id=normalize_pdb_id(pdb_id))


def cif_path_for(pdb_id: str, data_dir) -> Path:
    return Path(data_dir) / f"{normalize_pdb_id(pdb_id)}.cif"


def coords_npy_path_for(pdb_id: str, atom_name: str, data_dir) -> Path:
    safe_atom = atom_name.replace("'", "prime").replace("*", "star")
    return Path(data_dir) / f"{normalize_pdb_id(pdb_id)}_{safe_atom}_coords.npy"


def _looks_like_pdb_id(value: str) -> bool:
    path = Path(value)
    return path.suffix == "" and len(value.strip()) == 4


def _download_cif_if_needed(pdb_id: str, out_path: Path, url: str) -> bool:
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


def _resolve_mmcif_source(
    source,
    *,
    pdb_id: str | None,
    data_dir,
    download: bool,
) -> tuple[str, Path, str | None, bool]:
    if isinstance(source, Path):
        path = source
        return normalize_pdb_id(pdb_id or path.stem), path, None, False

    source_text = str(source)
    path = Path(source_text)
    if path.suffix.lower() in {".cif", ".mmcif"} or path.exists():
        return normalize_pdb_id(pdb_id or path.stem), path, None, False

    if not _looks_like_pdb_id(source_text):
        raise ValueError(
            "mmCIF source must be a PDB ID such as '1EHZ' or a path to a .cif file."
        )

    resolved_pdb_id = normalize_pdb_id(pdb_id or source_text)
    out_dir = Path(data_dir) if data_dir is not None else Path.cwd()
    path = cif_path_for(resolved_pdb_id, out_dir)
    url = rcsb_mmcif_url(resolved_pdb_id)
    downloaded = _download_cif_if_needed(resolved_pdb_id, path, url) if download else False
    if not path.exists():
        raise FileNotFoundError(f"mmCIF file does not exist: {path}")
    return resolved_pdb_id, path, url, downloaded


def _clean_value(value: str | None, default: str = "?") -> str:
    if value is None or value in {".", "?"}:
        return default
    return value


def _atom_name(row: dict) -> str:
    return _clean_value(
        row.get("_atom_site.auth_atom_id"),
        _clean_value(row.get("_atom_site.label_atom_id")),
    )


def _chain_id(row: dict) -> str:
    return _clean_value(
        row.get("_atom_site.auth_asym_id"),
        _clean_value(row.get("_atom_site.label_asym_id")),
    )


def _model_id(row: dict) -> int:
    raw = _clean_value(row.get("_atom_site.pdbx_PDB_model_num"), "1")
    try:
        return int(raw)
    except ValueError:
        return 1


def _is_primary_altloc(row: dict) -> bool:
    altloc = _clean_value(row.get("_atom_site.label_alt_id"), "")
    return altloc in {"", "A"}


def iter_atom_site_rows(cif_path):
    """Yield row dictionaries from the first ``_atom_site`` loop."""
    lines = Path(cif_path).read_text().splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue

        index += 1
        tags = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("_"):
                tags.append(stripped.split()[0])
                index += 1
                continue
            break

        if not tags:
            continue
        is_atom_site = all(tag.startswith("_atom_site.") for tag in tags)

        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped == "#":
                index += 1
                break
            if stripped == "loop_" or stripped.startswith("_"):
                break
            if is_atom_site:
                values = shlex.split(stripped, posix=True)
                if len(values) == len(tags):
                    yield dict(zip(tags, values))
            index += 1


def _row_matches_atom(row: dict, *, atom_name: str, model_id: int) -> bool:
    if _clean_value(row.get("_atom_site.group_PDB")) != "ATOM":
        return False
    if _model_id(row) != model_id:
        return False
    if _atom_name(row) != atom_name:
        return False
    return _is_primary_altloc(row)


def available_mmcif_atom_chains(
    cif_path,
    *,
    atom_name: str,
    model_id: int = 1,
) -> Counter:
    counts = Counter()
    for row in iter_atom_site_rows(cif_path):
        if _row_matches_atom(row, atom_name=atom_name, model_id=model_id):
            counts[_chain_id(row)] += 1
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
        raise ValueError("No matching atoms were found in the mmCIF _atom_site loop.")
    raise ValueError(
        "Multiple chains are present; pass chain_id explicitly. "
        f"Available chains: {format_chain_counts(chain_counts)}"
    )


def parse_mmcif_backbone(
    cif_path,
    *,
    atom_name: str,
    chain_id: str,
    model_id: int = 1,
) -> tuple[np.ndarray, list[dict], list[str]]:
    coords = []
    records = []
    issues = []
    skipped_altloc = 0

    for row_number, row in enumerate(iter_atom_site_rows(cif_path), start=1):
        if _clean_value(row.get("_atom_site.group_PDB")) != "ATOM":
            continue
        if _model_id(row) != model_id:
            continue
        if _atom_name(row) != atom_name:
            continue
        if not _is_primary_altloc(row):
            skipped_altloc += 1
            continue
        if _chain_id(row) != chain_id:
            continue

        try:
            x = float(row["_atom_site.Cartn_x"])
            y = float(row["_atom_site.Cartn_y"])
            z = float(row["_atom_site.Cartn_z"])
        except (KeyError, ValueError) as exc:
            issues.append(f"atom_site row {row_number}: could not parse coordinates ({exc})")
            continue

        records.append(
            {
                "chain_id": chain_id,
                "residue_name": _clean_value(
                    row.get("_atom_site.auth_comp_id"),
                    _clean_value(row.get("_atom_site.label_comp_id")),
                ),
                "resseq": _clean_value(
                    row.get("_atom_site.auth_seq_id"),
                    _clean_value(row.get("_atom_site.label_seq_id")),
                ),
                "atom_name": atom_name,
                "model_id": model_id,
                "row_number": row_number,
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


def from_mmcif_backbone(
    source,
    *,
    pdb_id: str | None = None,
    chain_id: str | None = None,
    atom_name: str = "CA",
    model_id: int = 1,
    data_dir=None,
    download: bool = True,
    save_coords: bool = False,
    coords_npy_path=None,
    input_id: str | None = None,
    input_kind: str = "mmcif_backbone",
    closed: bool = False,
    closure: str | None = None,
    metadata: dict | None = None,
) -> MMCIFBackboneInputResult:
    """Load an ordered atom trace from an RCSB-style mmCIF file.

    ``source`` may be a local ``.cif``/``.mmcif`` path or a four-character
    RCSB identifier. The parser reads the first ``_atom_site`` loop and expects
    each complete atom-site row on one physical line. When more than one chain
    contains the requested atom, pass ``chain_id`` explicitly; the loader never
    chooses a biological chain silently.

    The returned ``coords`` retain source order. Its ``graph`` follows the
    package's embedded ``MultiGraph(pos/pts)`` contract. Recoverable malformed
    atom rows are recorded in ``result.issues`` when enough valid atoms remain;
    fatal schema, selection, and insufficient-coordinate failures raise.
    """
    if model_id < 1:
        raise ValueError("model_id must be a positive integer.")
    atom_name = atom_name.strip()
    if not atom_name:
        raise ValueError("atom_name must not be empty.")

    resolved_pdb_id, cif_path, source_url, downloaded = _resolve_mmcif_source(
        source,
        pdb_id=pdb_id,
        data_dir=data_dir,
        download=download,
    )
    chain_counts = available_mmcif_atom_chains(cif_path, atom_name=atom_name, model_id=model_id)
    selected_chain_id = select_chain_id(chain_id, chain_counts)
    coords, records, issues = parse_mmcif_backbone(
        cif_path,
        atom_name=atom_name,
        chain_id=selected_chain_id,
        model_id=model_id,
    )
    if coords.shape[0] < 2:
        raise RuntimeError(
            f"Only {coords.shape[0]} {atom_name!r} atoms found for "
            f"{resolved_pdb_id} chain {selected_chain_id}, model {model_id}."
        )

    if coords_npy_path is not None:
        resolved_coords_path = Path(coords_npy_path)
    elif save_coords:
        out_dir = Path(data_dir) if data_dir is not None else cif_path.parent
        resolved_coords_path = coords_npy_path_for(resolved_pdb_id, atom_name, out_dir)
    else:
        resolved_coords_path = None
    coords_saved, saved_coords_shape = _save_coords(coords, resolved_coords_path)

    meta = dict(metadata or {})
    meta.update(
        {
            "source": "RCSB mmCIF" if source_url else "mmCIF",
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
        input_id=input_id or f"{resolved_pdb_id}_{selected_chain_id}_{atom_name}_mmcif",
        source_format="mmcif",
        source_path=cif_path,
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

    return MMCIFBackboneInputResult(
        pdb_id=resolved_pdb_id,
        chain_id=selected_chain_id,
        model_id=model_id,
        atom_name=atom_name,
        source_url=source_url,
        cif_path=cif_path,
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


__all__ = [
    "MMCIFBackboneInputResult",
    "available_mmcif_atom_chains",
    "cif_path_for",
    "coords_npy_path_for",
    "from_mmcif_backbone",
    "iter_atom_site_rows",
    "parse_mmcif_backbone",
    "rcsb_mmcif_url",
    "select_chain_id",
]
