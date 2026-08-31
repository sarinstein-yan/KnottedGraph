"""Crosslinked-protein input adapters.

The helpers in this module turn PDB ``SSBOND``/``LINK`` records or mmCIF
``_struct_conn`` rows into the embedded ``networkx.MultiGraph(pos/pts)``
contract used throughout KnottedGraph.  Backbone segments are represented by
polylines between chain termini and crosslink residues; physical crosslinks
are represented by separate multiedges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import defaultdict
from pathlib import Path
import shlex
from typing import Iterable, Iterator, Mapping, Sequence

import networkx as nx
import numpy as np

from knotted_graph.core.embedding import validate_embedding

from .mmcif import _resolve_mmcif_source
from .pdb import _resolve_pdb_source


DEFAULT_CROSSLINK_TYPES = frozenset({"disulfide", "covalent"})
METAL_RESIDUES = frozenset({"CA", "CO", "CU", "FE", "K", "MG", "MN", "NA", "NI", "ZN"})
SOLVENT_RESIDUES = frozenset({"DOD", "HOH", "WAT"})


@dataclass(frozen=True, order=True)
class ResidueKey:
    """Stable identifier for one residue within one coordinate model."""

    chain_id: str
    sequence_id: str
    insertion_code: str = ""

    @property
    def label(self) -> str:
        suffix = self.insertion_code if self.insertion_code else ""
        return f"{self.chain_id}:{self.sequence_id}{suffix}"


@dataclass(frozen=True)
class CrosslinkEndpoint:
    """One endpoint of a physical crosslink."""

    residue: ResidueKey
    residue_name: str
    atom_name: str | None = None


@dataclass(frozen=True)
class CrosslinkRecord:
    """Normalized PDB/mmCIF crosslink record."""

    crosslink_id: str
    kind: str
    endpoint_a: CrosslinkEndpoint
    endpoint_b: CrosslinkEndpoint
    source_record: str
    distance: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    @property
    def chains(self) -> tuple[str, str]:
        return self.endpoint_a.residue.chain_id, self.endpoint_b.residue.chain_id

    @property
    def canonical_endpoint_key(self) -> tuple[tuple[str, str, str, str], ...]:
        endpoints = []
        for endpoint in (self.endpoint_a, self.endpoint_b):
            residue = endpoint.residue
            endpoints.append(
                (
                    residue.chain_id,
                    residue.sequence_id,
                    residue.insertion_code,
                    endpoint.atom_name or "",
                )
            )
        return tuple(sorted(endpoints))


@dataclass
class CrosslinkedProteinInputResult:
    """Parsed crosslinks, backbone coordinates, and embedded graph."""

    pdb_id: str
    source_format: str
    source_path: Path
    source_url: str | None
    downloaded: bool
    model_id: int
    chain_ids: tuple[str, ...]
    backbone_atom: str
    allowed_crosslink_types: tuple[str, ...] | None
    crosslinks: list[CrosslinkRecord]
    excluded_crosslinks: list[CrosslinkRecord]
    atom_records: list[dict]
    graph: nx.MultiGraph
    issues: list[str]
    metadata: dict


def _clean(value: object | None, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text in {".", "?"}:
        return default
    return text


def _float_or_none(value: object | None) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _sequence_number(residue: ResidueKey) -> int | None:
    try:
        return int(residue.sequence_id)
    except ValueError:
        return None


def _is_backbone_link(
    endpoint_a: CrosslinkEndpoint,
    endpoint_b: CrosslinkEndpoint,
) -> bool:
    """Return whether a LINK row is an adjacent peptide C--N bond."""

    if endpoint_a.residue.chain_id != endpoint_b.residue.chain_id:
        return False
    atom_pair = {
        (endpoint_a.atom_name or "").upper(),
        (endpoint_b.atom_name or "").upper(),
    }
    if atom_pair != {"C", "N"}:
        return False
    sequence_a = _sequence_number(endpoint_a.residue)
    sequence_b = _sequence_number(endpoint_b.residue)
    return (
        sequence_a is not None
        and sequence_b is not None
        and abs(sequence_a - sequence_b) <= 1
    )


def _normalize_crosslink_kind(
    raw_kind: str,
    endpoint_a: CrosslinkEndpoint,
    endpoint_b: CrosslinkEndpoint,
) -> str:
    normalized = raw_kind.strip().lower()
    if normalized.startswith("disulf"):
        return "disulfide"
    if (
        endpoint_a.residue_name.upper() in SOLVENT_RESIDUES
        or endpoint_b.residue_name.upper() in SOLVENT_RESIDUES
    ):
        return "solvent_coordination"
    if normalized.startswith("metalc") or (
        endpoint_a.residue_name.upper() in METAL_RESIDUES
        or endpoint_b.residue_name.upper() in METAL_RESIDUES
    ):
        return "metal_coordination"
    if normalized.startswith("hydrog"):
        return "hydrogen_bond"
    if normalized.startswith("covale") or normalized in {"link", "covalent"}:
        if (
            endpoint_a.residue_name.upper() == "CYS"
            and endpoint_b.residue_name.upper() == "CYS"
            and (endpoint_a.atom_name or "").upper() == "SG"
            and (endpoint_b.atom_name or "").upper() == "SG"
        ):
            return "disulfide"
        if _is_backbone_link(endpoint_a, endpoint_b):
            return "backbone_link"
        return "covalent"
    return normalized or "other"


def _endpoint(
    *,
    chain_id: object,
    sequence_id: object,
    insertion_code: object = "",
    residue_name: object = "",
    atom_name: object | None = None,
) -> CrosslinkEndpoint:
    return CrosslinkEndpoint(
        residue=ResidueKey(
            chain_id=_clean(chain_id, "?"),
            sequence_id=_clean(sequence_id, "?"),
            insertion_code=_clean(insertion_code),
        ),
        residue_name=_clean(residue_name, "UNK").upper(),
        atom_name=_clean(atom_name).upper() or None,
    )


def _deduplicate_crosslinks(
    records: Iterable[CrosslinkRecord],
) -> list[CrosslinkRecord]:
    """Deduplicate equivalent LINK/SSBOND rows, preferring explicit SSBOND rows."""

    by_key: dict[
        tuple[str, tuple[tuple[str, str, str, str], ...]], CrosslinkRecord
    ] = {}
    for record in records:
        key = (record.kind, record.canonical_endpoint_key)
        existing = by_key.get(key)
        if existing is None or (
            existing.source_record != "SSBOND" and record.source_record == "SSBOND"
        ):
            by_key[key] = record
    return sorted(
        by_key.values(),
        key=lambda record: (
            record.kind,
            record.canonical_endpoint_key,
            record.crosslink_id,
        ),
    )


def parse_pdb_crosslinks(pdb_path: str | Path) -> list[CrosslinkRecord]:
    """Parse PDB ``SSBOND`` and ``LINK`` records."""

    records: list[CrosslinkRecord] = []
    for line_number, line in enumerate(
        Path(pdb_path).read_text().splitlines(), start=1
    ):
        record_name = line[:6].strip().upper()
        if record_name == "SSBOND":
            endpoint_a = _endpoint(
                chain_id=line[15:16],
                sequence_id=line[17:21],
                insertion_code=line[21:22],
                residue_name=line[11:14],
                atom_name="SG",
            )
            endpoint_b = _endpoint(
                chain_id=line[29:30],
                sequence_id=line[31:35],
                insertion_code=line[35:36],
                residue_name=line[25:28],
                atom_name="SG",
            )
            serial = _clean(line[7:10], str(line_number))
            records.append(
                CrosslinkRecord(
                    crosslink_id=f"ssbond:{serial}",
                    kind="disulfide",
                    endpoint_a=endpoint_a,
                    endpoint_b=endpoint_b,
                    source_record="SSBOND",
                    distance=_float_or_none(line[73:78] if len(line) >= 78 else None),
                    metadata={"line_number": line_number, "raw_record": line},
                )
            )
        elif record_name == "LINK":
            endpoint_a = _endpoint(
                chain_id=line[21:22],
                sequence_id=line[22:26],
                insertion_code=line[26:27],
                residue_name=line[17:20],
                atom_name=line[12:16],
            )
            endpoint_b = _endpoint(
                chain_id=line[51:52],
                sequence_id=line[52:56],
                insertion_code=line[56:57],
                residue_name=line[47:50],
                atom_name=line[42:46],
            )
            kind = _normalize_crosslink_kind("link", endpoint_a, endpoint_b)
            records.append(
                CrosslinkRecord(
                    crosslink_id=f"link:{line_number}",
                    kind=kind,
                    endpoint_a=endpoint_a,
                    endpoint_b=endpoint_b,
                    source_record="LINK",
                    distance=_float_or_none(line[73:78] if len(line) >= 78 else None),
                    metadata={"line_number": line_number, "raw_record": line},
                )
            )
    return _deduplicate_crosslinks(records)


def iter_mmcif_loop_rows(
    cif_path: str | Path,
    category_prefix: str,
) -> Iterator[dict[str, str]]:
    """Yield rows from mmCIF loops whose tags share *category_prefix*.

    The parser supports ordinary quoted and wrapped rows.  Semicolon-delimited
    multiline values are intentionally ignored because neither ``_atom_site``
    nor ``_struct_conn`` uses them for the fields consumed here.
    """

    lines = Path(cif_path).read_text().splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        tags: list[str] = []
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

        matching = all(tag.startswith(category_prefix) for tag in tags)
        tokens: list[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped == "#":
                index += 1
                break
            if (
                stripped == "loop_"
                or stripped.startswith("_")
                or stripped.startswith("data_")
            ):
                break
            if matching:
                if stripped.startswith(";"):
                    while index + 1 < len(lines):
                        index += 1
                        if lines[index].startswith(";"):
                            break
                else:
                    tokens.extend(shlex.split(stripped, posix=True))
                    while len(tokens) >= len(tags):
                        row_tokens = tokens[: len(tags)]
                        tokens = tokens[len(tags) :]
                        yield dict(zip(tags, row_tokens))
            index += 1


def _first(row: Mapping[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = _clean(row.get(key))
        if value:
            return value
    return default


def parse_mmcif_crosslinks(cif_path: str | Path) -> list[CrosslinkRecord]:
    """Parse normalized crosslinks from an mmCIF ``_struct_conn`` loop."""

    records: list[CrosslinkRecord] = []
    for row_number, row in enumerate(
        iter_mmcif_loop_rows(cif_path, "_struct_conn."),
        start=1,
    ):
        endpoint_a = _endpoint(
            chain_id=_first(
                row,
                "_struct_conn.ptnr1_auth_asym_id",
                "_struct_conn.ptnr1_label_asym_id",
                default="?",
            ),
            sequence_id=_first(
                row,
                "_struct_conn.ptnr1_auth_seq_id",
                "_struct_conn.ptnr1_label_seq_id",
                default="?",
            ),
            insertion_code=row.get("_struct_conn.pdbx_ptnr1_PDB_ins_code", ""),
            residue_name=_first(
                row,
                "_struct_conn.ptnr1_auth_comp_id",
                "_struct_conn.ptnr1_label_comp_id",
                default="UNK",
            ),
            atom_name=_first(
                row,
                "_struct_conn.ptnr1_auth_atom_id",
                "_struct_conn.ptnr1_label_atom_id",
            ),
        )
        endpoint_b = _endpoint(
            chain_id=_first(
                row,
                "_struct_conn.ptnr2_auth_asym_id",
                "_struct_conn.ptnr2_label_asym_id",
                default="?",
            ),
            sequence_id=_first(
                row,
                "_struct_conn.ptnr2_auth_seq_id",
                "_struct_conn.ptnr2_label_seq_id",
                default="?",
            ),
            insertion_code=row.get("_struct_conn.pdbx_ptnr2_PDB_ins_code", ""),
            residue_name=_first(
                row,
                "_struct_conn.ptnr2_auth_comp_id",
                "_struct_conn.ptnr2_label_comp_id",
                default="UNK",
            ),
            atom_name=_first(
                row,
                "_struct_conn.ptnr2_auth_atom_id",
                "_struct_conn.ptnr2_label_atom_id",
            ),
        )
        raw_kind = _first(row, "_struct_conn.conn_type_id", default="other")
        records.append(
            CrosslinkRecord(
                crosslink_id=_first(
                    row,
                    "_struct_conn.id",
                    default=f"struct_conn:{row_number}",
                ),
                kind=_normalize_crosslink_kind(raw_kind, endpoint_a, endpoint_b),
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                source_record="_struct_conn",
                distance=_float_or_none(row.get("_struct_conn.pdbx_dist_value")),
                metadata={"row_number": row_number, "conn_type_id": raw_kind},
            )
        )
    return _deduplicate_crosslinks(records)


def parse_pdb_atoms(pdb_path: str | Path, *, model_id: int = 1) -> list[dict]:
    """Parse primary-location ATOM/HETATM coordinates from one PDB model."""

    records: list[dict] = []
    current_model: int | None = None
    has_models = False
    for line_number, line in enumerate(
        Path(pdb_path).read_text().splitlines(), start=1
    ):
        if line.startswith("MODEL"):
            has_models = True
            try:
                current_model = int(line[10:14].strip() or line[5:].strip())
            except ValueError:
                current_model = None
            continue
        if line.startswith("ENDMDL"):
            current_model = None
            continue
        if has_models and current_model != model_id:
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        altloc = _clean(line[16:17])
        if altloc not in {"", "A"}:
            continue
        try:
            coord = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
        except ValueError:
            continue
        records.append(
            {
                "group": line[:6].strip(),
                "atom_name": _clean(line[12:16]).upper(),
                "residue_name": _clean(line[17:20], "UNK").upper(),
                "chain_id": _clean(line[21:22], "?"),
                "sequence_id": _clean(line[22:26], "?"),
                "insertion_code": _clean(line[26:27]),
                "model_id": model_id if has_models else 1,
                "coord": coord,
                "line_number": line_number,
            }
        )
    return records


def parse_mmcif_atoms(cif_path: str | Path, *, model_id: int = 1) -> list[dict]:
    """Parse primary-location ATOM/HETATM coordinates from mmCIF."""

    records: list[dict] = []
    for row_number, row in enumerate(
        iter_mmcif_loop_rows(cif_path, "_atom_site."),
        start=1,
    ):
        try:
            row_model = int(_first(row, "_atom_site.pdbx_PDB_model_num", default="1"))
        except ValueError:
            row_model = 1
        if row_model != model_id:
            continue
        altloc = _first(row, "_atom_site.label_alt_id")
        if altloc not in {"", "A"}:
            continue
        try:
            coord = np.asarray(
                [
                    float(row["_atom_site.Cartn_x"]),
                    float(row["_atom_site.Cartn_y"]),
                    float(row["_atom_site.Cartn_z"]),
                ],
                dtype=float,
            )
        except (KeyError, ValueError):
            continue
        records.append(
            {
                "group": _first(row, "_atom_site.group_PDB", default="ATOM"),
                "atom_name": _first(
                    row,
                    "_atom_site.auth_atom_id",
                    "_atom_site.label_atom_id",
                ).upper(),
                "residue_name": _first(
                    row,
                    "_atom_site.auth_comp_id",
                    "_atom_site.label_comp_id",
                    default="UNK",
                ).upper(),
                "chain_id": _first(
                    row,
                    "_atom_site.auth_asym_id",
                    "_atom_site.label_asym_id",
                    default="?",
                ),
                "sequence_id": _first(
                    row,
                    "_atom_site.auth_seq_id",
                    "_atom_site.label_seq_id",
                    default="?",
                ),
                "insertion_code": _first(row, "_atom_site.pdbx_PDB_ins_code"),
                "model_id": row_model,
                "coord": coord,
                "row_number": row_number,
            }
        )
    return records


def _residue_key(record: Mapping[str, object]) -> ResidueKey:
    return ResidueKey(
        chain_id=_clean(record.get("chain_id"), "?"),
        sequence_id=_clean(record.get("sequence_id"), "?"),
        insertion_code=_clean(record.get("insertion_code")),
    )


def _atom_lookup(
    atom_records: Sequence[Mapping[str, object]],
) -> dict[tuple[ResidueKey, str], np.ndarray]:
    lookup: dict[tuple[ResidueKey, str], np.ndarray] = {}
    for record in atom_records:
        atom_name = _clean(record.get("atom_name")).upper()
        if not atom_name:
            continue
        lookup.setdefault(
            (_residue_key(record), atom_name),
            np.asarray(record["coord"], dtype=float),
        )
    return lookup


def _crosslink_atom_lane_offsets(
    crosslinks: Sequence[CrosslinkRecord],
    endpoint_positions: Mapping[ResidueKey, np.ndarray],
    atom_lookup: Mapping[tuple[ResidueKey, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Separate abstract crosslink edges that reuse one physical atom stalk.

    A residue is represented by one backbone node.  If two PDB ``LINK`` rows
    reuse the same side-chain atom, naively drawing both residue-to-atom
    segments makes distinct graph edges overlap in 3-D, so no generic planar
    projection exists.  Small deterministic lanes preserve the intended
    residue-level multigraph while putting its embedding in general position.
    """

    sites: dict[
        tuple[ResidueKey, str],
        list[tuple[str, str]],
    ] = defaultdict(list)
    for record in crosslinks:
        for side, endpoint in (("a", record.endpoint_a), ("b", record.endpoint_b)):
            atom_name = (endpoint.atom_name or "").upper()
            atom_position = atom_lookup.get((endpoint.residue, atom_name))
            anchor = endpoint_positions.get(endpoint.residue)
            if atom_position is None or anchor is None:
                continue
            if np.allclose(atom_position, anchor):
                continue
            sites[(endpoint.residue, atom_name)].append((record.crosslink_id, side))

    offsets: dict[tuple[str, str], np.ndarray] = {}
    coordinate_axes = np.eye(3, dtype=float)
    for (residue, atom_name), occurrences in sorted(
        sites.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        if len(occurrences) < 2:
            continue
        anchor = np.asarray(endpoint_positions[residue], dtype=float)
        atom = np.asarray(atom_lookup[(residue, atom_name)], dtype=float)
        tangent = atom - anchor
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            continue
        tangent /= tangent_norm
        axis = min(
            coordinate_axes, key=lambda value: abs(float(np.dot(tangent, value)))
        )
        normal = np.cross(tangent, axis)
        normal /= np.linalg.norm(normal)
        spacing = min(0.1, max(0.01, 0.02 * tangent_norm))
        ordered = sorted(occurrences)
        center = 0.5 * (len(ordered) - 1)
        for lane, occurrence in enumerate(ordered):
            offsets[occurrence] = normal * (spacing * (lane - center))
    return offsets


def build_crosslinked_protein_graph(
    atom_records: Sequence[Mapping[str, object]],
    crosslinks: Sequence[CrosslinkRecord],
    *,
    pdb_id: str,
    source_format: str,
    chain_ids: Sequence[str] | None = None,
    backbone_atom: str = "CA",
) -> tuple[nx.MultiGraph, list[CrosslinkRecord], list[str]]:
    """Build an embedded graph from backbone atoms and crosslinks.

    Only crosslinks whose two residues have a selected backbone atom are added.
    Returned crosslinks therefore correspond one-to-one with graph edges whose
    ``edge_kind`` is ``"crosslink"``.
    """

    backbone_atom = backbone_atom.strip().upper()
    backbone_by_chain: dict[str, list[Mapping[str, object]]] = {}
    seen_residues: set[ResidueKey] = set()
    for record in atom_records:
        if _clean(record.get("group"), "ATOM").upper() != "ATOM":
            continue
        if _clean(record.get("atom_name")).upper() != backbone_atom:
            continue
        residue = _residue_key(record)
        if residue in seen_residues:
            continue
        seen_residues.add(residue)
        backbone_by_chain.setdefault(residue.chain_id, []).append(record)

    available_chains = tuple(sorted(backbone_by_chain))
    selected_chains = tuple(chain_ids) if chain_ids is not None else available_chains
    missing_chains = sorted(set(selected_chains) - set(available_chains))
    if missing_chains:
        raise ValueError(
            f"Selected chains are missing {backbone_atom} atoms: {missing_chains}. "
            f"Available chains: {available_chains}"
        )
    if not selected_chains:
        raise ValueError(f"No chains with {backbone_atom} atoms were found.")

    selected_crosslinks = []
    for record in crosslinks:
        endpoints = (record.endpoint_a, record.endpoint_b)
        nonmetal_endpoints = [
            endpoint
            for endpoint in endpoints
            if endpoint.residue_name.upper() not in METAL_RESIDUES
        ]
        if nonmetal_endpoints and all(
            endpoint.residue.chain_id in selected_chains
            for endpoint in nonmetal_endpoints
        ):
            selected_crosslinks.append(record)
    anchor_residues = {
        endpoint.residue
        for record in selected_crosslinks
        for endpoint in (record.endpoint_a, record.endpoint_b)
        if endpoint.residue in seen_residues
    }
    atom_lookup = _atom_lookup(atom_records)
    graph = nx.MultiGraph()
    issues: list[str] = []
    residue_positions: dict[ResidueKey, np.ndarray] = {}

    for chain_id in selected_chains:
        chain_records = backbone_by_chain[chain_id]
        chain_residues = [_residue_key(record) for record in chain_records]
        chain_index = {residue: index for index, residue in enumerate(chain_residues)}
        chain_anchors = {chain_residues[0], chain_residues[-1]}
        chain_anchors.update(
            residue for residue in anchor_residues if residue.chain_id == chain_id
        )
        ordered_anchors = sorted(
            (residue for residue in chain_anchors if residue in chain_index),
            key=chain_index.__getitem__,
        )
        for residue in ordered_anchors:
            record = chain_records[chain_index[residue]]
            position = np.asarray(record["coord"], dtype=float)
            residue_positions[residue] = position
            graph.add_node(
                residue,
                pos=position.copy(),
                node_type=(
                    "crosslink_residue"
                    if residue in anchor_residues
                    else "chain_endpoint"
                ),
                chain_id=residue.chain_id,
                sequence_id=residue.sequence_id,
                insertion_code=residue.insertion_code,
                residue_name=_clean(record.get("residue_name"), "UNK"),
                backbone_atom=backbone_atom,
            )
        for segment_index, (start, end) in enumerate(
            zip(ordered_anchors, ordered_anchors[1:])
        ):
            start_index = chain_index[start]
            end_index = chain_index[end]
            segment_records = chain_records[start_index : end_index + 1]
            points = np.asarray(
                [record["coord"] for record in segment_records], dtype=float
            )
            graph.add_edge(
                start,
                end,
                key=f"backbone:{chain_id}:{segment_index}",
                pts=points,
                edge_kind="backbone",
                chain_id=chain_id,
                start_residue=start.label,
                end_residue=end.label,
                residue_keys=[
                    asdict(_residue_key(record)) for record in segment_records
                ],
            )

    endpoint_positions = dict(residue_positions)
    for record in selected_crosslinks:
        for endpoint in (record.endpoint_a, record.endpoint_b):
            residue = endpoint.residue
            if residue in endpoint_positions:
                continue
            atom_name = endpoint.atom_name or ""
            atom_position = atom_lookup.get((residue, atom_name))
            if (
                atom_position is None
                or endpoint.residue_name.upper() not in METAL_RESIDUES
            ):
                continue
            endpoint_positions[residue] = atom_position
            graph.add_node(
                residue,
                pos=atom_position.copy(),
                node_type="metal_center",
                chain_id=residue.chain_id,
                sequence_id=residue.sequence_id,
                insertion_code=residue.insertion_code,
                residue_name=endpoint.residue_name,
                backbone_atom=None,
            )

    atom_lane_offsets = _crosslink_atom_lane_offsets(
        selected_crosslinks,
        endpoint_positions,
        atom_lookup,
    )

    missing_endpoints: set[ResidueKey] = set()
    included_crosslinks: list[CrosslinkRecord] = []
    for record in selected_crosslinks:
        residue_a = record.endpoint_a.residue
        residue_b = record.endpoint_b.residue
        if residue_a not in endpoint_positions or residue_b not in endpoint_positions:
            for residue in (residue_a, residue_b):
                if (
                    residue not in endpoint_positions
                    and residue not in missing_endpoints
                ):
                    missing_endpoints.add(residue)
                    issues.append(
                        f"crosslink endpoint {residue.label} has no representable atom"
                    )
            continue
        start = endpoint_positions[residue_a]
        end = endpoint_positions[residue_b]
        points = [start]
        atom_a = record.endpoint_a.atom_name
        atom_b = record.endpoint_b.atom_name
        if atom_a and (residue_a, atom_a) in atom_lookup:
            points.append(
                atom_lookup[(residue_a, atom_a)]
                + atom_lane_offsets.get((record.crosslink_id, "a"), 0.0)
            )
        if atom_b and (residue_b, atom_b) in atom_lookup:
            points.append(
                atom_lookup[(residue_b, atom_b)]
                + atom_lane_offsets.get((record.crosslink_id, "b"), 0.0)
            )
        points.append(end)
        cleaned_points = [np.asarray(points[0], dtype=float)]
        for point in points[1:]:
            if not np.allclose(point, cleaned_points[-1]):
                cleaned_points.append(np.asarray(point, dtype=float))
        if len(cleaned_points) < 2:
            issues.append(f"crosslink {record.crosslink_id} collapses to one point")
            continue
        graph.add_edge(
            residue_a,
            residue_b,
            key=f"crosslink:{record.crosslink_id}",
            pts=np.asarray(cleaned_points, dtype=float),
            edge_kind="crosslink",
            crosslink_id=record.crosslink_id,
            crosslink_type=record.kind,
            source_record=record.source_record,
            distance=record.distance,
            endpoint_a=asdict(record.endpoint_a),
            endpoint_b=asdict(record.endpoint_b),
            general_position_offsets={
                side: offset.tolist()
                for side in ("a", "b")
                if (offset := atom_lane_offsets.get((record.crosslink_id, side)))
                is not None
            },
        )
        included_crosslinks.append(record)

    graph.graph.update(
        {
            "input_kind": "crosslinked_protein",
            "input_id": f"{pdb_id}_{'-'.join(selected_chains)}_crosslinked",
            "pdb_id": pdb_id,
            "source_format": source_format,
            "model_id": (
                int(str(atom_records[0].get("model_id", 1))) if atom_records else 1
            ),
            "chain_ids": selected_chains,
            "backbone_atom": backbone_atom,
            "crosslink_count": len(included_crosslinks),
            "crosslink_ids": tuple(
                record.crosslink_id for record in included_crosslinks
            ),
            "general_position_offset_count": len(atom_lane_offsets),
            "general_position_max_offset": max(
                (
                    float(np.linalg.norm(offset))
                    for offset in atom_lane_offsets.values()
                ),
                default=0.0,
            ),
        }
    )
    issues.extend(validate_embedding(graph))
    return graph, included_crosslinks, issues


def load_crosslinked_protein(
    source: str | Path,
    *,
    source_format: str | None = None,
    pdb_id: str | None = None,
    chain_ids: Sequence[str] | None = None,
    model_id: int = 1,
    backbone_atom: str = "CA",
    allowed_crosslink_types: Iterable[str] | None = DEFAULT_CROSSLINK_TYPES,
    crosslink_ids: Iterable[str] | None = None,
    data_dir: str | Path | None = None,
    download: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> CrosslinkedProteinInputResult:
    """Load a crosslinked protein from a PDB ID or local PDB/mmCIF file."""

    if model_id < 1:
        raise ValueError("model_id must be a positive integer")
    source_text = str(source)
    if source_format is None:
        suffix = Path(source_text).suffix.lower()
        source_format = "mmcif" if suffix in {".cif", ".mmcif"} else "pdb"
    source_format = source_format.lower().lstrip(".")

    if source_format == "pdb":
        resolved_id, source_path, source_url, downloaded = _resolve_pdb_source(
            source,
            pdb_id=pdb_id,
            data_dir=data_dir,
            download=download,
        )
        atom_records = parse_pdb_atoms(source_path, model_id=model_id)
        parsed_crosslinks = parse_pdb_crosslinks(source_path)
    elif source_format in {"cif", "mmcif"}:
        resolved_id, source_path, source_url, downloaded = _resolve_mmcif_source(
            source,
            pdb_id=pdb_id,
            data_dir=data_dir,
            download=download,
        )
        source_format = "mmcif"
        atom_records = parse_mmcif_atoms(source_path, model_id=model_id)
        parsed_crosslinks = parse_mmcif_crosslinks(source_path)
    else:
        raise ValueError("source_format must be 'pdb' or 'mmcif'")

    allowed = (
        None
        if allowed_crosslink_types is None
        else {value.strip().lower() for value in allowed_crosslink_types}
    )
    requested_ids = (
        None
        if crosslink_ids is None
        else {str(value).strip() for value in crosslink_ids}
    )
    if requested_ids is not None:
        available_ids = {record.crosslink_id for record in parsed_crosslinks}
        unknown_ids = sorted(requested_ids - available_ids)
        if unknown_ids:
            raise ValueError(f"Unknown crosslink IDs for {resolved_id}: {unknown_ids}")
    selected_by_type = [
        record
        for record in parsed_crosslinks
        if (allowed is None or record.kind in allowed)
        and (requested_ids is None or record.crosslink_id in requested_ids)
    ]
    excluded = [
        record for record in parsed_crosslinks if record not in selected_by_type
    ]
    graph, included, issues = build_crosslinked_protein_graph(
        atom_records,
        selected_by_type,
        pdb_id=resolved_id,
        source_format=source_format,
        chain_ids=chain_ids,
        backbone_atom=backbone_atom,
    )
    included_ids = {record.crosslink_id for record in included}
    excluded.extend(
        record for record in selected_by_type if record.crosslink_id not in included_ids
    )
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "pdb_id": resolved_id,
            "source_format": source_format,
            "source_path": str(source_path),
            "model_id": model_id,
            "chain_ids": graph.graph["chain_ids"],
            "backbone_atom": backbone_atom.upper(),
            "allowed_crosslink_types": tuple(sorted(allowed))
            if allowed is not None
            else None,
            "requested_crosslink_ids": (
                tuple(sorted(requested_ids)) if requested_ids is not None else None
            ),
        }
    )
    graph.graph.update(result_metadata)
    return CrosslinkedProteinInputResult(
        pdb_id=resolved_id,
        source_format=source_format,
        source_path=Path(source_path),
        source_url=source_url,
        downloaded=downloaded,
        model_id=model_id,
        chain_ids=tuple(graph.graph["chain_ids"]),
        backbone_atom=backbone_atom.upper(),
        allowed_crosslink_types=(
            tuple(sorted(allowed)) if allowed is not None else None
        ),
        crosslinks=included,
        excluded_crosslinks=_deduplicate_crosslinks(excluded),
        atom_records=atom_records,
        graph=graph,
        issues=issues,
        metadata=result_metadata,
    )


__all__ = [
    "CrosslinkEndpoint",
    "CrosslinkRecord",
    "CrosslinkedProteinInputResult",
    "DEFAULT_CROSSLINK_TYPES",
    "ResidueKey",
    "build_crosslinked_protein_graph",
    "iter_mmcif_loop_rows",
    "load_crosslinked_protein",
    "parse_mmcif_atoms",
    "parse_mmcif_crosslinks",
    "parse_pdb_atoms",
    "parse_pdb_crosslinks",
]
