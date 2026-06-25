"""Coordinate-chain input adapter.

This module converts ordered 3D coordinate data into the ``MultiGraph(pos/pts)``
convention used by KnottedGraph examples and downstream spatial-graph code.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from knotted_graph.core.embedding import validate_embedding


SUPPORTED_COORDINATE_SUFFIXES = {".csv", ".dat", ".json", ".npy", ".tsv", ".txt", ".xyz"}
SUPPORTED_CLOSURES = {None, "direct", "metadata_only"}


@dataclass
class CoordinateInputResult:
    """Parsed coordinate-chain input and its graph representation."""

    input_id: str
    source_path: Path | None
    source_format: str
    coords: np.ndarray
    graph: nx.MultiGraph
    closed: bool
    closure_method: str | None
    metadata: dict
    issues: list[str]


def validate_coords(coords, *, min_points: int = 2) -> np.ndarray:
    """Return a finite float array with shape ``(N, 3)``."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape (N, 3), got {arr.shape}.")
    if arr.shape[0] < min_points:
        raise ValueError(f"Need at least {min_points} points, got {arr.shape[0]}.")
    if not np.isfinite(arr).all():
        raise ValueError("Coordinates contain NaN or infinite values.")
    return arr


def _coords_are_closed(coords: np.ndarray) -> bool:
    return bool(np.allclose(coords[0], coords[-1]))


def _close_coords_direct(coords: np.ndarray) -> np.ndarray:
    if _coords_are_closed(coords):
        return coords
    return np.vstack([coords, coords[0]])


def _validate_closure(closure: str | None) -> None:
    if closure not in SUPPORTED_CLOSURES:
        raise ValueError(f"Unsupported closure {closure!r}. Supported: {sorted(c for c in SUPPORTED_CLOSURES if c)} plus None.")


def _normalize_path(source) -> Path:
    return source if isinstance(source, Path) else Path(source)


def _infer_source_format(path: Path, source_format: str | None) -> str:
    if source_format:
        return source_format.lower().lstrip(".")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_COORDINATE_SUFFIXES:
        raise ValueError(
            f"Unsupported coordinate-chain suffix {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_COORDINATE_SUFFIXES)}"
        )
    return suffix.lstrip(".")


def _resolve_graph_coords(
    coords: np.ndarray,
    *,
    closed: bool,
    closure: str | None,
) -> tuple[np.ndarray, bool]:
    """Return coordinates for the graph plus whether the graph is closed."""
    _validate_closure(closure)
    coords = validate_coords(coords)

    if closure == "direct":
        return _close_coords_direct(coords), True

    if closed and _coords_are_closed(coords):
        return coords, True

    if closed and closure == "metadata_only":
        return coords, False

    if closed:
        raise ValueError(
            "Input was marked closed but the first and last points differ. "
            "Pass closure='direct' to add a closing segment, or "
            "closure='metadata_only' to preserve coordinates and only record the intent."
        )

    return coords, False


def coordinates_to_multigraph(
    coords,
    *,
    closed: bool = False,
    closure: str | None = None,
    input_id: str = "coordinate_curve",
    source_format: str = "array",
    source_path: Path | None = None,
    metadata: dict | None = None,
) -> nx.MultiGraph:
    """Convert an ordered coordinate chain to a ``networkx.MultiGraph``.

    Open curves are represented by two nodes, ``start`` and ``end``. Closed
    curves are represented by a single self-loop edge attached to
    ``loop_anchor``. Direct closure is explicit: pass ``closure="direct"`` to
    add the segment from the last point back to the first point.
    """
    graph_coords, graph_closed = _resolve_graph_coords(coords, closed=closed, closure=closure)
    meta = dict(metadata or {})

    graph = nx.MultiGraph()
    graph.graph.update(meta)
    graph.graph["input_kind"] = "coordinate_curve"
    graph.graph["input_id"] = input_id
    graph.graph["curve_id"] = input_id
    graph.graph["source_format"] = source_format
    graph.graph["source_path"] = str(source_path) if source_path else None
    graph.graph["is_closed"] = bool(closed or graph_closed)
    graph.graph["closure_method"] = closure
    graph.graph["graph_is_closed"] = graph_closed

    if graph_closed:
        graph.add_node("loop_anchor", pos=graph_coords[0].copy(), node_type="closure_anchor")
        graph.add_edge(
            "loop_anchor",
            "loop_anchor",
            key="curve",
            pts=graph_coords.copy(),
            input_id=input_id,
            closed=True,
            closure_method=closure,
        )
    else:
        graph.add_node("start", pos=graph_coords[0].copy(), node_type="endpoint")
        graph.add_node("end", pos=graph_coords[-1].copy(), node_type="endpoint")
        graph.add_edge(
            "start",
            "end",
            key="curve",
            pts=graph_coords.copy(),
            input_id=input_id,
            closed=bool(closed),
            closure_method=closure,
        )

    return graph


def validate_curve_graph(graph: nx.MultiGraph) -> list[str]:
    """Return schema issues for a coordinate-curve graph."""
    if not isinstance(graph, nx.MultiGraph):
        return validate_embedding(graph)

    issues = validate_embedding(graph)
    if graph.number_of_edges() != 1:
        issues.append(f"expected 1 edge, got {graph.number_of_edges()}")
    return issues


def _load_npy_coords(path: Path) -> np.ndarray:
    return validate_coords(np.load(path))


def _load_csv_coords(
    path: Path,
    *,
    columns: tuple[str, str, str],
    delimiter: str = ",",
) -> np.ndarray:
    coords = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV file is missing coordinate columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                coords.append([float(row[column]) for column in columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {row_number}: invalid coordinate value") from exc
    return validate_coords(coords)


def _load_json_coords(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "points" in data:
            data = data["points"]
        elif "coords" in data:
            data = data["coords"]
        else:
            raise ValueError("JSON coordinate file must contain 'points' or 'coords'.")
    return validate_coords(data)


def _load_table_coords(
    path: Path,
    *,
    delimiter: str | None,
    comment_prefix: str = "#",
) -> np.ndarray:
    coords = []
    skipped_header = False
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(comment_prefix):
            continue
        tokens = [token for token in line.split(delimiter) if token]
        if len(tokens) < 3:
            raise ValueError(f"line {line_number}: expected at least 3 columns")
        try:
            coords.append([float(value) for value in tokens[:3]])
        except ValueError as exc:
            if not coords and not skipped_header:
                skipped_header = True
                continue
            raise ValueError(f"line {line_number}: invalid coordinate value") from exc
    return validate_coords(coords)


def _parse_xyz_coord_tokens(tokens: list[str], line_number: int) -> tuple[float, float, float]:
    if len(tokens) < 3:
        raise ValueError(f"line {line_number}: expected at least 3 tokens")

    start = 0
    try:
        float(tokens[0])
    except ValueError:
        start = 1

    if len(tokens) < start + 3:
        raise ValueError(f"line {line_number}: expected x y z coordinates")

    try:
        return tuple(float(value) for value in tokens[start : start + 3])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid XYZ coordinate value") from exc


def _load_xyz_coords(path: Path) -> np.ndarray:
    raw_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"XYZ file is empty: {path}")

    data_lines = raw_lines
    try:
        atom_count = int(raw_lines[0])
    except ValueError:
        atom_count = None
    if atom_count is not None:
        data_lines = raw_lines[2:]
        if len(data_lines) != atom_count:
            raise ValueError(
                f"XYZ atom count says {atom_count}, but found {len(data_lines)} rows."
            )

    return validate_coords(
        [
            _parse_xyz_coord_tokens(line.split(), line_number)
            for line_number, line in enumerate(data_lines, start=1)
        ]
    )


def _load_coords_from_path(
    path: Path,
    *,
    source_format: str,
    columns: tuple[str, str, str],
    delimiter: str | None,
) -> np.ndarray:
    if source_format == "npy":
        return _load_npy_coords(path)
    if source_format == "json":
        return _load_json_coords(path)
    if source_format == "xyz":
        return _load_xyz_coords(path)
    if source_format == "csv":
        return _load_csv_coords(path, columns=columns, delimiter=delimiter or ",")
    if source_format == "tsv":
        return _load_table_coords(path, delimiter=delimiter or "\t")
    if source_format in {"dat", "txt"}:
        return _load_table_coords(path, delimiter=delimiter)
    raise ValueError(f"Unsupported coordinate-chain source format {source_format!r}.")


def _edge_points(graph: nx.MultiGraph) -> np.ndarray:
    for _, _, data in graph.edges(data=True):
        return np.asarray(data["pts"], dtype=float)
    raise ValueError("coordinate graph has no edge")


def from_coordinate_chain(
    source,
    *,
    input_id: str | None = None,
    source_format: str | None = None,
    columns: tuple[str, str, str] = ("x", "y", "z"),
    delimiter: str | None = None,
    closed: bool = False,
    closure: str | None = None,
    metadata: dict | None = None,
) -> CoordinateInputResult:
    """Load an ordered 3D coordinate chain.

    ``source`` may be an ``(N, 3)`` array-like object or a path to one of the
    supported lightweight coordinate formats. The returned graph follows the
    package's ``MultiGraph(pos/pts)`` convention.
    """
    meta = dict(metadata or {})

    if isinstance(source, (str, Path)):
        path = _normalize_path(source)
        fmt = _infer_source_format(path, source_format)
        coords = _load_coords_from_path(path, source_format=fmt, columns=columns, delimiter=delimiter)
        resolved_input_id = input_id or path.stem
        source_path = path
    else:
        fmt = (source_format or "array").lower().lstrip(".")
        coords = validate_coords(source)
        resolved_input_id = input_id or "coordinate_curve"
        source_path = None

    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        closure=closure,
        input_id=resolved_input_id,
        source_format=fmt,
        source_path=source_path,
        metadata=meta,
    )
    issues = validate_curve_graph(graph)

    return CoordinateInputResult(
        input_id=resolved_input_id,
        source_path=source_path,
        source_format=fmt,
        coords=coords.copy(),
        graph=graph,
        closed=bool(closed or graph.graph.get("graph_is_closed")),
        closure_method=closure,
        metadata=meta,
        issues=issues,
    )


def write_xyz_coords(
    coords,
    path,
    *,
    labels: Iterable[str] | None = None,
    comment: str = "coordinate curve",
) -> None:
    """Write coordinates to a small XYZ file."""
    arr = validate_coords(coords)
    path = _normalize_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = ["C"] * arr.shape[0]
    labels = list(labels)
    if len(labels) != arr.shape[0]:
        raise ValueError("labels length must match coordinate count.")

    lines = [str(arr.shape[0]), comment]
    for label, (x, y, z) in zip(labels, arr):
        lines.append(f"{label} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "CoordinateInputResult",
    "coordinates_to_multigraph",
    "from_coordinate_chain",
    "validate_coords",
    "validate_curve_graph",
    "write_xyz_coords",
]
