"""Generic coordinate-curve adapter for Task 2 prototypes.

The adapter converts ordered 3D coordinates from common lightweight formats
into the same ``networkx.MultiGraph`` convention used by the rest of the
examples: node ``pos`` attributes and edge ``pts`` arrays.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


@dataclass
class CoordinateCurveResult:
    """Container for a coordinate curve and its graph representation."""

    curve_id: str
    source_path: Path | None
    source_format: str
    closed: bool
    coords: np.ndarray
    graph: nx.MultiGraph
    issues: list[str]


def validate_coords(coords: np.ndarray, min_points: int = 2) -> np.ndarray:
    """Return a clean float ``(N, 3)`` coordinate array."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected coords with shape (N, 3), got {arr.shape}.")
    if arr.shape[0] < min_points:
        raise ValueError(f"Need at least {min_points} points, got {arr.shape[0]}.")
    if not np.isfinite(arr).all():
        raise ValueError("Coordinates contain NaN or infinite values.")
    return arr


def close_coords_if_needed(coords: np.ndarray) -> np.ndarray:
    """Append the first point when a closed curve is not explicitly closed."""
    coords = validate_coords(coords)
    if np.allclose(coords[0], coords[-1]):
        return coords
    return np.vstack([coords, coords[0]])


def coordinates_to_multigraph(
    coords: np.ndarray,
    *,
    closed: bool = False,
    curve_id: str = "coordinate_curve",
    source_format: str = "array",
    source_path: Path | None = None,
    metadata: dict | None = None,
) -> nx.MultiGraph:
    """Convert one ordered coordinate chain into a ``MultiGraph(pos/pts)``."""
    pts = validate_coords(coords)
    if closed:
        pts = close_coords_if_needed(pts)

    graph = nx.MultiGraph()
    graph.graph["input_kind"] = "coordinate_curve"
    graph.graph["curve_id"] = curve_id
    graph.graph["source_format"] = source_format
    graph.graph["source_path"] = str(source_path) if source_path else None
    graph.graph["is_closed"] = closed
    if metadata:
        graph.graph.update(metadata)

    if closed:
        graph.add_node("loop_anchor", pos=pts[0].copy())
        graph.add_edge(
            "loop_anchor",
            "loop_anchor",
            key="curve",
            pts=pts.copy(),
            curve_id=curve_id,
            closed=True,
        )
    else:
        graph.add_node("start", pos=pts[0].copy())
        graph.add_node("end", pos=pts[-1].copy())
        graph.add_edge(
            "start",
            "end",
            key="curve",
            pts=pts.copy(),
            curve_id=curve_id,
            closed=False,
        )

    return graph


def validate_curve_graph(graph: nx.MultiGraph, expected_points: int) -> list[str]:
    """Return schema issues for the coordinate-curve graph, if any."""
    issues = []
    if not isinstance(graph, nx.MultiGraph):
        return ["graph is not a networkx.MultiGraph"]

    for node, data in graph.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            issues.append(f"node {node!r} is missing 'pos'")
            continue
        pos = np.asarray(pos)
        if pos.shape != (3,):
            issues.append(f"node {node!r} has pos shape {pos.shape}, expected (3,)")

    edge_count = 0
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_count += 1
        pts = data.get("pts")
        if pts is None:
            issues.append(f"edge {(u, v, key)!r} is missing 'pts'")
            continue
        pts = np.asarray(pts)
        if pts.shape != (expected_points, 3):
            issues.append(
                f"edge {(u, v, key)!r} has pts shape {pts.shape}, "
                f"expected ({expected_points}, 3)"
            )
    if edge_count != 1:
        issues.append(f"expected 1 edge, got {edge_count}")
    return issues


def load_npy_coords(path: Path) -> np.ndarray:
    return validate_coords(np.load(path))


def save_npy_coords(coords: np.ndarray, path: Path) -> tuple[bool, tuple[int, ...]]:
    coords = validate_coords(coords)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords)
    loaded = np.load(path)
    return path.exists() and np.array_equal(loaded, coords), loaded.shape


def load_csv_coords(
    path: Path,
    *,
    columns: tuple[str, str, str] = ("x", "y", "z"),
    delimiter: str = ",",
) -> np.ndarray:
    """Load coordinates from a CSV file with named x/y/z columns."""
    coords = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV file is missing coordinate columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                coords.append(tuple(float(row[column]) for column in columns))
            except ValueError as exc:
                raise ValueError(f"line {row_number}: could not parse CSV coords") from exc
    return validate_coords(np.asarray(coords, dtype=float))


def write_csv_coords(
    coords: np.ndarray,
    path: Path,
    *,
    columns: tuple[str, str, str] = ("x", "y", "z"),
) -> None:
    coords = validate_coords(coords)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(coords)


def load_json_coords(path: Path) -> np.ndarray:
    """Load coordinates from a small JSON coordinate-curve file.

    Accepted shapes:

    - ``[[x, y, z], ...]``
    - ``{"points": [[x, y, z], ...]}``
    - ``{"coords": [[x, y, z], ...]}``
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "points" in data:
            data = data["points"]
        elif "coords" in data:
            data = data["coords"]
        else:
            raise ValueError("JSON coordinate file must contain 'points' or 'coords'.")
    return validate_coords(np.asarray(data, dtype=float))


def write_json_coords(
    coords: np.ndarray,
    path: Path,
    *,
    closed: bool = False,
    curve_id: str = "coordinate_curve",
) -> None:
    coords = validate_coords(coords)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "curve_id": curve_id,
        "closed": closed,
        "points": coords.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_table_coords(
    path: Path,
    *,
    delimiter: str | None = None,
    comment_prefix: str = "#",
) -> np.ndarray:
    """Load bare table rows containing x y z coordinates.

    The default ``delimiter=None`` accepts arbitrary whitespace, so it works for
    ``.dat`` and simple TSV-like files. If the first non-comment row is a header,
    it is skipped when it contains non-numeric tokens.
    """
    coords = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(comment_prefix):
            continue
        tokens = line.split(delimiter)
        tokens = [token for token in tokens if token]
        if len(tokens) < 3:
            raise ValueError(f"line {line_number}: expected at least 3 columns")
        try:
            coords.append(tuple(float(value) for value in tokens[:3]))
        except ValueError:
            if not coords:
                continue
            raise ValueError(f"line {line_number}: could not parse table coords")
    return validate_coords(np.asarray(coords, dtype=float))


def write_table_coords(
    coords: np.ndarray,
    path: Path,
    *,
    delimiter: str = "\t",
    header: str = "x\ty\tz",
) -> None:
    coords = validate_coords(coords)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header]
    for x, y, z in coords:
        lines.append(delimiter.join([f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"]))
    path.write_text("\n".join(lines) + "\n")


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
        raise ValueError(f"line {line_number}: could not parse XYZ coords") from exc


def load_xyz_coords(path: Path) -> np.ndarray:
    """Load coordinates from an XYZ-style file.

    Both standard molecular XYZ files and bare whitespace-separated ``x y z``
    rows are accepted. When an element/name column is present, it is ignored.
    """
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

    coords = [
        _parse_xyz_coord_tokens(line.split(), line_number)
        for line_number, line in enumerate(data_lines, start=1)
    ]
    return validate_coords(np.asarray(coords, dtype=float))


def write_xyz_coords(
    coords: np.ndarray,
    path: Path,
    *,
    labels: Iterable[str] | None = None,
    comment: str = "coordinate curve",
) -> None:
    coords = validate_coords(coords)
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = ["C"] * coords.shape[0]
    labels = list(labels)
    if len(labels) != coords.shape[0]:
        raise ValueError("labels length must match coordinate count.")

    lines = [str(coords.shape[0]), comment]
    for label, (x, y, z) in zip(labels, coords):
        lines.append(f"{label} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n")


def build_curve_from_array(
    coords: np.ndarray,
    *,
    closed: bool = False,
    curve_id: str = "coordinate_curve",
    metadata: dict | None = None,
) -> CoordinateCurveResult:
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        curve_id=curve_id,
        source_format="array",
        metadata=metadata,
    )
    expected_points = np.asarray(next(iter(graph.edges(data=True)))[2]["pts"]).shape[0]
    issues = validate_curve_graph(graph, expected_points=expected_points)
    return CoordinateCurveResult(
        curve_id=curve_id,
        source_path=None,
        source_format="array",
        closed=closed,
        coords=validate_coords(coords),
        graph=graph,
        issues=issues,
    )


def build_curve_from_csv(
    path: Path,
    *,
    closed: bool = False,
    curve_id: str | None = None,
    columns: tuple[str, str, str] = ("x", "y", "z"),
) -> CoordinateCurveResult:
    coords = load_csv_coords(path, columns=columns)
    curve_id = curve_id or path.stem
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        curve_id=curve_id,
        source_format="csv",
        source_path=path,
    )
    expected_points = np.asarray(next(iter(graph.edges(data=True)))[2]["pts"]).shape[0]
    issues = validate_curve_graph(graph, expected_points=expected_points)
    return CoordinateCurveResult(curve_id, path, "csv", closed, coords, graph, issues)


def build_curve_from_json(
    path: Path,
    *,
    closed: bool = False,
    curve_id: str | None = None,
) -> CoordinateCurveResult:
    coords = load_json_coords(path)
    curve_id = curve_id or path.stem
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        curve_id=curve_id,
        source_format="json",
        source_path=path,
    )
    expected_points = np.asarray(next(iter(graph.edges(data=True)))[2]["pts"]).shape[0]
    issues = validate_curve_graph(graph, expected_points=expected_points)
    return CoordinateCurveResult(curve_id, path, "json", closed, coords, graph, issues)


def build_curve_from_table(
    path: Path,
    *,
    closed: bool = False,
    curve_id: str | None = None,
    source_format: str | None = None,
    delimiter: str | None = None,
) -> CoordinateCurveResult:
    coords = load_table_coords(path, delimiter=delimiter)
    curve_id = curve_id or path.stem
    source_format = source_format or path.suffix.lstrip(".") or "table"
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        curve_id=curve_id,
        source_format=source_format,
        source_path=path,
    )
    expected_points = np.asarray(next(iter(graph.edges(data=True)))[2]["pts"]).shape[0]
    issues = validate_curve_graph(graph, expected_points=expected_points)
    return CoordinateCurveResult(curve_id, path, source_format, closed, coords, graph, issues)


def build_curve_from_npy(
    path: Path,
    *,
    closed: bool = False,
    curve_id: str | None = None,
) -> CoordinateCurveResult:
    coords = load_npy_coords(path)
    curve_id = curve_id or path.stem
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        curve_id=curve_id,
        source_format="npy",
        source_path=path,
    )
    expected_points = np.asarray(next(iter(graph.edges(data=True)))[2]["pts"]).shape[0]
    issues = validate_curve_graph(graph, expected_points=expected_points)
    return CoordinateCurveResult(curve_id, path, "npy", closed, coords, graph, issues)


def build_curve_from_xyz(
    path: Path,
    *,
    closed: bool = False,
    curve_id: str | None = None,
) -> CoordinateCurveResult:
    coords = load_xyz_coords(path)
    curve_id = curve_id or path.stem
    graph = coordinates_to_multigraph(
        coords,
        closed=closed,
        curve_id=curve_id,
        source_format="xyz",
        source_path=path,
    )
    expected_points = np.asarray(next(iter(graph.edges(data=True)))[2]["pts"]).shape[0]
    issues = validate_curve_graph(graph, expected_points=expected_points)
    return CoordinateCurveResult(curve_id, path, "xyz", closed, coords, graph, issues)
