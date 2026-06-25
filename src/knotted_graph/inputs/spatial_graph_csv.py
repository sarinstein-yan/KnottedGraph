"""Spatial graph CSV input adapter.

The adapter reads a pair of node/edge CSV files and returns an embedded
``networkx.MultiGraph`` using node ``pos`` attributes and edge ``pts`` arrays.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from knotted_graph.core.embedding import as_polyline, validate_embedding


@dataclass
class SpatialGraphInputResult:
    """Parsed spatial graph and validation details."""

    graph_id: str
    nodes_path: Path
    edges_path: Path
    graph: nx.MultiGraph
    metadata: dict
    issues: list[str]


def _as_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _fieldnames(reader: csv.DictReader, *, label: str) -> set[str]:
    names = reader.fieldnames
    if not names:
        raise ValueError(f"{label} CSV is missing a header row.")
    return set(names)


def _resolve_column(
    fieldnames: set[str],
    requested: str,
    *,
    aliases: tuple[str, ...] = (),
    required: bool = True,
    label: str,
) -> str | None:
    if requested in fieldnames:
        return requested
    for alias in aliases:
        if alias in fieldnames:
            return alias
    if required:
        choices = (requested, *aliases)
        raise ValueError(f"{label} CSV is missing required column; expected one of {choices}.")
    return None


def _strip(value, *, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} is missing.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} is empty.")
    return stripped


def _parse_float(value: str | None, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _optional_attrs(row: dict[str, str], reserved: set[str]) -> dict:
    attrs = {}
    for key, value in row.items():
        if key in reserved or value is None:
            continue
        attrs[key] = value
    return attrs


def _parse_points_json(
    value: str | None,
    *,
    edge_id: str,
    source_pos: np.ndarray,
    target_pos: np.ndarray,
) -> np.ndarray | None:
    if value is None or value.strip() == "":
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"edge {edge_id!r} has invalid points_json.") from exc
    pts = as_polyline(payload, label=f"edge {edge_id!r} points_json")
    if not np.allclose(pts[0], source_pos):
        raise ValueError(f"edge {edge_id!r} points_json first point does not match source position.")
    if not np.allclose(pts[-1], target_pos):
        raise ValueError(f"edge {edge_id!r} points_json last point does not match target position.")
    return pts


def validate_spatial_graph(graph: nx.MultiGraph) -> list[str]:
    """Return validation issues for a ``MultiGraph(pos/pts)`` object."""
    return validate_embedding(graph)


def from_spatial_graph_csv(
    nodes_csv,
    edges_csv,
    *,
    graph_id: str | None = None,
    node_id_col: str = "node_id",
    edge_id_col: str = "edge_id",
    source_col: str = "source",
    target_col: str = "target",
    coord_cols: tuple[str, str, str] = ("x", "y", "z"),
    points_col: str | None = "points_json",
    metadata: dict | None = None,
) -> SpatialGraphInputResult:
    """Load a 3D embedded graph from node and edge CSV files.

    Required node columns are an ID column plus x/y/z coordinates. Required
    edge columns are source and target. Optional columns such as ``label`` and
    ``type`` are preserved as node or edge attributes.
    """
    nodes_path = _as_path(nodes_csv)
    edges_path = _as_path(edges_csv)
    meta = dict(metadata or {})
    resolved_graph_id = graph_id or edges_path.stem

    graph = nx.MultiGraph()
    graph.graph.update(meta)
    graph.graph["input_kind"] = "spatial_graph_csv"
    graph.graph["graph_id"] = resolved_graph_id
    graph.graph["source_format"] = "csv"
    graph.graph["nodes_path"] = str(nodes_path)
    graph.graph["edges_path"] = str(edges_path)

    with nodes_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = _fieldnames(reader, label="node")
        resolved_node_id_col = _resolve_column(
            fields,
            node_id_col,
            aliases=("id",),
            label="node",
        )
        missing_coords = [column for column in coord_cols if column not in fields]
        if missing_coords:
            raise ValueError(f"node CSV is missing coordinate columns: {missing_coords}")
        reserved = {resolved_node_id_col, *coord_cols}

        for row_number, row in enumerate(reader, start=2):
            node_id = _strip(row.get(resolved_node_id_col), label=f"node CSV line {row_number} id")
            if node_id in graph:
                raise ValueError(f"node CSV line {row_number}: duplicate node ID {node_id!r}.")
            pos = np.array(
                [
                    _parse_float(row.get(coord_cols[0]), label=f"node {node_id!r} x"),
                    _parse_float(row.get(coord_cols[1]), label=f"node {node_id!r} y"),
                    _parse_float(row.get(coord_cols[2]), label=f"node {node_id!r} z"),
                ],
                dtype=float,
            )
            attrs = _optional_attrs(row, reserved)
            graph.add_node(node_id, pos=pos, **attrs)

    seen_edge_ids: set[str] = set()
    with edges_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = _fieldnames(reader, label="edge")
        resolved_source_col = _resolve_column(fields, source_col, label="edge")
        resolved_target_col = _resolve_column(fields, target_col, label="edge")
        resolved_edge_id_col = _resolve_column(
            fields,
            edge_id_col,
            aliases=("id", "key"),
            required=False,
            label="edge",
        )
        reserved = {resolved_source_col, resolved_target_col}
        if resolved_edge_id_col:
            reserved.add(resolved_edge_id_col)
        if points_col:
            reserved.add(points_col)

        for edge_index, row in enumerate(reader):
            row_number = edge_index + 2
            source = _strip(row.get(resolved_source_col), label=f"edge CSV line {row_number} source")
            target = _strip(row.get(resolved_target_col), label=f"edge CSV line {row_number} target")
            if source not in graph:
                raise ValueError(f"edge CSV line {row_number}: unknown source {source!r}.")
            if target not in graph:
                raise ValueError(f"edge CSV line {row_number}: unknown target {target!r}.")

            if resolved_edge_id_col:
                edge_id = _strip(row.get(resolved_edge_id_col), label=f"edge CSV line {row_number} id")
                if edge_id in seen_edge_ids:
                    raise ValueError(f"edge CSV line {row_number}: duplicate edge ID {edge_id!r}.")
            else:
                edge_id = f"edge_{edge_index}"
            seen_edge_ids.add(edge_id)

            source_pos = np.asarray(graph.nodes[source]["pos"], dtype=float)
            target_pos = np.asarray(graph.nodes[target]["pos"], dtype=float)
            pts = None
            if points_col and points_col in fields:
                pts = _parse_points_json(
                    row.get(points_col),
                    edge_id=edge_id,
                    source_pos=source_pos,
                    target_pos=target_pos,
                )
            if pts is None:
                pts = np.vstack([source_pos, target_pos])

            attrs = _optional_attrs(row, reserved)
            graph.add_edge(source, target, key=edge_id, pts=pts.copy(), edge_id=edge_id, **attrs)

    issues = validate_spatial_graph(graph)
    return SpatialGraphInputResult(
        graph_id=resolved_graph_id,
        nodes_path=nodes_path,
        edges_path=edges_path,
        graph=graph,
        metadata=meta,
        issues=issues,
    )


__all__ = [
    "SpatialGraphInputResult",
    "from_spatial_graph_csv",
    "validate_spatial_graph",
]
