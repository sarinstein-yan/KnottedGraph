"""Abstract spatial-graph JSON adapter for Task 2 prototypes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


@dataclass
class SpatialGraphResult:
    """Container for a loaded spatial graph."""

    graph_id: str
    source_path: Path
    graph: nx.MultiGraph
    issues: list[str]


def validate_point(value, *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{label} must be a 3D point, got shape {point.shape}.")
    if not np.isfinite(point).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return point


def validate_points(value, *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError(f"{label} must contain at least two points.")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return points


def _node_items(nodes_payload):
    if isinstance(nodes_payload, dict):
        for node_id, pos in nodes_payload.items():
            yield str(node_id), {"pos": pos}
        return
    if isinstance(nodes_payload, list):
        for index, node in enumerate(nodes_payload):
            if "id" not in node:
                raise ValueError(f"node entry {index} is missing 'id'.")
            yield str(node["id"]), node
        return
    raise ValueError("'nodes' must be either a mapping or a list.")


def spatial_graph_from_json_payload(payload: dict) -> nx.MultiGraph:
    """Convert a friendly JSON graph payload into ``MultiGraph(pos/pts)``."""
    if "nodes" not in payload:
        raise ValueError("Spatial graph JSON is missing 'nodes'.")
    if "edges" not in payload:
        raise ValueError("Spatial graph JSON is missing 'edges'.")

    graph = nx.MultiGraph()
    graph_id = str(payload.get("graph_id", "spatial_graph"))
    graph.graph["input_kind"] = "abstract_spatial_graph"
    graph.graph["graph_id"] = graph_id
    graph.graph["is_closed"] = False
    graph.graph.update(payload.get("metadata", {}))

    for node_id, node_data in _node_items(payload["nodes"]):
        if "pos" not in node_data:
            raise ValueError(f"node {node_id!r} is missing 'pos'.")
        graph.add_node(node_id, pos=validate_point(node_data["pos"], label=f"node {node_id!r} pos"))

    for edge_index, edge in enumerate(payload["edges"]):
        try:
            source = str(edge["source"])
            target = str(edge["target"])
        except KeyError as exc:
            raise ValueError(f"edge entry {edge_index} is missing {exc}.") from exc

        if source not in graph:
            raise ValueError(f"edge entry {edge_index} references unknown source {source!r}.")
        if target not in graph:
            raise ValueError(f"edge entry {edge_index} references unknown target {target!r}.")

        if "points" in edge:
            pts = validate_points(edge["points"], label=f"edge {edge_index} points")
        elif "pts" in edge:
            pts = validate_points(edge["pts"], label=f"edge {edge_index} pts")
        else:
            pts = np.vstack([graph.nodes[source]["pos"], graph.nodes[target]["pos"]])

        if not np.allclose(pts[0], graph.nodes[source]["pos"]):
            raise ValueError(f"edge entry {edge_index} first point does not match source pos.")
        if not np.allclose(pts[-1], graph.nodes[target]["pos"]):
            raise ValueError(f"edge entry {edge_index} last point does not match target pos.")

        edge_key = str(edge.get("id", edge.get("key", f"edge_{edge_index}")))
        attrs = {
            key: value
            for key, value in edge.items()
            if key not in {"source", "target", "points", "pts", "id", "key"}
        }
        graph.add_edge(source, target, key=edge_key, pts=pts.copy(), **attrs)

    return graph


def validate_spatial_graph(graph: nx.MultiGraph) -> list[str]:
    """Return schema issues for a spatial graph, if any."""
    issues = []
    if not isinstance(graph, nx.MultiGraph):
        return ["graph is not a networkx.MultiGraph"]
    if graph.number_of_nodes() < 1:
        issues.append("graph has no nodes")
    if graph.number_of_edges() < 1:
        issues.append("graph has no edges")

    for node, data in graph.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            issues.append(f"node {node!r} is missing 'pos'")
            continue
        if np.asarray(pos).shape != (3,):
            issues.append(f"node {node!r} pos has shape {np.asarray(pos).shape}")

    for u, v, key, data in graph.edges(keys=True, data=True):
        pts = data.get("pts")
        if pts is None:
            issues.append(f"edge {(u, v, key)!r} is missing 'pts'")
            continue
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 3:
            issues.append(f"edge {(u, v, key)!r} pts has shape {pts.shape}")
            continue
        if "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
            u_pos = graph.nodes[u]["pos"]
            v_pos = graph.nodes[v]["pos"]
            direct = np.allclose(pts[0], u_pos) and np.allclose(pts[-1], v_pos)
            reverse = np.allclose(pts[0], v_pos) and np.allclose(pts[-1], u_pos)
            if not (direct or reverse):
                issues.append(
                    f"edge {(u, v, key)!r} endpoints do not match its node positions"
                )
    return issues


def build_spatial_graph_from_json(path: Path) -> SpatialGraphResult:
    payload = json.loads(path.read_text())
    graph = spatial_graph_from_json_payload(payload)
    graph_id = graph.graph.get("graph_id", path.stem)
    graph.graph["source_format"] = "json"
    graph.graph["source_path"] = str(path)
    issues = validate_spatial_graph(graph)
    return SpatialGraphResult(graph_id=graph_id, source_path=path, graph=graph, issues=issues)


def write_spatial_graph_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _empty_csv_value(value: str | None) -> bool:
    return value is None or value.strip() == ""


def _parse_edge_points_json(value: str, *, edge_id: str) -> list[list[float]]:
    try:
        points = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"edge {edge_id!r} has invalid points_json") from exc
    validate_points(points, label=f"edge {edge_id!r} points_json")
    return points


def spatial_graph_payload_from_csv(
    nodes_path: Path,
    edges_path: Path,
    *,
    graph_id: str | None = None,
) -> dict:
    """Build a friendly spatial-graph payload from node/edge CSV files.

    Node CSV columns:

    - required: ``id``, ``x``, ``y``, ``z``

    Edge CSV columns:

    - required: ``source``, ``target``
    - optional: ``id`` or ``key``
    - optional: ``points_json`` containing ``[[x, y, z], ...]``
    """
    nodes = []
    with nodes_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "x", "y", "z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"node CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            node_id = row["id"].strip()
            if not node_id:
                raise ValueError(f"node CSV line {row_number}: empty id")
            try:
                pos = [float(row["x"]), float(row["y"]), float(row["z"])]
            except ValueError as exc:
                raise ValueError(f"node CSV line {row_number}: invalid coordinates") from exc
            nodes.append({"id": node_id, "pos": pos})

    edges = []
    with edges_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"edge CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            source = row["source"].strip()
            target = row["target"].strip()
            if not source or not target:
                raise ValueError(f"edge CSV line {row_number}: empty source/target")
            edge_id = row.get("id") or row.get("key") or f"edge_{row_number - 2}"
            edge = {"id": edge_id.strip(), "source": source, "target": target}
            if not _empty_csv_value(row.get("points_json")):
                edge["points"] = _parse_edge_points_json(row["points_json"], edge_id=edge["id"])
            edges.append(edge)

    return {
        "graph_id": graph_id or edges_path.stem,
        "metadata": {
            "description": "spatial graph loaded from node/edge CSV files",
            "example_kind": "csv_spatial_graph",
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
        },
        "nodes": nodes,
        "edges": edges,
    }


def build_spatial_graph_from_csv(
    nodes_path: Path,
    edges_path: Path,
    *,
    graph_id: str | None = None,
) -> SpatialGraphResult:
    payload = spatial_graph_payload_from_csv(nodes_path, edges_path, graph_id=graph_id)
    graph = spatial_graph_from_json_payload(payload)
    graph_id = graph.graph.get("graph_id", edges_path.stem)
    graph.graph["source_format"] = "csv"
    graph.graph["source_path"] = f"{nodes_path};{edges_path}"
    issues = validate_spatial_graph(graph)
    return SpatialGraphResult(graph_id=graph_id, source_path=edges_path, graph=graph, issues=issues)


def write_spatial_graph_csv(payload: dict, nodes_path: Path, edges_path: Path) -> None:
    """Write a spatial-graph payload as node/edge CSV files."""
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "z"])
        for node_id, node_data in _node_items(payload["nodes"]):
            x, y, z = validate_point(node_data["pos"], label=f"node {node_id!r} pos")
            writer.writerow([node_id, f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"])

    with edges_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "target", "points_json"])
        for edge_index, edge in enumerate(payload["edges"]):
            edge_id = edge.get("id", edge.get("key", f"edge_{edge_index}"))
            points = edge.get("points", edge.get("pts", ""))
            points_json = json.dumps(points) if points != "" else ""
            writer.writerow([edge_id, edge["source"], edge["target"], points_json])
