from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .resampling import ResamplingOptions, resample_polyline_for_options


@dataclass(frozen=True)
class GraphCurveMapping:
    """Bookkeeping needed to write relaxed vertices back to a MultiGraph."""

    node_indices: dict[Any, int]
    edge_vertex_indices: dict[str, list[int]]
    edge_refs: dict[str, tuple[Any, Any, Any]]
    edge_order: tuple[str, ...]
    segment_edges: tuple[tuple[int, int], ...]
    vertex_count: int
    resampling_report: dict[str, Any] | None = None


def _as_point3(value: Any, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{label} must be a 3-vector, got shape {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{label} contains non-finite values")
    return point


def _as_polyline(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must be an array-like object with shape (N, 3)")
    if len(points) < 2:
        raise ValueError(f"{label} must contain at least two points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} contains non-finite values")
    return points.copy()


def _drop_consecutive_duplicates(points: np.ndarray, *, atol: float = 1e-10) -> np.ndarray:
    keep = [0]
    for i in range(1, len(points)):
        if not np.allclose(points[i], points[keep[-1]], atol=atol, rtol=0.0):
            keep.append(i)
    return points[np.asarray(keep, dtype=int)]


def _oriented_edge_polyline(
    G: nx.MultiGraph,
    u: Any,
    v: Any,
    key: Any,
    data: dict[str, Any],
) -> np.ndarray:
    start = _as_point3(G.nodes[u].get("pos"), f"node {u!r} 'pos'")
    end = _as_point3(G.nodes[v].get("pos"), f"node {v!r} 'pos'")

    if data.get("pts") is None:
        points = np.vstack([start, end])
    else:
        points = _as_polyline(data["pts"], f"edge {(u, v, key)!r} 'pts'")

    forward = float(np.linalg.norm(points[0] - start) + np.linalg.norm(points[-1] - end))
    reverse = float(np.linalg.norm(points[0] - end) + np.linalg.norm(points[-1] - start))
    if reverse < forward:
        points = points[::-1].copy()

    points[0] = start
    points[-1] = end
    points = _drop_consecutive_duplicates(points)
    if len(points) < 2:
        raise ValueError(f"edge {(u, v, key)!r} collapsed to fewer than two distinct points")
    return points


def graph_to_curve_arrays(
    G: nx.MultiGraph,
    *,
    resampling_options: ResamplingOptions | None = None,
) -> tuple[np.ndarray, GraphCurveMapping]:
    """Flatten a KnottedGraph spatial MultiGraph into Repulsor vertices/segments."""

    if not isinstance(G, nx.MultiGraph):
        raise TypeError("relax_spatial_graph expects a networkx.MultiGraph")
    if G.is_directed():
        raise TypeError("relax_spatial_graph expects an undirected MultiGraph")
    if G.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node")
    if G.number_of_edges() == 0:
        raise ValueError("graph must contain at least one edge")

    vertices: list[np.ndarray] = []
    node_indices: dict[Any, int] = {}
    for node, data in G.nodes(data=True):
        node_indices[node] = len(vertices)
        vertices.append(_as_point3(data.get("pos"), f"node {node!r} 'pos'"))

    edge_vertex_indices: dict[str, list[int]] = {}
    edge_refs: dict[str, tuple[Any, Any, Any]] = {}
    edge_order: list[str] = []
    segment_edges: list[tuple[int, int]] = []
    edge_resampling: dict[str, Any] = {}

    for edge_index, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
        edge_id = f"edge_{edge_index}"
        points = _oriented_edge_polyline(G, u, v, key, data)
        points, resampling_report = resample_polyline_for_options(points, resampling_options)
        indices = [node_indices[u]]
        for point in points[1:-1]:
            vertices.append(np.asarray(point, dtype=float))
            indices.append(len(vertices) - 1)
        indices.append(node_indices[v])

        if any(a == b for a, b in zip(indices, indices[1:])):
            raise ValueError(f"edge {(u, v, key)!r} contains a zero-length curve segment")

        edge_order.append(edge_id)
        edge_vertex_indices[edge_id] = indices
        edge_refs[edge_id] = (u, v, key)
        edge_resampling[edge_id] = resampling_report
        segment_edges.extend((int(a), int(b)) for a, b in zip(indices, indices[1:]))

    vertex_array = np.asarray(vertices, dtype=float)
    mapping = GraphCurveMapping(
        node_indices=node_indices,
        edge_vertex_indices=edge_vertex_indices,
        edge_refs=edge_refs,
        edge_order=tuple(edge_order),
        segment_edges=tuple(segment_edges),
        vertex_count=len(vertices),
        resampling_report={
            "enabled": resampling_options is not None,
            "options": asdict(resampling_options) if resampling_options is not None else None,
            "edges": edge_resampling,
        },
    )
    return vertex_array, mapping


def write_graph_obj(vertices: np.ndarray, segment_edges: tuple[tuple[int, int], ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for point in vertices:
            f.write(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
        for a, b in segment_edges:
            f.write(f"l {a + 1} {b + 1}\n")


def write_graph_curve(
    vertices: np.ndarray,
    segment_edges: tuple[tuple[int, int], ...],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(f"vertices {len(vertices)}\n")
        for point in vertices:
            f.write(f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
        f.write(f"edges {len(segment_edges)}\n")
        for a, b in segment_edges:
            f.write(f"{a} {b}\n")


def write_pinned_vertices(indices: list[int] | tuple[int, ...] | set[int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for index in sorted(int(i) for i in indices):
            f.write(f"{index}\n")


def graph_from_curve_vertices(
    template: nx.MultiGraph,
    vertices: np.ndarray,
    mapping: GraphCurveMapping,
) -> nx.MultiGraph:
    if len(vertices) != mapping.vertex_count:
        raise ValueError(
            f"expected {mapping.vertex_count} vertices from Repulsor, got {len(vertices)}"
        )

    relaxed = template.copy()
    for node, index in mapping.node_indices.items():
        relaxed.nodes[node]["pos"] = np.asarray(vertices[index], dtype=float).copy()

    for edge_id in mapping.edge_order:
        u, v, key = mapping.edge_refs[edge_id]
        indices = np.asarray(mapping.edge_vertex_indices[edge_id], dtype=int)
        relaxed.edges[u, v, key]["pts"] = np.asarray(vertices[indices], dtype=float).copy()

    return relaxed


def reindex_mapping(
    mapping: GraphCurveMapping,
    old_to_new: dict[int, int],
    edge_vertex_indices: dict[str, list[int]],
) -> GraphCurveMapping:
    node_indices = {
        node: int(old_to_new[index])
        for node, index in mapping.node_indices.items()
    }
    segment_edges: list[tuple[int, int]] = []
    for edge_id in mapping.edge_order:
        indices = edge_vertex_indices[edge_id]
        segment_edges.extend((int(a), int(b)) for a, b in zip(indices, indices[1:]))

    return GraphCurveMapping(
        node_indices=node_indices,
        edge_vertex_indices=edge_vertex_indices,
        edge_refs=mapping.edge_refs,
        edge_order=mapping.edge_order,
        segment_edges=tuple(segment_edges),
        vertex_count=len(old_to_new),
        resampling_report=mapping.resampling_report,
    )


def mapping_metadata(mapping: GraphCurveMapping) -> dict[str, Any]:
    return {
        "node_count": len(mapping.node_indices),
        "edge_count": len(mapping.edge_order),
        "curve_vertex_count": mapping.vertex_count,
        "curve_segment_count": len(mapping.segment_edges),
        "edge_order": list(mapping.edge_order),
        "resampling": mapping.resampling_report,
        "edges": {
            edge_id: {
                "u": repr(mapping.edge_refs[edge_id][0]),
                "v": repr(mapping.edge_refs[edge_id][1]),
                "key": repr(mapping.edge_refs[edge_id][2]),
                "vertex_indices": mapping.edge_vertex_indices[edge_id],
            }
            for edge_id in mapping.edge_order
        },
    }
