from __future__ import annotations

import logging
from typing import Any, Sequence

import fastrdp
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EmbeddingValidationError",
    "as_point3",
    "as_polyline",
    "drop_consecutive_duplicates",
    "oriented_edge_polyline",
    "validate_embedding",
    "ensure_embedding",
    "is_embedding",
    "idx_to_coord",
    "get_all_edge_pts",
    "total_edge_pts",
    "smooth_edges",
    "remove_leaf_nodes",
    "simplify_edges",
    "contract_short_edges",
]


class EmbeddingValidationError(ValueError):
    """Raised when a graph cannot satisfy the embedded graph contract."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


def as_point3(value: Any, label: str) -> np.ndarray:
    """Return *value* as a finite 3D point."""

    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{label} must be a 3D point, got shape {point.shape}.")
    if not np.isfinite(point).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return point.copy()


def as_polyline(value: Any, label: str) -> np.ndarray:
    """Return *value* as a finite polyline with shape ``(N, 3)``."""

    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError(f"{label} must contain at least two points.")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return points.copy()


def drop_consecutive_duplicates(points: np.ndarray, *, atol: float = 1e-10) -> np.ndarray:
    """Drop adjacent duplicate points from a polyline."""

    if len(points) == 0:
        return points
    keep = [0]
    for index in range(1, len(points)):
        if not np.allclose(points[index], points[keep[-1]], atol=atol, rtol=0.0):
            keep.append(index)
    return points[np.asarray(keep, dtype=int)]


def oriented_edge_polyline(
    graph: nx.MultiGraph,
    u: Any,
    v: Any,
    key: Any,
    data: dict[str, Any],
) -> np.ndarray:
    """Return an edge polyline oriented from node *u* to node *v*."""

    start = as_point3(graph.nodes[u].get("pos"), f"node {u!r} 'pos'")
    end = as_point3(graph.nodes[v].get("pos"), f"node {v!r} 'pos'")

    if data.get("pts") is None:
        points = np.vstack([start, end])
    else:
        points = as_polyline(data["pts"], f"edge {(u, v, key)!r} 'pts'")

    forward = float(np.linalg.norm(points[0] - start) + np.linalg.norm(points[-1] - end))
    reverse = float(np.linalg.norm(points[0] - end) + np.linalg.norm(points[-1] - start))
    if reverse < forward:
        points = points[::-1].copy()

    points[0] = start
    points[-1] = end
    points = drop_consecutive_duplicates(points)
    if len(points) < 2:
        raise ValueError(f"edge {(u, v, key)!r} collapsed to fewer than two distinct points.")
    return points


def validate_embedding(graph: nx.MultiGraph) -> list[str]:
    """Return issues for a normalizable embedded ``MultiGraph(pos/pts)``."""

    issues: list[str] = []
    if not isinstance(graph, nx.MultiGraph):
        return ["graph is not a networkx.MultiGraph"]
    if graph.is_directed():
        issues.append("graph must be undirected")
    if graph.number_of_nodes() == 0:
        issues.append("graph has no nodes")
    if graph.number_of_edges() == 0:
        issues.append("graph has no edges")

    valid_positions: dict[Any, np.ndarray] = {}
    for node, data in graph.nodes(data=True):
        if "pos" not in data:
            issues.append(f"node {node!r} is missing 'pos'")
            continue
        try:
            valid_positions[node] = as_point3(data["pos"], f"node {node!r} pos")
        except ValueError as exc:
            issues.append(str(exc))

    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("pts") is None:
            continue
        try:
            pts = as_polyline(data["pts"], f"edge {(u, v, key)!r} pts")
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if u not in valid_positions or v not in valid_positions:
            continue
        u_pos = valid_positions[u]
        v_pos = valid_positions[v]
        direct = np.allclose(pts[0], u_pos) and np.allclose(pts[-1], v_pos)
        reverse = np.allclose(pts[0], v_pos) and np.allclose(pts[-1], u_pos)
        if not (direct or reverse):
            issues.append(f"edge {(u, v, key)!r} endpoints do not match node positions")

    return issues


def ensure_embedding(
    graph: nx.MultiGraph,
    *,
    copy: bool = True,
    normalize: bool = True,
) -> nx.MultiGraph:
    """Validate and optionally normalize an embedded spatial graph."""

    issues = validate_embedding(graph)
    if issues:
        raise EmbeddingValidationError(issues)

    result = graph.copy() if copy else graph
    if not normalize:
        return result

    for _, data in result.nodes(data=True):
        data["pos"] = as_point3(data["pos"], "node 'pos'")

    for u, v, key, data in result.edges(keys=True, data=True):
        data["pts"] = oriented_edge_polyline(result, u, v, key, data)

    return result


def is_embedding(graph: nx.MultiGraph) -> bool:
    """Return whether *graph* satisfies the embedded graph contract."""

    return not validate_embedding(graph)


def idx_to_coord(
    indices: ArrayLike,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> NDArray:
    """Convert an array of 3D image indices to spatial coordinates."""

    array = np.asarray(indices)
    if array.shape[-1] != 3:
        raise ValueError("Input array must have shape (..., 3).")

    return array * spacing + origin


def get_all_edge_pts(G: nx.MultiGraph) -> NDArray:
    """Get all edge points from the graph as a single array."""

    graph = ensure_embedding(G, copy=False, normalize=False)
    edge_pts_list = [oriented_edge_polyline(graph, u, v, k, data) for u, v, k, data in graph.edges(keys=True, data=True)]
    return np.concatenate(edge_pts_list)


def total_edge_pts(G: nx.MultiGraph) -> int:
    """Count the total number of edge-polyline points."""

    return len(get_all_edge_pts(G))


def smooth_edges(
    G: nx.MultiGraph,
    epsilon: float = 0.0,
    copy: bool = True,
) -> nx.MultiGraph:
    """Simplify edge polylines with Ramer-Douglas-Peucker smoothing."""

    H = ensure_embedding(G, copy=copy, normalize=True)
    for u, v, key, pts in H.edges(keys=True, data="pts"):
        if pts is None:
            continue
        pts_arr = np.asarray(pts, dtype=float)
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 3:
            continue

        simplified = fastrdp.rdpN(pts_arr, epsilon)
        H[u][v][key]["pts"] = oriented_edge_polyline(
            H,
            u,
            v,
            key,
            {"pts": simplified},
        )
    return H


def remove_leaf_nodes(G: nx.MultiGraph) -> nx.MultiGraph:
    """Remove degree-1 leaves from a copy of an embedded graph."""

    H = G.copy()
    while True:
        leaf_nodes = [node for node, degree in H.degree() if degree == 1]
        if not leaf_nodes:
            break
        if len(leaf_nodes) == H.number_of_nodes():
            random_node = leaf_nodes[0]
            result = nx.MultiGraph()
            result.add_node(random_node, **H.nodes[random_node])
            return result
        for node in leaf_nodes:
            H.remove_node(node)
    return H


def contract_short_edges(
    G: nx.MultiGraph,
    min_length: float = 0.30,
    *,
    copy: bool = True,
) -> nx.MultiGraph:
    """Contract edges whose endpoint distance is below ``min_length``.

    This is useful after skeletonization, where voxelization can introduce tiny
    spurious edges between nearby junction nodes.  The operation preserves
    embedded edge polylines by moving incident edge endpoints onto the merged
    vertex.
    """

    H = ensure_embedding(G, copy=copy, normalize=True)

    def endpoint_distance(u: Any, v: Any) -> float:
        return float(np.linalg.norm(H.nodes[u]["pos"] - H.nodes[v]["pos"]))

    def relink_edge_points(
        pts: Any,
        old_endpoint: np.ndarray,
        new_endpoint: np.ndarray,
        other_endpoint: np.ndarray,
    ) -> np.ndarray:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            arr = np.vstack([old_endpoint, other_endpoint])

        if np.linalg.norm(arr[0] - old_endpoint) <= np.linalg.norm(arr[-1] - old_endpoint):
            arr[0] = new_endpoint
            arr[-1] = other_endpoint
        else:
            arr[-1] = new_endpoint
            arr[0] = other_endpoint

        return drop_consecutive_duplicates(arr)

    while True:
        candidates: list[tuple[float, Any, Any]] = []
        for u, v in H.edges():
            if u == v:
                continue
            length = endpoint_distance(u, v)
            if length < min_length:
                candidates.append((length, u, v))

        if not candidates:
            break

        _, u, v = min(candidates, key=lambda item: item[0])
        degree_u, degree_v = H.degree[u], H.degree[v]
        if degree_u > degree_v:
            keep, kill = u, v
        elif degree_v > degree_u:
            keep, kill = v, u
        else:
            keep, kill = (u, v) if str(u) <= str(v) else (v, u)

        keep_pos = np.asarray(H.nodes[keep]["pos"], dtype=float)
        kill_pos = np.asarray(H.nodes[kill]["pos"], dtype=float)
        merged_pos = 0.5 * (keep_pos + kill_pos)
        H.nodes[keep]["pos"] = merged_pos

        for a, b, key, data in list(H.edges(keep, keys=True, data=True)):
            if kill in (a, b):
                continue
            other = b if a == keep else a
            other_pos = np.asarray(H.nodes[other]["pos"], dtype=float)
            H[a][b][key]["pts"] = relink_edge_points(
                data.get("pts", np.vstack([keep_pos, other_pos])),
                old_endpoint=keep_pos,
                new_endpoint=merged_pos,
                other_endpoint=other_pos,
            )

        incident_edges = list(H.edges(kill, keys=True, data=True))
        for a, b, key, data in incident_edges:
            other = b if a == kill else a
            if H.has_edge(a, b, key):
                H.remove_edge(a, b, key)

            if other == keep:
                continue

            other_pos = np.asarray(H.nodes[other]["pos"], dtype=float)
            edge_data = dict(data or {})
            edge_data["pts"] = relink_edge_points(
                edge_data.get("pts", np.vstack([kill_pos, other_pos])),
                old_endpoint=kill_pos,
                new_endpoint=merged_pos,
                other_endpoint=other_pos,
            )
            H.add_edge(keep, other, **edge_data)

        if kill in H:
            H.remove_node(kill)

    return ensure_embedding(H, copy=False, normalize=True)


def _append_edge_pts(path: list[np.ndarray], edge_pts: Any) -> None:
    if edge_pts is None or len(edge_pts) == 0:
        return

    pts = np.asarray(edge_pts, dtype=float)
    if np.array_equal(pts[-1], path[-1]):
        pts = pts[::-1]

    if np.array_equal(pts[0], path[-1]):
        path.extend(pts[1:])
        return

    raise RuntimeError(
        "Edge segment does not connect contiguously:\n"
        f"  current tail = {path[-1]}\n"
        f"  segment ends = ({pts[0]}, {pts[-1]})"
    )


def _edge_tag(u: int, v: int, key: int) -> tuple[int, int, int]:
    """Return a canonical tag for an undirected multiedge."""

    return (u, v, key) if u <= v else (v, u, key)


def _has_cycles(G: nx.MultiGraph) -> bool:
    """Quick check for any cycle in *G*."""

    if G.number_of_edges() == 0:
        return False
    try:
        nx.find_cycle(G)
        return True
    except nx.NetworkXNoCycle:
        return False


def _collapse_component_with_junctions(
    G: nx.MultiGraph,
    comp: set[int],
    H: nx.MultiGraph,
) -> None:
    """Collapse chains inside a component that has junction nodes."""

    junctions = {node for node in comp if G.degree(node) > 2}
    for node in junctions:
        H.add_node(node, **G.nodes[node])

    seen_edges: set[tuple[int, int, int]] = set()

    for junction in junctions:
        for neighbor, edge_dict in G.adj[junction].items():
            for key, attrs in edge_dict.items():
                tag = _edge_tag(junction, neighbor, key)
                if tag in seen_edges:
                    continue
                seen_edges.add(tag)

                path_pts: list[np.ndarray] = [G.nodes[junction]["pos"]]
                _append_edge_pts(path_pts, attrs.get("pts", []))

                previous, current = junction, neighbor
                while current not in junctions and G.degree(current) == 2:
                    path_pts.append(G.nodes[current]["pos"])
                    next_candidates = [node for node in G.neighbors(current) if node != previous]
                    if not next_candidates:
                        break
                    nxt = next_candidates[0]

                    for key2, attrs2 in G[current][nxt].items():
                        tag2 = _edge_tag(current, nxt, key2)
                        if tag2 not in seen_edges:
                            seen_edges.add(tag2)
                            _append_edge_pts(path_pts, attrs2.get("pts", []))
                            break
                    previous, current = current, nxt

                path_pts.append(G.nodes[current]["pos"])
                if current not in H:
                    H.add_node(current, **G.nodes[current])

                H.add_edge(junction, current, pts=np.asarray(path_pts))


def _collapse_cycle_component(
    G: nx.MultiGraph,
    comp: set[int],
    H: nx.MultiGraph,
) -> None:
    """Collapse a component with no junctions to a self-loop."""

    rep = next((node for node in comp if G.degree(node) == 2), None) or next(iter(comp))
    H.add_node(rep, **G.nodes[rep])

    if G.degree(rep) == 0:
        return

    path_pts: list[np.ndarray] = [G.nodes[rep]["pos"]]
    seen_edges: set[tuple[int, int, int]] = set()

    previous, current = rep, next(iter(G.neighbors(rep)))

    for key, attrs in G[previous][current].items():
        tag = _edge_tag(previous, current, key)
        if tag not in seen_edges:
            seen_edges.add(tag)
            _append_edge_pts(path_pts, attrs.get("pts", []))
            break

    while current != rep:
        path_pts.append(G.nodes[current]["pos"])
        next_candidates = [node for node in G.neighbors(current) if node != previous]
        if not next_candidates:
            break
        nxt = next_candidates[0]

        for key2, attrs2 in G[current][nxt].items():
            tag2 = _edge_tag(current, nxt, key2)
            if tag2 not in seen_edges:
                seen_edges.add(tag2)
                _append_edge_pts(path_pts, attrs2.get("pts", []))
                break
        previous, current = current, nxt

    path_pts.append(G.nodes[rep]["pos"])
    H.add_edge(rep, rep, pts=np.asarray(path_pts))


def _copy_component(
    G: nx.MultiGraph,
    comp: set,
    H: nx.MultiGraph,
) -> None:
    """Copy one component, including all node and keyed-edge metadata."""

    for node in comp:
        H.add_node(node, **G.nodes[node])
    for u, v, key, data in G.subgraph(comp).edges(keys=True, data=True):
        H.add_edge(u, v, key=key, **data)


def simplify_edges(G: nx.MultiGraph) -> nx.MultiGraph:
    """Simplify degree-2 chains without discarding embedded connectivity.

    Components containing cycles or junctions are represented by embedded
    edges between their significant vertices. Acyclic components are returned
    normalized but otherwise unchanged so that paths, trees, and per-edge
    metadata cannot disappear implicitly. Use :func:`remove_leaf_nodes`
    explicitly when terminal branches should be removed.
    """

    G = ensure_embedding(G, copy=True, normalize=True)

    H = nx.MultiGraph()
    H.graph.update(G.graph)
    for comp in nx.connected_components(G):
        component = G.subgraph(comp)
        if not _has_cycles(component):
            logging.info(
                "Preserving one normalized acyclic component. Degree-2 chains "
                "are not collapsed because doing so could discard per-edge metadata."
            )
            _copy_component(G, comp, H)
        elif any(G.degree(node) > 2 for node in comp):
            _collapse_component_with_junctions(G, comp, H)
        else:
            _collapse_cycle_component(G, comp, H)

    return nx.convert_node_labels_to_integers(H)
