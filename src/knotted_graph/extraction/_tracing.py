"""Deterministic sparse tracing primitives for optimized 3-D skeleton extraction."""

from __future__ import annotations

import networkx as nx
import numpy as np


def adjacency_components(adjacency: list[list[int]]) -> list[list[int]]:
    """Return connected voxel components in deterministic scan order."""
    seen: set[int] = set()
    components: list[list[int]] = []
    for seed in range(len(adjacency)):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in reversed(adjacency[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(sorted(component))
    return components


def _expand_zone(seed: set[int], adjacency: list[list[int]], hops: int) -> set[int]:
    zone = set(seed)
    frontier = set(seed)
    for _ in range(hops):
        frontier = {v for u in frontier for v in adjacency[u]} - zone
        zone.update(frontier)
        if not frontier:
            break
    return zone


def _zone_components(zone: set[int], adjacency: list[list[int]]) -> list[list[int]]:
    components: list[list[int]] = []
    seen: set[int] = set()
    for seed in sorted(zone):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in reversed(adjacency[u]):
                if v in zone and v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(sorted(component))
    components.sort(key=lambda component: component[0])
    return components


def _trace_cycle(coords: np.ndarray, adjacency: list[list[int]]) -> np.ndarray:
    """Trace a pure degree-2 voxel component as a closed polyline."""
    start = 0
    previous = -1
    current = start
    order = [start]
    for _ in range(len(coords) + 1):
        candidates = [v for v in adjacency[current] if v != previous]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            order.append(start)
            return coords[order].astype(float, copy=True)
        order.append(nxt)
        previous, current = current, nxt
    raise RuntimeError("pure skeleton cycle could not be traced")


def trace_component(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    junction_hops: int,
) -> nx.MultiGraph:
    """Collapse one connected junction zone and trace chains between zones."""
    if len(coords) == 0:
        return nx.MultiGraph()

    degree = np.fromiter((len(row) for row in adjacency), dtype=np.int16)
    special = set(np.flatnonzero(degree != 2).tolist())
    if not special:
        points = _trace_cycle(coords, adjacency)
        graph = nx.MultiGraph()
        graph.add_node(0, pos=points[0].copy())
        graph.add_edge(
            0,
            0,
            pts=points,
            weight=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        )
        return graph

    zone = _expand_zone(special, adjacency, junction_hops)
    components = _zone_components(zone, adjacency)
    component_of = {
        voxel: index for index, component in enumerate(components) for voxel in component
    }

    graph = nx.MultiGraph()
    for index, component in enumerate(components):
        graph.add_node(index, pos=np.rint(coords[component].mean(axis=0)).astype(float))

    visited: set[tuple[int, int]] = set()
    n_voxels = len(coords)

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    for source_component, component in enumerate(components):
        for source_voxel in component:
            for neighbour in adjacency[source_voxel]:
                if neighbour in zone:
                    continue
                first = edge_key(source_voxel, neighbour)
                if first in visited:
                    continue
                path = [source_voxel, neighbour]
                visited.add(first)
                previous, current = source_voxel, neighbour
                while current not in zone:
                    candidates = [v for v in adjacency[current] if v != previous]
                    if not candidates:
                        break
                    nxt = next(
                        (v for v in candidates if edge_key(current, v) not in visited),
                        candidates[0],
                    )
                    visited.add(edge_key(current, nxt))
                    previous, current = current, nxt
                    path.append(current)
                    if len(path) > n_voxels + 2:
                        raise RuntimeError("skeleton path tracing did not terminate")
                if current not in zone:
                    continue
                target_component = component_of[current]
                points = coords[path].astype(float, copy=True)
                points[0] = graph.nodes[source_component]["pos"]
                points[-1] = graph.nodes[target_component]["pos"]
                keep = np.ones(len(points), dtype=bool)
                if len(points) > 1:
                    keep[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
                points = points[keep]
                if len(points) < 2:
                    continue
                graph.add_edge(
                    source_component,
                    target_component,
                    pts=points,
                    weight=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                )

    if junction_hops > 0:
        max_zero_loop_points = max(4, 2 * junction_hops + 3)
        for u, v, key, data in list(graph.edges(keys=True, data=True)):
            if u == v and len(data.get("pts", ())) <= max_zero_loop_points:
                graph.remove_edge(u, v, key)
    return graph


def remove_leaves_for_diagnostic(graph: nx.MultiGraph) -> nx.MultiGraph:
    result = nx.MultiGraph(graph)
    while True:
        leaves = [node for node, degree in result.degree() if degree == 1]
        if not leaves:
            break
        if len(leaves) == result.number_of_nodes():
            keep = leaves[0]
            attrs = dict(result.nodes[keep])
            result.clear()
            result.add_node(keep, **attrs)
            break
        result.remove_nodes_from(leaves)
    return result


def reduced_weighted(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Collapse degree-2 chains while retaining accumulated edge lengths."""
    source = nx.MultiGraph(graph)
    source.remove_nodes_from([node for node, degree in source.degree() if degree == 0])
    result = nx.MultiGraph()
    seen: set[tuple[object, object, object]] = set()

    def edge_tag(u: object, v: object, key: object) -> tuple[object, object, object]:
        return (u, v, key) if repr(u) <= repr(v) else (v, u, key)

    for component in nx.connected_components(source):
        terminals = {node for node in component if source.degree(node) != 2}
        if not terminals:
            representative = min(component, key=repr)
            result.add_node(representative, pos=source.nodes[representative].get("pos"))
            total = sum(
                float(data.get("weight", 0.0))
                for _, _, _, data in source.subgraph(component).edges(keys=True, data=True)
            )
            result.add_edge(representative, representative, weight=total)
            continue

        for node in terminals:
            result.add_node(node, pos=source.nodes[node].get("pos"))
        for start in terminals:
            for neighbour, edge_dict in source.adj[start].items():
                for key, attrs in edge_dict.items():
                    tag = edge_tag(start, neighbour, key)
                    if tag in seen:
                        continue
                    seen.add(tag)
                    length = float(attrs.get("weight", 0.0))
                    current = neighbour
                    while current not in terminals and source.degree(current) == 2:
                        found = None
                        for candidate, next_edges in source.adj[current].items():
                            for next_key, next_attrs in next_edges.items():
                                next_tag = edge_tag(current, candidate, next_key)
                                if next_tag not in seen:
                                    seen.add(next_tag)
                                    found = (candidate, next_attrs)
                                    break
                            if found:
                                break
                        if not found:
                            break
                        current, next_attrs = found
                        length += float(next_attrs.get("weight", 0.0))
                    if current not in result:
                        result.add_node(current, pos=source.nodes[current].get("pos"))
                    result.add_edge(start, current, weight=length)
    return result
