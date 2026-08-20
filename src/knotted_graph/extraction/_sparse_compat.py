"""Sparse zero-radius parser with exact historical edge-order semantics."""

from __future__ import annotations

import networkx as nx
import numpy as np


def trace_zero_radius_compatible(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> nx.MultiGraph:
    """Reproduce ``poly2graph`` scan semantics without a full-volume pass.

    Node blobs are discovered in global foreground-scan order. Edges are then
    started by scanning every node voxel in that same order. During a trace,
    the historical parser keeps the last matching edge neighbour in its
    lexicographic 3x3x3 offset scan; preserving that detail makes downstream
    edge insertion order and point arrays exactly compatible.
    """
    n_voxels = len(coords)
    if n_voxels == 0:
        return nx.MultiGraph()

    degree = np.fromiter((len(row) for row in adjacency), dtype=np.int16)
    is_node_voxel = degree != 2
    node_label = np.full(n_voxels, -1, dtype=np.intp)
    edge_alive = ~is_node_voxel
    seen = np.zeros(n_voxels, dtype=bool)
    node_components: list[list[int]] = []

    for seed in range(n_voxels):
        if not is_node_voxel[seed] or seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        component: list[int] = []
        has_edge_neighbour = False
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency[u]:
                if is_node_voxel[v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
                if not is_node_voxel[v]:
                    has_edge_neighbour = True

        # Match skeleton2graph(..., iso=False): omit isolated voxel blobs.
        if not has_edge_neighbour:
            continue
        component.sort()
        node_id = len(node_components)
        for voxel in component:
            node_label[voxel] = node_id
        node_components.append(component)

    node_positions = [
        np.rint(coords[component].mean(axis=0)).astype(float)
        for component in node_components
    ]
    graph = nx.MultiGraph()
    for node_id, position in enumerate(node_positions):
        graph.add_node(node_id, pos=position)

    def trace(start: int) -> tuple[int, int, np.ndarray]:
        first_node = -1
        second_node = -1
        first_node_voxel = -1
        second_node_voxel = -1
        current = start
        path: list[int] = []

        for _ in range(n_voxels + 1):
            path.append(current)
            edge_alive[current] = False
            next_edge = -1

            for neighbour in adjacency[current]:
                label = int(node_label[neighbour])
                if label >= 0:
                    if first_node < 0:
                        first_node = label
                        first_node_voxel = neighbour
                    else:
                        second_node = label
                        second_node_voxel = neighbour
                # Deliberately overwrite: poly2graph's historical tracer uses
                # newp=cp and therefore follows the last live edge neighbour.
                if edge_alive[neighbour]:
                    next_edge = neighbour

            if second_node >= 0:
                points = coords[
                    [first_node_voxel, *path, second_node_voxel]
                ].astype(float, copy=True)
                points[0] = node_positions[first_node]
                points[-1] = node_positions[second_node]
                return first_node, second_node, points

            if next_edge < 0:
                raise RuntimeError(
                    "skeleton edge trace terminated without a node"
                )
            current = next_edge

        raise RuntimeError("skeleton edge trace did not terminate")

    # Global foreground-scan order, matching poly2graph _parse_struc.
    for node_voxel in range(n_voxels):
        if node_label[node_voxel] < 0:
            continue
        for neighbour in adjacency[node_voxel]:
            if edge_alive[neighbour]:
                source, target, points = trace(neighbour)
                graph.add_edge(
                    source,
                    target,
                    pts=points,
                    weight=float(
                        np.linalg.norm(
                            np.diff(points, axis=0),
                            axis=1,
                        ).sum()
                    ),
                )

    # Remaining degree-2 components are rings. Promote their first foreground
    # voxel to a node exactly as poly2graph's ring=True path does.
    for voxel in range(n_voxels):
        if not edge_alive[voxel]:
            continue
        node_id = len(node_positions)
        position = coords[voxel].astype(float)
        node_positions.append(position)
        node_label[voxel] = node_id
        edge_alive[voxel] = False
        graph.add_node(node_id, pos=position)

        for neighbour in adjacency[voxel]:
            if edge_alive[neighbour]:
                source, target, points = trace(neighbour)
                graph.add_edge(
                    source,
                    target,
                    pts=points,
                    weight=float(
                        np.linalg.norm(
                            np.diff(points, axis=0),
                            axis=1,
                        ).sum()
                    ),
                )

    return graph
