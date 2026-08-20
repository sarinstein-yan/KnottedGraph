"""Convert skeletonized images into spatial graph objects.

The topology-aware backend operates on the sparse foreground skeleton rather
than scanning the full image volume. For 3-D inputs it is the default backend.
Its zero-radius mode reproduces the historical voxel-graph semantics while
avoiding the full-volume ``poly2graph`` marking pass. When a maximum junction
valence is supplied, a fail-closed multi-scale correction may additionally
collapse digital junction fragments, but only after the repaired topology is
stable at the next scale.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "skeleton_image_to_graph",
    "topology_aware_skeleton_image_to_graph",
]

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _sparse_voxel_adjacency(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Return sorted foreground indices, coordinates, and 26-neighbour lists."""
    flat = np.flatnonzero(image)
    if flat.size == 0:
        return flat, np.empty((0, 3), dtype=np.intp), []

    shape = image.shape
    coords = np.column_stack(np.unravel_index(flat, shape)).astype(
        np.intp,
        copy=False,
    )
    strides = np.asarray(
        (shape[1] * shape[2], shape[2], 1),
        dtype=np.int64,
    )
    adjacency: list[list[int]] = [[] for _ in range(flat.size)]

    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        if (dx, dy, dz) <= (0, 0, 0):
            continue

        delta = dx * strides[0] + dy * strides[1] + dz
        query = flat + delta
        positions = np.searchsorted(flat, query)
        valid = positions < flat.size
        left = np.flatnonzero(valid)
        right = positions[valid]
        exact = flat[right] == query[valid]
        left = left[exact]
        right = right[exact]
        if left.size == 0:
            continue

        wanted = np.asarray((dx, dy, dz), dtype=np.intp)
        actual = coords[right] - coords[left]
        keep = np.all(actual == wanted, axis=1)
        for u, v in zip(left[keep].tolist(), right[keep].tolist()):
            adjacency[u].append(v)
            adjacency[v].append(u)

    for u, row in enumerate(adjacency):
        row.sort(key=lambda v: tuple((coords[v] - coords[u]).tolist()))

    return flat, coords, adjacency


def _adjacency_components(adjacency: list[list[int]]) -> list[list[int]]:
    """Return 26-connected voxel components in deterministic scan order."""
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


def _expand_zone(
    seed: set[int],
    adjacency: list[list[int]],
    hops: int,
) -> set[int]:
    zone = set(seed)
    frontier = set(seed)
    for _ in range(hops):
        frontier = {v for u in frontier for v in adjacency[u]} - zone
        zone.update(frontier)
        if not frontier:
            break
    return zone


def _zone_components(
    zone: set[int],
    adjacency: list[list[int]],
) -> list[list[int]]:
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


def _trace_cycle(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
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


def _trace_component(
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
            weight=float(
                np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            ),
        )
        return graph

    zone = _expand_zone(special, adjacency, junction_hops)
    components = _zone_components(zone, adjacency)
    component_of = {
        voxel: index
        for index, component in enumerate(components)
        for voxel in component
    }

    graph = nx.MultiGraph()
    for index, component in enumerate(components):
        graph.add_node(
            index,
            pos=np.rint(coords[component].mean(axis=0)).astype(float),
        )

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
                    candidates = [
                        v for v in adjacency[current] if v != previous
                    ]
                    if not candidates:
                        break
                    nxt = next(
                        (
                            v
                            for v in candidates
                            if edge_key(current, v) not in visited
                        ),
                        candidates[0],
                    )
                    visited.add(edge_key(current, nxt))
                    previous, current = current, nxt
                    path.append(current)
                    if len(path) > n_voxels + 2:
                        raise RuntimeError(
                            "skeleton path tracing did not terminate"
                        )

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
                    weight=float(
                        np.linalg.norm(
                            np.diff(points, axis=0),
                            axis=1,
                        ).sum()
                    ),
                )

    if junction_hops > 0:
        max_zero_loop_points = max(4, 2 * junction_hops + 3)
        for u, v, key, data in list(
            graph.edges(keys=True, data=True)
        ):
            if (
                u == v
                and len(data.get("pts", ())) <= max_zero_loop_points
            ):
                graph.remove_edge(u, v, key)

    return graph


def _trace_all(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    junction_hops: int,
) -> nx.MultiGraph:
    """Trace every disconnected voxel component without losing link components."""
    graph = nx.MultiGraph()
    next_id = 0

    for indices in _adjacency_components(adjacency):
        remap = {old: index for index, old in enumerate(indices)}
        local_coords = coords[np.asarray(indices, dtype=np.intp)]
        local_adjacency = [
            [remap[v] for v in adjacency[old] if v in remap]
            for old in indices
        ]
        local = _trace_component(
            local_coords,
            local_adjacency,
            junction_hops=junction_hops,
        )
        mapping = {node: node + next_id for node in local.nodes()}
        local = nx.relabel_nodes(local, mapping, copy=True)
        graph = nx.compose(graph, local)
        next_id += local.number_of_nodes()

    graph.remove_nodes_from(
        [node for node, degree in graph.degree() if degree == 0]
    )
    return graph


def _remove_leaves_for_diagnostic(graph: nx.MultiGraph) -> nx.MultiGraph:
    result = nx.MultiGraph(graph)
    while True:
        leaves = [
            node for node, degree in result.degree() if degree == 1
        ]
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


def _homeomorph_core(graph: nx.MultiGraph) -> nx.MultiGraph:
    source = nx.MultiGraph(graph)
    source.remove_nodes_from(
        [node for node, degree in source.degree() if degree == 0]
    )
    result = nx.MultiGraph()
    seen: set[tuple[object, object, object]] = set()

    def edge_tag(
        u: object,
        v: object,
        key: object,
    ) -> tuple[object, object, object]:
        return (u, v, key) if repr(u) <= repr(v) else (v, u, key)

    for component in nx.connected_components(source):
        terminals = {
            node for node in component if source.degree(node) != 2
        }
        if not terminals:
            representative = min(component, key=repr)
            result.add_node(representative)
            result.add_edge(representative, representative)
            continue

        for node in sorted(terminals, key=repr):
            result.add_node(node)

        for start in sorted(terminals, key=repr):
            for neighbour, edge_dict in source.adj[start].items():
                for key in edge_dict:
                    tag = edge_tag(start, neighbour, key)
                    if tag in seen:
                        continue
                    seen.add(tag)
                    current = neighbour

                    while (
                        current not in terminals
                        and source.degree(current) == 2
                    ):
                        found = None
                        for candidate, next_edges in source.adj[current].items():
                            for next_key in next_edges:
                                next_tag = edge_tag(
                                    current,
                                    candidate,
                                    next_key,
                                )
                                if next_tag not in seen:
                                    seen.add(next_tag)
                                    found = candidate
                                    break
                            if found is not None:
                                break
                        if found is None:
                            break
                        current = found

                    if current not in result:
                        result.add_node(current)
                    result.add_edge(start, current)

    return result


def _diagnostic_graph(graph: nx.MultiGraph) -> nx.MultiGraph:
    return _homeomorph_core(_remove_leaves_for_diagnostic(graph))


def _reduced_weighted(graph: nx.MultiGraph) -> nx.MultiGraph:
    source = nx.MultiGraph(graph)
    source.remove_nodes_from(
        [node for node, degree in source.degree() if degree == 0]
    )
    result = nx.MultiGraph()
    seen: set[tuple[object, object, object]] = set()

    def edge_tag(
        u: object,
        v: object,
        key: object,
    ) -> tuple[object, object, object]:
        return (u, v, key) if repr(u) <= repr(v) else (v, u, key)

    for component in nx.connected_components(source):
        terminals = {
            node for node in component if source.degree(node) != 2
        }
        if not terminals:
            representative = min(component, key=repr)
            result.add_node(
                representative,
                pos=source.nodes[representative].get("pos"),
            )
            total = sum(
                float(data.get("weight", 0.0))
                for _, _, _, data in source.subgraph(component).edges(
                    keys=True,
                    data=True,
                )
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

                    while (
                        current not in terminals
                        and source.degree(current) == 2
                    ):
                        found = None
                        for candidate, next_edges in source.adj[current].items():
                            for next_key, next_attrs in next_edges.items():
                                next_tag = edge_tag(
                                    current,
                                    candidate,
                                    next_key,
                                )
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
                        result.add_node(
                            current,
                            pos=source.nodes[current].get("pos"),
                        )
                    result.add_edge(start, current, weight=length)

    return result


def _anomaly_score_pruned(
    graph: nx.MultiGraph,
    ratio: float = 0.15,
) -> tuple[int, float]:
    pruned = _remove_leaves_for_diagnostic(graph)
    reduced = _reduced_weighted(pruned)
    count = 0
    worst = 1.0
    pairs: set[tuple[object, object]] = set()

    for u, v in reduced.edges():
        if u != v and reduced.degree(u) >= 3 and reduced.degree(v) >= 3:
            pairs.add(tuple(sorted((u, v), key=repr)))

    for u, v in pairs:
        pair_lengths = [
            float(data.get("weight", 0.0))
            for data in reduced.get_edge_data(u, v, default={}).values()
        ]
        refs_u: list[float] = []
        refs_v: list[float] = []

        for a, b, _, data in reduced.edges(u, keys=True, data=True):
            other = b if a == u else a
            if other not in (u, v):
                refs_u.append(float(data.get("weight", 0.0)))
        for a, b, _, data in reduced.edges(v, keys=True, data=True):
            other = b if a == v else a
            if other not in (u, v):
                refs_v.append(float(data.get("weight", 0.0)))

        if not refs_u or not refs_v:
            continue
        scale = min(float(np.median(refs_u)), float(np.median(refs_v)))
        if scale <= 0:
            continue

        score = min(pair_lengths) / scale
        if score < ratio:
            count += 1
            worst = min(worst, score)

    return count, worst


def _constrained_select(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    max_degree: int,
    max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    def build(hops: int) -> tuple[nx.MultiGraph, nx.MultiGraph, bool]:
        graph = _trace_all(
            coords,
            adjacency,
            junction_hops=hops,
        )
        core = _diagnostic_graph(graph)
        max_observed_degree = max(
            (degree for _, degree in core.degree()),
            default=0,
        )
        anomaly_count = _anomaly_score_pruned(
            graph,
            anomaly_ratio,
        )[0]
        clean = (
            max_observed_degree <= max_degree
            and anomaly_count == 0
        )
        return graph, core, clean

    base_graph, base_core, base_clean = build(0)
    if base_clean:
        return base_graph

    previous = (base_graph, base_core, base_clean)
    for hops in range(1, max_hops + 1):
        current = build(hops)
        graph, core, clean = current
        previous_graph, previous_core, previous_clean = previous
        if (
            previous_clean
            and clean
            and nx.is_isomorphic(previous_core, core)
        ):
            return previous_graph
        previous = current

    return base_graph


def topology_aware_skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Convert a 3-D one-voxel skeleton into an embedded ``MultiGraph``.

    With no valence hint the routine uses the sparse zero-radius conversion,
    which preserves the historical graph topology and rounded voxel-centroid
    convention. Supplying ``max_junction_degree`` enables a fail-closed
    multi-scale repair of split junction blobs. A repair is accepted only when
    it satisfies the valence bound and the same cleaned topology persists at
    the next graph-distance scale; otherwise the zero-radius graph is returned.
    """
    image = np.asarray(skeleton_image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")
    if adaptive_max_hops < 0:
        raise ValueError("adaptive_max_hops must be non-negative")

    _, coords, adjacency = _sparse_voxel_adjacency(image)
    if max_junction_degree is None:
        return _trace_all(coords, adjacency, junction_hops=0)
    if max_junction_degree < 1:
        raise ValueError("max_junction_degree must be positive")

    return _constrained_select(
        coords,
        adjacency,
        max_degree=max_junction_degree,
        max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )


def skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    backend: str = "auto",
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Convert a skeletonized image into a ``networkx.MultiGraph``.

    ``backend="auto"`` uses the topology-aware sparse extractor for 3-D inputs
    and retains ``poly2graph`` as the compatibility backend for non-3-D images.
    ``backend="poly2graph"`` can still be requested explicitly for historical
    comparisons. ``max_junction_degree`` is an optional topology constraint
    used only by the adaptive 3-D junction-repair stage.
    """
    image = np.asarray(skeleton_image)
    if backend == "auto":
        backend = "topology_aware" if image.ndim == 3 else "poly2graph"

    if backend == "topology_aware":
        return topology_aware_skeleton_image_to_graph(
            image,
            max_junction_degree=max_junction_degree,
            adaptive_max_hops=adaptive_max_hops,
            anomaly_ratio=anomaly_ratio,
        )
    if backend != "poly2graph":
        raise ValueError(
            "backend must be 'auto', 'poly2graph', or 'topology_aware'"
        )

    from poly2graph import skeleton2graph

    return skeleton2graph(image)
