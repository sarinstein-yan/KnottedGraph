"""Convert skeletonized images into spatial graph objects.

The topology-aware backend works directly on the sparse set of foreground
skeleton voxels.  It avoids the full-volume marking/parsing scan used by
``poly2graph`` and treats a junction as a *zone* rather than as an arbitrary
single voxel.  This is important for 3-D thinning, where one physical
branch-point is commonly represented by a small connected cloud of voxels.
"""

from __future__ import annotations

from collections.abc import Iterable

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


def _sparse_voxel_adjacency(image: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Return foreground coordinates and their 26-neighbour adjacency.

    ``np.flatnonzero`` is deliberate here: for sparse one-voxel skeletons it is
    substantially faster than ``np.argwhere`` and avoids a second full-volume
    scan.  Neighbour lookup is performed by vectorised binary searches in the
    already-sorted flat index array.
    """
    flat = np.flatnonzero(image)
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.intp), []

    shape = image.shape
    coords = np.column_stack(np.unravel_index(flat, shape)).astype(np.intp, copy=False)
    strides = np.asarray((shape[1] * shape[2], shape[2], 1), dtype=np.int64)
    adjacency: list[list[int]] = [[] for _ in range(flat.size)]

    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        # Every undirected pair is emitted once.
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

        # Linear-index arithmetic can wrap across a row/plane boundary.  The
        # coordinate check removes precisely those false neighbours.
        wanted = np.asarray((dx, dy, dz), dtype=np.intp)
        actual = coords[right] - coords[left]
        keep = np.all(actual == wanted, axis=1)
        for u, v in zip(left[keep].tolist(), right[keep].tolist()):
            adjacency[u].append(v)
            adjacency[v].append(u)

    return coords, adjacency


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
    for seed in zone:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency[u]:
                if v in zone and v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(component)
    return components


def _trace_zone_graph(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    junction_hops: int,
) -> nx.MultiGraph:
    """Collapse connected junction zones and trace degree-2 chains once."""
    if len(coords) == 0:
        return nx.MultiGraph()

    degree = np.fromiter((len(row) for row in adjacency), dtype=np.int16)
    special = set(np.flatnonzero(degree != 2).tolist())

    # A pure ring has no non-degree-2 voxels.  Represent it as one node with a
    # self-loop, matching the multigraph convention used by the package.
    if not special:
        graph = nx.MultiGraph()
        graph.add_node(0, pos=coords[0].astype(float))
        graph.add_edge(0, 0, pts=coords.copy(), weight=float(len(coords)))
        return graph

    zone = _expand_zone(special, adjacency, junction_hops)
    components = _zone_components(zone, adjacency)
    component_of = {voxel: i for i, comp in enumerate(components) for voxel in comp}

    graph = nx.MultiGraph()
    for i, comp in enumerate(components):
        graph.add_node(i, pos=coords[comp].mean(axis=0))

    visited: set[tuple[int, int]] = set()
    n_voxels = len(coords)

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    for source_component, component in enumerate(components):
        for source_voxel in component:
            for neighbour in adjacency[source_voxel]:
                if neighbour in zone:
                    continue
                first_key = edge_key(source_voxel, neighbour)
                if first_key in visited:
                    continue

                path = [source_voxel, neighbour]
                visited.add(first_key)
                previous, current = source_voxel, neighbour

                while current not in zone:
                    candidates = [v for v in adjacency[current] if v != previous]
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
                        raise RuntimeError("skeleton path tracing did not terminate")

                if current not in zone:
                    continue

                target_component = component_of[current]
                points = coords[path].astype(float, copy=False)
                length = float(
                    np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                )
                graph.add_edge(
                    source_component,
                    target_component,
                    pts=points,
                    weight=length,
                )

    # Tiny self-loops wholly contained in the expanded junction zone are the
    # classical zero-loop artefact of voxel skeleton graph conversion.
    max_zero_loop_points = max(4, 2 * junction_hops + 3)
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        if u == v and len(data.get("pts", ())) <= max_zero_loop_points:
            graph.remove_edge(u, v, key)

    return graph


def _topology_score(graph: nx.MultiGraph, max_junction_degree: int) -> tuple[int, int, int]:
    """Score obvious junction artefacts without using ground-truth information."""
    probe = nx.MultiGraph(graph)

    # Degree-2 subdivisions are topologically immaterial.  Suppressing them in
    # the *score only* prevents a discretisation-dependent split from affecting
    # adaptive junction-zone selection while leaving the returned geometry
    # untouched for the package's normal simplification stage.
    while True:
        node = next((n for n, d in probe.degree() if d == 2), None)
        if node is None:
            break
        incident = list(probe.edges(node, keys=True))
        if len(incident) != 2:
            break
        _, a, _ = incident[0]
        _, b, _ = incident[1]
        if a == node or b == node:
            break
        probe.remove_node(node)
        probe.add_edge(a, b)

    degrees = [degree for _, degree in probe.degree()]
    over = sum(degree > max_junction_degree for degree in degrees)
    under = sum(0 < degree < min(3, max_junction_degree) for degree in degrees)
    # For the common trivalent case this detects residual split/zero-loop
    # defects without knowing the target graph.
    cubic_residual = abs(2 * probe.number_of_edges() - 3 * probe.number_of_nodes())
    return over, under, cubic_residual


def topology_aware_skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    junction_hops: int = 2,
    max_junction_degree: int | None = None,
    adaptive_extra_hops: int = 1,
) -> nx.MultiGraph:
    """Convert a 3-D skeleton image using sparse topology-aware junction zones.

    Parameters
    ----------
    skeleton_image
        Three-dimensional binary one-voxel skeleton.
    junction_hops
        Number of voxel-graph hops absorbed into every endpoint/junction zone.
        Two is a conservative default for Lee-thinned 3-D volumes.
    max_junction_degree
        Optional topology hint.  When supplied, one slightly larger junction
        zone is evaluated only if the first result contains an obvious degree
        defect, and the lower-defect result is returned.  ``3`` is appropriate
        for the degree-at-most-three spatial graphs used by Yamada validation.
    adaptive_extra_hops
        Number of additional graph hops considered by the adaptive retry.

    Returns
    -------
    networkx.MultiGraph
        Junction/end-point nodes and traced skeleton paths.
    """
    image = np.asarray(skeleton_image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")
    if junction_hops < 0 or adaptive_extra_hops < 0:
        raise ValueError("junction hop counts must be non-negative")

    coords, adjacency = _sparse_voxel_adjacency(image)
    first = _trace_zone_graph(coords, adjacency, junction_hops=junction_hops)
    if max_junction_degree is None or adaptive_extra_hops == 0:
        return first

    first_score = _topology_score(first, max_junction_degree)
    if first_score == (0, 0, 0):
        return first

    second = _trace_zone_graph(
        coords,
        adjacency,
        junction_hops=junction_hops + adaptive_extra_hops,
    )
    second_score = _topology_score(second, max_junction_degree)
    return second if second_score < first_score else first


def skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    backend: str = "poly2graph",
    junction_hops: int = 2,
    max_junction_degree: int | None = None,
) -> nx.MultiGraph:
    """Convert a skeletonized image into a ``networkx.MultiGraph``.

    ``backend='poly2graph'`` preserves the historical package behaviour.
    ``backend='topology_aware'`` selects the sparse junction-zone extractor,
    which is designed for robust 3-D volume-to-graph recovery and does not
    require ``poly2graph``.
    """
    if backend == "topology_aware":
        return topology_aware_skeleton_image_to_graph(
            skeleton_image,
            junction_hops=junction_hops,
            max_junction_degree=max_junction_degree,
        )
    if backend != "poly2graph":
        raise ValueError("backend must be 'poly2graph' or 'topology_aware'")

    from poly2graph import skeleton2graph

    return skeleton2graph(skeleton_image)
