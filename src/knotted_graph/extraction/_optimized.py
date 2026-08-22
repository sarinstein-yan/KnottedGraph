"""Production entry point for optimized sparse 3-D skeleton extraction."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ._sparse_trace import trace_zero_radius
from ._topology_optimized import constrained_persistent_extract

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _suppress_redundant_diagonal_shortcuts(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> list[list[int]]:
    """Remove digital diagonal shortcuts without changing connectivity.

    A 26-neighbour voxel graph contains edges that are not centreline steps at
    all.  For example, three occupied voxels forming an ordinary right-angle
    polyline are pairwise 26-neighbours, so the raw adjacency is a triangle and
    a tracer incorrectly sees a cycle.  A diagonal edge ``u-v`` is redundant
    when an occupied common neighbour ``w`` connects the same endpoints through
    two *strictly shorter* lattice steps.  Such an edge can be removed safely:
    it already has a replacement path, and recursively any removed replacement
    edge has still shorter steps, terminating at face-neighbour edges.

    Sparse diagonal connections with no shorter common-neighbour path are kept,
    so a genuinely diagonal digital curve remains connected.  This is a local,
    deterministic reduction of adjacency artefacts; it does not delete voxels or
    infer topology from the requested graph valence.
    """
    if not adjacency:
        return adjacency

    neighbours = [set(row) for row in adjacency]
    remove: set[tuple[int, int]] = set()

    for u, row in enumerate(adjacency):
        for v in row:
            if v <= u:
                continue
            delta = coords[v] - coords[u]
            edge_sq = int(np.dot(delta, delta))
            if edge_sq <= 1:
                continue

            for w in neighbours[u].intersection(neighbours[v]):
                uw = coords[w] - coords[u]
                wv = coords[v] - coords[w]
                if int(np.dot(uw, uw)) < edge_sq and int(np.dot(wv, wv)) < edge_sq:
                    remove.add((u, v))
                    break

    if not remove:
        return adjacency

    reduced: list[list[int]] = []
    for u, row in enumerate(adjacency):
        reduced.append(
            [
                v
                for v in row
                if (min(u, v), max(u, v)) not in remove
            ]
        )
    return reduced


def sparse_adjacency_exact_cropped(
    image: np.ndarray,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return deterministic, shortcut-reduced 26-neighbour lists."""
    occupied = [
        np.flatnonzero(image.any(axis=(1, 2))),
        np.flatnonzero(image.any(axis=(0, 2))),
        np.flatnonzero(image.any(axis=(0, 1))),
    ]
    if any(len(indices) == 0 for indices in occupied):
        return np.empty((0, 3), dtype=np.intp), []

    starts = np.asarray([int(indices[0]) for indices in occupied], dtype=np.intp)
    stops = np.asarray([int(indices[-1]) + 1 for indices in occupied], dtype=np.intp)
    crop = image[tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))]

    flat = np.flatnonzero(crop)
    shape = crop.shape
    local_coords = np.column_stack(np.unravel_index(flat, shape)).astype(np.intp, copy=False)
    coords = local_coords + starts
    strides = np.asarray((shape[1] * shape[2], shape[2], 1), dtype=np.int64)
    adjacency: list[list[int]] = [[] for _ in range(flat.size)]

    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        query = flat + dx * strides[0] + dy * strides[1] + dz
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
        actual = local_coords[right] - local_coords[left]
        keep = np.all(actual == wanted, axis=1)
        for u, v in zip(left[keep].tolist(), right[keep].tolist()):
            adjacency[u].append(v)

    adjacency = _suppress_redundant_diagonal_shortcuts(coords, adjacency)
    return coords, adjacency


def extract(
    image: np.ndarray,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Extract a 3-D embedded graph using the canonical sparse optimizer."""
    image = np.asarray(image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")
    if adaptive_max_hops < 0:
        raise ValueError("adaptive_max_hops must be non-negative")
    if max_junction_degree is not None and max_junction_degree < 1:
        raise ValueError("max_junction_degree must be positive")

    coords, adjacency = sparse_adjacency_exact_cropped(image)
    if max_junction_degree is None:
        return trace_zero_radius(coords, adjacency)
    return constrained_persistent_extract(
        coords,
        adjacency,
        max_degree=max_junction_degree,
        max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
