"""Production entry point for the second-generation sparse extractor."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ._sparse_compat import trace_zero_radius_compatible
from ._topology_optimized import constrained_persistent_extract

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def sparse_adjacency_exact_cropped(
    image: np.ndarray,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return exact-order 26-neighbour lists in the global voxel frame.

    Empty margins are removed before ``flatnonzero``/``searchsorted`` so sparse
    extraction scales with the occupied bounding box rather than the full image
    volume. Unlike the first cropped prototype, all 26 *directed* offsets are
    queried in lexicographic order. This directly emits each voxel's neighbour
    list in the same order as the historical 3x3x3 parser and therefore avoids
    a Python sort while preserving literal downstream graph output.
    """
    occupied = [
        np.flatnonzero(image.any(axis=(1, 2))),
        np.flatnonzero(image.any(axis=(0, 2))),
        np.flatnonzero(image.any(axis=(0, 1))),
    ]
    if any(len(indices) == 0 for indices in occupied):
        return np.empty((0, 3), dtype=np.intp), []

    starts = np.asarray([int(indices[0]) for indices in occupied], dtype=np.intp)
    stops = np.asarray([int(indices[-1]) + 1 for indices in occupied], dtype=np.intp)
    crop = image[
        tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    ]

    flat = np.flatnonzero(crop)
    shape = crop.shape
    local_coords = np.column_stack(np.unravel_index(flat, shape)).astype(
        np.intp,
        copy=False,
    )
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

    return coords, adjacency


def extract(
    image: np.ndarray,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Extract a 3-D embedded graph using the production sparse optimizer."""
    image = np.asarray(image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")
    if adaptive_max_hops < 0:
        raise ValueError("adaptive_max_hops must be non-negative")
    if max_junction_degree is not None and max_junction_degree < 1:
        raise ValueError("max_junction_degree must be positive")

    coords, adjacency = sparse_adjacency_exact_cropped(image)
    if max_junction_degree is None:
        return trace_zero_radius_compatible(coords, adjacency)

    return constrained_persistent_extract(
        coords,
        adjacency,
        max_degree=max_junction_degree,
        max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
