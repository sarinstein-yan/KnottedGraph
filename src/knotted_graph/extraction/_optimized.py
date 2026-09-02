"""Production entry point for optimized sparse 3-D skeleton extraction."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ._topology_optimized import persistent_extract

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _shorter_intermediate_offsets(
    delta: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Return local intermediates that make a lattice edge redundant."""
    edge_sq = sum(value * value for value in delta)
    if edge_sq <= 1:
        return ()

    result: list[tuple[int, int, int]] = []
    for step in _NEIGHBOR_OFFSETS:
        remainder = tuple(delta[i] - step[i] for i in range(3))
        if remainder == (0, 0, 0) or any(abs(value) > 1 for value in remainder):
            continue
        step_sq = sum(value * value for value in step)
        remainder_sq = sum(value * value for value in remainder)
        if step_sq < edge_sq and remainder_sq < edge_sq:
            result.append(step)
    return tuple(result)


_SHORTER_INTERMEDIATES = {
    offset: _shorter_intermediate_offsets(offset)
    for offset in _NEIGHBOR_OFFSETS
}


def _occupied_at_offset(
    flat: np.ndarray,
    local_coords: np.ndarray,
    strides: np.ndarray,
    left: np.ndarray,
    offset: tuple[int, int, int],
) -> np.ndarray:
    """Vectorized occupancy query relative to each index in ``left``."""
    if left.size == 0:
        return np.zeros(0, dtype=bool)

    wanted = np.asarray(offset, dtype=np.intp)
    query = (
        flat[left]
        + int(wanted[0]) * strides[0]
        + int(wanted[1]) * strides[1]
        + int(wanted[2])
    )
    positions = np.searchsorted(flat, query)
    valid = positions < flat.size
    found = np.zeros(left.size, dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return found

    matched_positions = positions[valid]
    exact_flat = flat[matched_positions] == query[valid]
    valid_indices = valid_indices[exact_flat]
    matched_positions = matched_positions[exact_flat]
    if valid_indices.size == 0:
        return found

    exact_offset = np.all(
        local_coords[matched_positions] - local_coords[left[valid_indices]] == wanted,
        axis=1,
    )
    found[valid_indices[exact_offset]] = True
    return found


def _suppress_redundant_diagonal_shortcuts(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> list[list[int]]:
    """Reference diagonal-shortcut reducer retained for tests/diagnostics."""
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

    for offset in _NEIGHBOR_OFFSETS:
        wanted = np.asarray(offset, dtype=np.intp)
        query = (
            flat
            + int(wanted[0]) * strides[0]
            + int(wanted[1]) * strides[1]
            + int(wanted[2])
        )
        positions = np.searchsorted(flat, query)
        valid = positions < flat.size
        left = np.flatnonzero(valid)
        right = positions[valid]
        exact = flat[right] == query[valid]
        left = left[exact]
        right = right[exact]
        if left.size == 0:
            continue

        actual = local_coords[right] - local_coords[left]
        exact_offset = np.all(actual == wanted, axis=1)
        left = left[exact_offset]
        right = right[exact_offset]
        if left.size == 0:
            continue

        intermediates = _SHORTER_INTERMEDIATES[offset]
        if intermediates:
            redundant = np.zeros(left.size, dtype=bool)
            for intermediate in intermediates:
                redundant |= _occupied_at_offset(
                    flat,
                    local_coords,
                    strides,
                    left,
                    intermediate,
                )
                if np.all(redundant):
                    break
            left = left[~redundant]
            right = right[~redundant]

        for u, v in zip(left.tolist(), right.tolist()):
            adjacency[u].append(v)

    return coords, adjacency


def extract(
    image: np.ndarray,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Extract a 3-D embedded graph with multi-scale topology selection."""
    image = np.asarray(image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")
    if adaptive_max_hops < 0:
        raise ValueError("adaptive_max_hops must be non-negative")
    if max_junction_degree is not None and max_junction_degree < 1:
        raise ValueError("max_junction_degree must be positive")

    coords, adjacency = sparse_adjacency_exact_cropped(image)
    return persistent_extract(
        coords,
        adjacency,
        max_degree=max_junction_degree,
        max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
