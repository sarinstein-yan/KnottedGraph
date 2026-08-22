from __future__ import annotations

import numpy as np

from knotted_graph.extraction._optimized import (
    _NEIGHBOR_OFFSETS,
    _suppress_redundant_diagonal_shortcuts,
    sparse_adjacency_exact_cropped,
)


def _raw_sparse_adjacency(image: np.ndarray):
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
    adjacency = [[] for _ in range(flat.size)]

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


def test_integrated_shortcut_suppression_matches_reference_randomized():
    rng = np.random.default_rng(20260822)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(5, 16, size=3))
        density = float(rng.uniform(0.02, 0.30))
        image = rng.random(shape) < density
        if not image.any():
            continue

        expected_coords, raw = _raw_sparse_adjacency(image)
        expected = _suppress_redundant_diagonal_shortcuts(expected_coords, raw)
        actual_coords, actual = sparse_adjacency_exact_cropped(image)

        assert np.array_equal(actual_coords, expected_coords)
        assert actual == expected


def test_integrated_shortcut_suppression_keeps_true_diagonal_chain():
    image = np.zeros((7, 7, 7), dtype=bool)
    for index in range(1, 6):
        image[index, index, index] = True

    coords, adjacency = sparse_adjacency_exact_cropped(image)

    assert len(coords) == 5
    assert [len(row) for row in adjacency] == [1, 2, 2, 2, 1]
