from __future__ import annotations

import numpy as np
import pytest

from knotted_graph.extraction.skeleton import _occupied_bounds


def test_occupied_bounds_matches_argwhere_reference_randomized():
    rng = np.random.default_rng(20260822)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(4, 30, size=3))
        image = rng.random(shape) < float(rng.uniform(0.001, 0.4))
        if not image.any():
            image[tuple(int(value) for value in rng.integers(0, shape))] = True

        occupied = np.argwhere(image)
        expected_starts = occupied.min(axis=0)
        expected_stops = occupied.max(axis=0) + 1
        starts, stops = _occupied_bounds(image)

        assert np.array_equal(starts, expected_starts)
        assert np.array_equal(stops, expected_stops)


def test_occupied_bounds_rejects_empty_volume():
    with pytest.raises(ValueError, match="does not contain any True voxels"):
        _occupied_bounds(np.zeros((5, 6, 7), dtype=bool))
