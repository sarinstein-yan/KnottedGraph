import itertools

import numpy as np
from shapely import MultiLineString

from knotted_graph.projection.pd_code import (
    explode_to_segments,
    find_all_crossings,
)


def _coords(points):
    return {
        (round(point.x, 12), round(point.y, 12))
        for point in points
    }


def _pairwise_reference(lines):
    """Independent O(n^2) crossing finder used only as a test oracle."""
    segments = explode_to_segments(lines)
    seen = set()

    for i, j in itertools.combinations(range(len(segments)), 2):
        left = segments[i]
        right = segments[j]

        if left.touches(right):
            continue

        intersection = left.intersection(right)
        if intersection.is_empty:
            continue

        geometry_type = intersection.geom_type
        if geometry_type.startswith("Line") or geometry_type == "GeometryCollection":
            raise ValueError("Found overlapping (colinear) segments")

        if geometry_type == "Point":
            seen.add((intersection.x, intersection.y))
        elif geometry_type == "MultiPoint":
            for point in intersection.geoms:
                seen.add((point.x, point.y))

    return seen


def test_crossing_search_matches_independent_pairwise_reference():
    rng = np.random.default_rng(20260816)

    for _ in range(100):
        lines = []
        for _ in range(int(rng.integers(1, 6))):
            count = int(rng.integers(2, 7))
            xy = np.cumsum(
                rng.normal(size=(count, 2)),
                axis=0,
            )
            z = rng.normal(size=(count, 1))
            points = np.hstack([xy, z])
            points += rng.normal(
                scale=1e-7,
                size=points.shape,
            )
            lines.append(points.tolist())

        geometry = MultiLineString(lines)

        expected = {
            (round(x, 12), round(y, 12))
            for x, y in _pairwise_reference(geometry)
        }
        actual = _coords(
            find_all_crossings(geometry)
        )

        assert actual == expected


def test_crossing_search_handles_a_self_crossing_polyline():
    geometry = MultiLineString(
        [
            [
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, -1.0),
            ]
        ]
    )

    assert _coords(
        find_all_crossings(geometry)
    ) == {(0.5, 0.5)}
