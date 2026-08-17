from __future__ import annotations

import numpy as np
from shapely import LineString, Point
from shapely.strtree import STRtree

from knotted_graph.projection.pd_code import PDCode


def test_indexed_crossing_projection_matches_legacy_scan_randomized():
    rng = np.random.default_rng(20260817)

    for _ in range(100):
        count = int(rng.integers(2, 18))
        xyz = np.cumsum(rng.normal(size=(count, 3)), axis=0)
        edge = LineString(xyz.tolist())

        crossings = []
        # Generate points exactly on randomly selected segments, then append
        # unrelated points. This exercises ordering, duplicate suppression, and
        # the spatial-index candidate filter against the legacy O(S*C) oracle.
        for _ in range(int(rng.integers(0, 20))):
            segment_index = int(rng.integers(0, count - 1))
            t = float(rng.uniform(0.05, 0.95))
            a = xyz[segment_index]
            b = xyz[segment_index + 1]
            point = a + t * (b - a)
            crossings.append(Point(point))

        for _ in range(int(rng.integers(0, 20))):
            crossings.append(Point(rng.normal(size=3) * 20.0))

        expected = PDCode._project_crossings_on_edge(edge, crossings, tolerance=1e-8)
        if crossings:
            tree = STRtree(crossings)
            actual = PDCode._project_crossings_on_edge_indexed(
                edge,
                crossings,
                tree,
                tolerance=1e-8,
            )
        else:
            actual = []

        assert actual == expected


def test_indexed_crossing_projection_preserves_tolerance_boundary_behavior():
    edge = LineString([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)])
    tolerance = 1e-8
    crossings = [
        Point(2.0, 0.0, 0.0),
        Point(4.0, 0.5 * tolerance, 0.0),
        Point(6.0, 0.999 * tolerance, 0.0),
        Point(8.0, 1.001 * tolerance, 0.0),
    ]
    tree = STRtree(crossings)

    expected = PDCode._project_crossings_on_edge(edge, crossings, tolerance=tolerance)
    actual = PDCode._project_crossings_on_edge_indexed(
        edge,
        crossings,
        tree,
        tolerance=tolerance,
    )

    assert actual == expected
