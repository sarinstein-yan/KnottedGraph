from __future__ import annotations

import json
import statistics
import time

import numpy as np
from shapely import LineString, Point
from shapely.strtree import STRtree

from knotted_graph.projection.pd_code import PDCode


def timed(fn, repeats=5):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def case(segment_count: int, crossing_count: int):
    x = np.arange(segment_count + 1, dtype=float)
    xyz = np.column_stack((x, np.sin(x * 0.013), np.cos(x * 0.017)))
    edge = LineString(xyz.tolist())

    crossings = []
    for i in range(crossing_count // 2):
        segment_index = (i * 7919) % segment_count
        a = xyz[segment_index]
        b = xyz[segment_index + 1]
        crossings.append(Point(a + 0.37 * (b - a)))
    rng = np.random.default_rng(20260817 + segment_count + crossing_count)
    for _ in range(crossing_count - len(crossings)):
        crossings.append(Point(rng.normal(size=3) * segment_count * 3.0))

    tree = STRtree(crossings)
    legacy_time, legacy = timed(
        lambda: PDCode._project_crossings_on_edge(edge, crossings, tolerance=1e-8),
        repeats=3,
    )
    indexed_time, indexed = timed(
        lambda: PDCode._project_crossings_on_edge_indexed(
            edge, crossings, tree, tolerance=1e-8
        ),
        repeats=7,
    )
    if legacy != indexed:
        raise AssertionError("Indexed projection changed crossing assignments")

    return {
        "segments": segment_count,
        "crossings": crossing_count,
        "legacy_s": legacy_time,
        "indexed_s": indexed_time,
        "speedup": legacy_time / indexed_time,
        "intersections": len(indexed),
    }


def main():
    for segments, crossings in ((50, 50), (200, 200), (500, 500), (1000, 1000)):
        print(json.dumps(case(segments, crossings), separators=(",", ":")))


if __name__ == "__main__":
    main()
