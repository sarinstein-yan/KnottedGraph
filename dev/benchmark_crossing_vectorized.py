from __future__ import annotations

import json
import statistics
import time

import numpy as np
import shapely
from shapely import MultiLineString
from shapely.strtree import STRtree

from benchmark_crossing_bulk_query import make_geometry
from knotted_graph.projection.pd_code import PDCode


def vectorized_crossings(multilines):
    segments = np.asarray(PDCode._explode_to_segments(multilines), dtype=object)
    tree = STRtree(segments)
    seen = set()
    pairs = tree.query(segments)
    left = pairs[0].astype(np.intp, copy=False)
    right = pairs[1].astype(np.intp, copy=False)
    mask = right > left
    left = left[mask]
    right = right[mask]
    if not len(left):
        return seen

    seg_a = segments[left]
    seg_b = segments[right]
    keep = ~shapely.touches(seg_a, seg_b)
    seg_a = seg_a[keep]
    seg_b = seg_b[keep]
    if not len(seg_a):
        return seen

    intersections = shapely.intersection(seg_a, seg_b)
    for inter in intersections:
        if inter.is_empty:
            continue
        gtype = inter.geom_type
        if gtype.startswith("Line") or gtype == "GeometryCollection":
            raise ValueError("Found overlapping (colinear) segments")
        if gtype == "Point":
            seen.add((inter.x, inter.y))
        elif gtype == "MultiPoint":
            for point in inter.geoms:
                seen.add((point.x, point.y))
    return seen


def timed(fn, repeats=5):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def main():
    for segment_count in (50, 100, 250, 500, 1000):
        geometry = make_geometry(segment_count, 20260817 + segment_count)
        legacy_time, legacy_points = timed(
            lambda: {
                (point.x, point.y) for point in PDCode._find_all_crossings(geometry)
            },
            repeats=3,
        )
        vector_time, vector_points = timed(
            lambda: vectorized_crossings(geometry),
            repeats=5,
        )
        if legacy_points != vector_points:
            raise AssertionError("vectorized Shapely path changed the crossing set")
        print(json.dumps({
            "segments": segment_count,
            "crossings": len(legacy_points),
            "legacy_s": legacy_time,
            "vectorized_s": vector_time,
            "speedup": legacy_time / vector_time,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
