from __future__ import annotations

import json
import statistics
import time

import numpy as np
from shapely import MultiLineString
from shapely.strtree import STRtree

from knotted_graph.projection.pd_code import PDCode


def bulk_reference(multilines):
    segments = PDCode._explode_to_segments(multilines)
    tree = STRtree(segments)
    seen = set()
    pairs = tree.query(segments)
    candidates = sorted(
        ((int(i), int(j)) for i, j in zip(pairs[0], pairs[1]) if int(j) > int(i)),
        key=lambda pair: pair,
    )
    for i, j in candidates:
        seg = segments[i]
        other = segments[j]
        if seg.touches(other):
            continue
        inter = seg.intersection(other)
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


def make_geometry(segment_count, seed):
    rng = np.random.default_rng(seed)
    # Many short random chords in a common box create realistic STRtree
    # candidate pressure without overlapping colinear segments.
    lines = []
    for _ in range(segment_count):
        start = rng.uniform(-1.0, 1.0, size=3)
        end = rng.uniform(-1.0, 1.0, size=3)
        lines.append([start.tolist(), end.tolist()])
    return MultiLineString(lines)


def main():
    for segment_count in (50, 100, 250, 500):
        geometry = make_geometry(segment_count, 20260817 + segment_count)
        legacy_time, legacy = timed(lambda: {
            (point.x, point.y) for point in PDCode._find_all_crossings(geometry)
        }, repeats=3)
        bulk_time, bulk = timed(lambda: bulk_reference(geometry), repeats=5)
        if legacy != bulk:
            raise AssertionError("bulk STRtree query changed the crossing set")
        print(json.dumps({
            "segments": segment_count,
            "crossings": len(legacy),
            "legacy_s": legacy_time,
            "bulk_s": bulk_time,
            "speedup": legacy_time / bulk_time,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
