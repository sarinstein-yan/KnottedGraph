from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import networkx as nx
import numpy as np
from shapely import LineString, Point
from shapely.strtree import STRtree

from knotted_graph.extraction import skeletonize_volume
from knotted_graph.extraction._optimized import sparse_adjacency_exact_cropped
from knotted_graph.invariants.yamada import factorized_frontier as ff
from knotted_graph.invariants.yamada.state_compact import (
    PreparedCompactStateBuilder,
    _MINUS_PAIRS,
    _PLUS_PAIRS,
)
from knotted_graph.projection.pd_code import PDCode


BASELINE_SHA = "ebb0aae5d0e028aeaa2efd732a575ac1c68dd427"


def median_seconds(fn, *, repeats: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolution_tables(ordered_ports, port_count):
    plus_partner = [-1] * port_count
    minus_partner = [-1] * port_count
    for ports in ordered_ports:
        for a, b in _PLUS_PAIRS:
            pa, pb = ports[a], ports[b]
            plus_partner[pa] = pb
            plus_partner[pb] = pa
        for a, b in _MINUS_PAIRS:
            pa, pb = ports[a], ports[b]
            minus_partner[pa] = pb
            minus_partner[pb] = pa
    return tuple(plus_partner), tuple(minus_partner)


def random_prepared(seed: int, crossing_count: int) -> PreparedCompactStateBuilder:
    rng = random.Random(seed)
    vertex_count = 3
    fixed_port_count = 6
    crossing_ports = 4 * crossing_count
    port_count = crossing_ports + fixed_port_count
    ordered = tuple(
        tuple(range(4 * crossing, 4 * crossing + 4))
        for crossing in range(crossing_count)
    )
    fixed_terminal_index = [-1] * port_count
    crossing_for_port = [-1] * port_count
    for crossing, ports in enumerate(ordered):
        for port in ports:
            crossing_for_port[port] = crossing
    for port in range(crossing_ports, port_count):
        fixed_terminal_index[port] = (port - crossing_ports) % vertex_count

    ports = list(range(port_count))
    rng.shuffle(ports)
    arc_partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        arc_partner[left] = right
        arc_partner[right] = left
    plus, minus = _resolution_tables(ordered, port_count)
    return PreparedCompactStateBuilder(
        vertex_ids=tuple(range(vertex_count)),
        crossing_ids=tuple(range(crossing_count)),
        ordered_ports=ordered,
        arc_partner=tuple(arc_partner),
        fixed_terminal_index=tuple(fixed_terminal_index),
        crossing_for_port=tuple(crossing_for_port),
        plus_partner=plus,
        minus_partner=minus,
    )


def benchmark_yamada() -> dict:
    if not ff.native_factorized_available():
        raise RuntimeError(f"factorized backend unavailable: {ff.factorized_import_error()!r}")

    rows = []
    all_outputs = []
    for crossings in (4, 6, 8):
        cases = [
            random_prepared(821_000 + 101 * crossings + index, crossings)
            for index in range(4)
        ]
        outputs = [ff.compute_factorized_frontier_laurent(case) for case in cases]
        elapsed = median_seconds(
            lambda: [ff.compute_factorized_frontier_laurent(case) for case in cases],
            repeats=3,
            warmup=1,
        )
        rows.append(
            {
                "crossings": crossings,
                "cases": len(cases),
                "batch_ms": 1e3 * elapsed,
                "per_case_ms": 1e3 * elapsed / len(cases),
            }
        )
        all_outputs.append(outputs)
    return {"rows": rows, "fingerprint": digest_json(all_outputs)}


def projection_assignment_case(segments: int, crossing_stride: int):
    x = np.linspace(-3.0, 3.0, segments + 1)
    y = 0.36 * np.sin(2.4 * x) + 0.09 * np.sin(12.0 * x)
    z = 0.18 * np.cos(1.7 * x)
    edge = LineString(np.c_[x, y, z])
    indices = np.arange(7, len(x) - 7, crossing_stride)
    crossings = [Point(float(x[i]), float(y[i])) for i in indices]
    return edge, crossings


def normalized_incidences(values):
    return [(round(float(distance), 12), int(crossing_id)) for distance, crossing_id in values]


def make_projection_graph(samples: int = 480) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    slopes = (-1.18, -0.91, -0.63, -0.32, 0.05, 0.38, 0.72, 1.07)
    intercepts = (-0.22, 0.19, -0.08, 0.27, -0.31, 0.08, -0.15, 0.24)
    x = np.linspace(-1.0, 1.0, samples)
    for index, (slope, intercept) in enumerate(zip(slopes, intercepts, strict=True)):
        y = slope * x + intercept + 0.018 * np.sin((index + 2) * np.pi * x)
        z = -0.72 + 0.19 * index + 0.01 * x
        points = np.c_[x, y, z]
        u, v = f"u{index}", f"v{index}"
        graph.add_node(u, pos=points[0].copy())
        graph.add_node(v, pos=points[-1].copy())
        graph.add_edge(u, v, pts=points)
    return graph


def benchmark_projection() -> dict:
    rows = []
    fingerprints = []
    for segments, stride in ((600, 17), (1800, 13), (4200, 11)):
        edge, crossings = projection_assignment_case(segments, stride)
        tree = STRtree(crossings)
        output = PDCode._project_crossings_on_edge_indexed(
            edge, crossings, tree, tolerance=1e-8
        )
        elapsed = median_seconds(
            lambda: PDCode._project_crossings_on_edge_indexed(
                edge, crossings, tree, tolerance=1e-8
            ),
            repeats=5,
            warmup=1,
        )
        rows.append(
            {
                "segments": segments,
                "crossings": len(crossings),
                "incidences": len(output),
                "assignment_ms": 1e3 * elapsed,
            }
        )
        fingerprints.append(normalized_incidences(output))

    graph = make_projection_graph()
    pd = PDCode(graph).compute()
    full_elapsed = median_seconds(lambda: PDCode(graph).compute(), repeats=3, warmup=1)
    return {
        "rows": rows,
        "assignment_fingerprint": digest_json(fingerprints),
        "full_pd_ms": 1e3 * full_elapsed,
        "full_pd_fingerprint": hashlib.sha256(pd.encode()).hexdigest(),
        "full_pd_crossings": pd.count("X("),
    }


def random_walk_skeleton(seed: int, target_steps: int, shape=(144, 144, 144)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=bool)
    center = np.asarray(shape, dtype=int) // 2
    directions = np.asarray(
        [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1), (1, 1, 0), (1, 0, 1),
            (0, 1, 1), (-1, 1, 0), (0, -1, 1), (1, 0, -1),
        ],
        dtype=int,
    )
    walkers = [center.copy() for _ in range(12)]
    image[tuple(center)] = True
    for step in range(target_steps):
        index = step % len(walkers)
        if step and step % 251 == 0:
            occupied = np.argwhere(image)
            walkers[index] = occupied[int(rng.integers(0, len(occupied)))].copy()
        move = directions[int(rng.integers(0, len(directions)))]
        candidate = np.clip(walkers[index] + move, 3, np.asarray(shape) - 4)
        walkers[index] = candidate
        image[tuple(candidate)] = True
    return image


def adjacency_fingerprint(coords: np.ndarray, adjacency: list[list[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coords).tobytes())
    for row in adjacency:
        digest.update(np.asarray(row, dtype=np.int64).tobytes())
        digest.update(b";")
    return digest.hexdigest()


def make_thick_volume(size: int = 128) -> np.ndarray:
    from skimage.morphology import ball, dilation

    source = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    source[12 : c + 1, c, c] = True
    source[c, 12 : size - 12, c] = True
    source[c, c, 12 : size - 12] = True
    for i in range(18, size - 18):
        source[i, i, c] = True
    return dilation(source, footprint=ball(3))


def benchmark_skeleton() -> dict:
    rows = []
    fingerprints = []
    for steps in (1800, 4800, 9000):
        image = random_walk_skeleton(77_000 + steps, steps)
        coords, adjacency = sparse_adjacency_exact_cropped(image)
        elapsed = median_seconds(
            lambda: sparse_adjacency_exact_cropped(image), repeats=5, warmup=1
        )
        rows.append(
            {
                "requested_steps": steps,
                "occupied": int(np.count_nonzero(image)),
                "adjacency_ms": 1e3 * elapsed,
            }
        )
        fingerprints.append(adjacency_fingerprint(coords, adjacency))

    volume = make_thick_volume()
    skeleton = skeletonize_volume(volume)
    skeleton_elapsed = median_seconds(
        lambda: skeletonize_volume(volume), repeats=3, warmup=1
    )
    return {
        "rows": rows,
        "adjacency_fingerprint": digest_json(fingerprints),
        "skeletonize_ms": 1e3 * skeleton_elapsed,
        "skeleton_fingerprint": digest_array(skeleton),
        "volume_occupied": int(np.count_nonzero(volume)),
        "skeleton_occupied": int(np.count_nonzero(skeleton)),
    }


def crossing_only_prepared(seed: int, crossing_count: int) -> PreparedCompactStateBuilder:
    rng = random.Random(seed)
    port_count = 4 * crossing_count
    ordered = tuple(
        tuple(range(4 * crossing, 4 * crossing + 4))
        for crossing in range(crossing_count)
    )
    ports = list(range(port_count))
    rng.shuffle(ports)
    partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        partner[left] = right
        partner[right] = left
    crossing_for_port = [port // 4 for port in range(port_count)]
    plus, minus = _resolution_tables(ordered, port_count)
    return PreparedCompactStateBuilder(
        vertex_ids=(),
        crossing_ids=tuple(range(crossing_count)),
        ordered_ports=ordered,
        arc_partner=tuple(partner),
        fixed_terminal_index=tuple([-1] * port_count),
        crossing_for_port=tuple(crossing_for_port),
        plus_partner=plus,
        minus_partner=minus,
    )


def normalize_rii(value):
    if value is None:
        return None
    first, second, splices = value
    return int(first), int(second), tuple(tuple(int(x) for x in pair) for pair in splices)


def benchmark_rii() -> dict:
    rows = []
    outputs = []
    for crossings in (10, 20, 40, 80):
        cases = [
            crossing_only_prepared(93_000 + crossings * 17 + index, crossings)
            for index in range(12)
        ]
        values = [normalize_rii(case._find_reidemeister_ii_pair()) for case in cases]
        elapsed = median_seconds(
            lambda: [case._find_reidemeister_ii_pair() for case in cases],
            repeats=5,
            warmup=1,
        )
        rows.append(
            {
                "crossings": crossings,
                "cases": len(cases),
                "batch_ms": 1e3 * elapsed,
                "per_case_ms": 1e3 * elapsed / len(cases),
            }
        )
        outputs.append(values)
    return {"rows": rows, "fingerprint": digest_json(outputs)}


def run_benchmarks() -> dict:
    return {
        "baseline_sha": BASELINE_SHA,
        "yamada": benchmark_yamada(),
        "projection": benchmark_projection(),
        "skeleton": benchmark_skeleton(),
        "rii": benchmark_rii(),
    }


def compare_results(base: dict, optimized: dict) -> dict:
    checks = {
        "yamada_exact": base["yamada"]["fingerprint"] == optimized["yamada"]["fingerprint"],
        "projection_assignment_exact": base["projection"]["assignment_fingerprint"] == optimized["projection"]["assignment_fingerprint"],
        "projection_pd_exact": base["projection"]["full_pd_fingerprint"] == optimized["projection"]["full_pd_fingerprint"],
        "skeleton_adjacency_exact": base["skeleton"]["adjacency_fingerprint"] == optimized["skeleton"]["adjacency_fingerprint"],
        "skeletonization_exact": base["skeleton"]["skeleton_fingerprint"] == optimized["skeleton"]["skeleton_fingerprint"],
        "rii_exact": base["rii"]["fingerprint"] == optimized["rii"]["fingerprint"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"medium baseline equivalence failed: {failed}")

    def row_speedups(section: str, field: str):
        return [
            b[field] / o[field]
            for b, o in zip(base[section]["rows"], optimized[section]["rows"], strict=True)
        ]

    speedups = {
        "yamada_batch": row_speedups("yamada", "batch_ms"),
        "projection_assignment": row_speedups("projection", "assignment_ms"),
        "projection_full_pd": base["projection"]["full_pd_ms"] / optimized["projection"]["full_pd_ms"],
        "skeleton_adjacency": row_speedups("skeleton", "adjacency_ms"),
        "skeletonize": base["skeleton"]["skeletonize_ms"] / optimized["skeleton"]["skeletonize_ms"],
        "rii": row_speedups("rii", "batch_ms"),
    }
    # Evidence gates for the changes that were accepted specifically because of
    # measured performance. Use the two larger medium cases to reduce noise.
    if min(speedups["projection_assignment"][-2:]) < 1.5:
        raise AssertionError(
            f"projection batching lost its >=1.5x medium-scale gain: {speedups['projection_assignment']}"
        )
    if min(speedups["skeleton_adjacency"][-2:]) < 1.5:
        raise AssertionError(
            f"skeleton adjacency lost its >=1.5x medium-scale gain: {speedups['skeleton_adjacency']}"
        )
    if min(speedups["rii"][-2:]) < 1.5:
        raise AssertionError(f"RII indexing lost its >=1.5x scaling gain: {speedups['rii']}")

    return {"checks": checks, "speedups": speedups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "OPT"), type=Path)
    args = parser.parse_args()

    if args.compare:
        base = json.loads(args.compare[0].read_text())
        optimized = json.loads(args.compare[1].read_text())
        result = compare_results(base, optimized)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    result = run_benchmarks()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
