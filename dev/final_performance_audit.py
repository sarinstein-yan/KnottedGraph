from __future__ import annotations

import json
import random
import statistics
import time
from contextlib import contextmanager

import networkx as nx
import numpy as np
import shapely
from shapely import LineString, Point
from shapely.strtree import STRtree

from knotted_graph.extraction import skeleton_image_to_graph, skeletonize_volume
from knotted_graph.extraction._optimized import sparse_adjacency_exact_cropped
from knotted_graph.extraction._topology_optimized import constrained_persistent_extract
from knotted_graph.invariants.yamada import factorized_frontier as ff
from knotted_graph.invariants.yamada.state_compact import (
    PreparedCompactStateBuilder,
    _MINUS_PAIRS,
    _PLUS_PAIRS,
)
from knotted_graph.projection.pd_code import PDCode


YAMADA_KEYS = (
    "factor_types",
    "port_factor",
    "wire_partner",
    "wire_type",
    "plus_partner",
    "minus_partner",
    "factor_order",
)


def median_seconds(fn, *, repeats: int = 7, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


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


def random_prepared(seed: int, crossing_count: int):
    rng = random.Random(seed)
    vertex_count = rng.randint(1, 4)
    fixed_port_count = 2 * rng.randint(1, 4)
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
        fixed_terminal_index[port] = rng.randrange(vertex_count)

    ports = list(range(port_count))
    rng.shuffle(ports)
    arc_partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        arc_partner[left] = right
        arc_partner[right] = left

    plus, minus = _resolution_tables(ordered, port_count)
    return PreparedCompactStateBuilder(
        vertex_ids=tuple(f"v{index}" for index in range(vertex_count)),
        crossing_ids=tuple(f"c{index}" for index in range(crossing_count)),
        ordered_ports=ordered,
        arc_partner=tuple(arc_partner),
        fixed_terminal_index=tuple(fixed_terminal_index),
        crossing_for_port=tuple(crossing_for_port),
        plus_partner=plus,
        minus_partner=minus,
    )


def audit_yamada():
    if not ff.native_factorized_available():
        raise RuntimeError(f"factorized native backend unavailable: {ff.factorized_import_error()!r}")

    native = ff._yamada_factorized_frontier.compute_factorized_frontier
    rows = []
    for crossings in (1, 3, 5, 7):
        prepared = [
            random_prepared(90210 + 113 * crossings + offset, crossings)
            for offset in range(3)
        ]
        data = [ff.build_factorized_frontier(case) for case in prepared]
        list_args = [[list(item[key]) for key in YAMADA_KEYS] for item in data]
        tuple_args = [[item[key] for key in YAMADA_KEYS] for item in data]

        expected = [native(*args) for args in list_args]
        tuple_supported = True
        try:
            direct_tuple = [native(*args) for args in tuple_args]
        except TypeError:
            tuple_supported = False
            direct_tuple = None
        if tuple_supported and direct_tuple != expected:
            raise AssertionError("tuple-direct pybind call changed Yamada output")

        build_s = median_seconds(
            lambda: [ff.build_factorized_frontier(case) for case in prepared]
        )
        list_conversion_s = median_seconds(
            lambda: [[list(item[key]) for key in YAMADA_KEYS] for item in data]
        )
        native_list_s = median_seconds(lambda: [native(*args) for args in list_args])
        tuple_native_s = (
            median_seconds(lambda: [native(*args) for args in tuple_args])
            if tuple_supported
            else None
        )
        raw = expected
        output_conversion_s = median_seconds(
            lambda: [
                tuple((int(power), int(coefficient)) for power, coefficient in value)
                for value in raw
            ]
        )
        total_s = median_seconds(
            lambda: [ff.compute_factorized_frontier_laurent(case) for case in prepared]
        )
        rows.append(
            {
                "crossings": crossings,
                "batch_size": len(prepared),
                "build_ms": 1e3 * build_s,
                "explicit_list_conversion_ms": 1e3 * list_conversion_s,
                "native_with_lists_ms": 1e3 * native_list_s,
                "native_with_tuples_ms": None if tuple_native_s is None else 1e3 * tuple_native_s,
                "output_conversion_ms": 1e3 * output_conversion_s,
                "production_total_ms": 1e3 * total_s,
                "preprocess_fraction": build_s / total_s if total_s else 0.0,
                "explicit_list_fraction": list_conversion_s / total_s if total_s else 0.0,
            }
        )

    overflow_count = 0
    overflow_trials = 0
    for crossings in (7, 8, 9):
        case = random_prepared(100000 + crossings, crossings)
        data = ff.build_factorized_frontier(case)
        args = [list(data[key]) for key in YAMADA_KEYS]
        overflow_trials += 1
        try:
            native(*args)
        except OverflowError:
            overflow_count += 1

    return {
        "stage_rows": rows,
        "native_overflows": overflow_count,
        "overflow_trials": overflow_trials,
    }


def project_crossings_batched(
    edge: LineString,
    crossings: list[Point],
    crossing_tree: STRtree,
    tolerance: float = 1e-8,
):
    if not crossings:
        return []

    coords = np.asarray(edge.coords, dtype=float)
    if len(coords) < 2:
        return []
    segments = np.asarray(
        [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)],
        dtype=object,
    )
    pairs = crossing_tree.query(
        segments,
        predicate="dwithin",
        distance=float(tolerance),
    )
    if pairs.size == 0:
        return []

    seg_ids = pairs[0].astype(np.intp, copy=False)
    crossing_ids = pairs[1].astype(np.intp, copy=False)
    crossing_array = np.asarray(crossings, dtype=object)
    seg_candidates = segments[seg_ids]
    point_candidates = crossing_array[crossing_ids]

    distances = shapely.distance(seg_candidates, point_candidates)
    keep = np.asarray(distances < tolerance)
    if not np.any(keep):
        return []

    seg_ids = seg_ids[keep]
    crossing_ids = crossing_ids[keep]
    seg_candidates = seg_candidates[keep]
    point_candidates = point_candidates[keep]
    local = np.asarray(
        shapely.line_locate_point(seg_candidates, point_candidates),
        dtype=float,
    )

    lengths = np.asarray(shapely.length(segments), dtype=float)
    starts = np.empty(len(lengths), dtype=float)
    starts[0] = 0.0
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])

    intersections = [
        (float(starts[int(seg_id)] + local_distance), int(crossing_id))
        for seg_id, crossing_id, local_distance in zip(
            seg_ids.tolist(),
            crossing_ids.tolist(),
            local.tolist(),
            strict=True,
        )
    ]
    return PDCode._deduplicate_crossing_distances(
        intersections,
        tolerance=tolerance,
    )


def projection_assignment_cases():
    x = np.linspace(-2.0, 2.0, 1201)
    y = 0.4 * np.sin(3.0 * x) + 0.12 * np.sin(11.0 * x)
    z = 0.2 * np.cos(2.0 * x)
    edge = LineString(np.c_[x, y, z])
    sample_ids = np.arange(30, len(x) - 30, 13)
    crossings = [Point(float(x[i]), float(y[i])) for i in sample_ids]

    t = np.linspace(0.2, 2.0 * np.pi + 0.2, 1601)
    edge_self = LineString(np.c_[np.sin(t), np.sin(2.0 * t), 0.2 * np.cos(t)])
    repeated = [Point(0.0, 0.0)]
    return [(edge, crossings), (edge_self, repeated)]


def make_projection_graph(samples: int = 260) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    slopes = (-1.10, -0.77, -0.43, -0.08, 0.31, 0.66, 1.03)
    intercepts = (-0.31, 0.18, -0.07, 0.27, -0.22, 0.06, -0.13)
    x = np.linspace(-1.0, 1.0, samples)
    for index, (slope, intercept) in enumerate(zip(slopes, intercepts, strict=True)):
        y = slope * x + intercept + 0.025 * np.sin((index + 2) * np.pi * x)
        z = (-0.55 + 0.18 * index) + 0.015 * x
        points = np.c_[x, y, z]
        u = f"u{index}"
        v = f"v{index}"
        graph.add_node(u, pos=points[0].copy())
        graph.add_node(v, pos=points[-1].copy())
        graph.add_edge(u, v, pts=points)
    return graph


@contextmanager
def patched_projection_assignment(fn):
    original = PDCode._project_crossings_on_edge_indexed
    PDCode._project_crossings_on_edge_indexed = staticmethod(fn)
    try:
        yield
    finally:
        PDCode._project_crossings_on_edge_indexed = original


def audit_projection():
    assignment_rows = []
    for index, (edge, crossings) in enumerate(projection_assignment_cases()):
        tree = STRtree(crossings)
        current = PDCode._project_crossings_on_edge_indexed(edge, crossings, tree)
        candidate = project_crossings_batched(edge, crossings, tree)
        if current != candidate:
            raise AssertionError(
                f"batched crossing projection mismatch in case {index}: "
                f"{len(current)} != {len(candidate)}"
            )
        current_s = median_seconds(
            lambda: PDCode._project_crossings_on_edge_indexed(edge, crossings, tree),
            repeats=5,
            warmup=1,
        )
        candidate_s = median_seconds(
            lambda: project_crossings_batched(edge, crossings, tree),
            repeats=5,
            warmup=1,
        )
        assignment_rows.append(
            {
                "case": index,
                "segments": len(edge.coords) - 1,
                "crossings": len(crossings),
                "incidences": len(current),
                "current_ms": 1e3 * current_s,
                "batched_ms": 1e3 * candidate_s,
                "speedup": current_s / candidate_s if candidate_s else float("inf"),
            }
        )

    graph = make_projection_graph()
    baseline_pd = PDCode(graph).compute()
    with patched_projection_assignment(project_crossings_batched):
        candidate_pd = PDCode(graph).compute()
    if candidate_pd != baseline_pd:
        raise AssertionError("batched crossing projection changed the full PD code")

    baseline_s = median_seconds(lambda: PDCode(graph).compute(), repeats=3, warmup=1)
    with patched_projection_assignment(project_crossings_batched):
        candidate_s = median_seconds(lambda: PDCode(graph).compute(), repeats=3, warmup=1)

    return {
        "assignment_rows": assignment_rows,
        "full_pd_crossings": baseline_pd.count("X("),
        "full_current_ms": 1e3 * baseline_s,
        "full_batched_ms": 1e3 * candidate_s,
        "full_speedup": baseline_s / candidate_s if candidate_s else float("inf"),
    }


def make_small_junction_skeleton(size: int = 72) -> np.ndarray:
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    image[8 : c + 1, c, c] = True
    image[c, 8 : c + 1, c] = True
    image[c, c, c : size - 8] = True
    return image


def audit_skeleton():
    from skimage.morphology import ball, dilation

    source = make_small_junction_skeleton()
    volume = dilation(source, footprint=ball(2))
    skeleton = skeletonize_volume(volume)

    skeletonize_s = median_seconds(
        lambda: skeletonize_volume(volume),
        repeats=3,
        warmup=1,
    )
    adjacency_s = median_seconds(
        lambda: sparse_adjacency_exact_cropped(skeleton),
        repeats=7,
        warmup=2,
    )
    coords, adjacency = sparse_adjacency_exact_cropped(skeleton)
    topology_s = median_seconds(
        lambda: constrained_persistent_extract(
            coords,
            adjacency,
            max_degree=3,
            max_hops=4,
            anomaly_ratio=0.15,
        ),
        repeats=5,
        warmup=1,
    )
    total_s = median_seconds(
        lambda: skeleton_image_to_graph(skeleton, max_junction_degree=3),
        repeats=5,
        warmup=1,
    )
    graph = skeleton_image_to_graph(skeleton, max_junction_degree=3)

    return {
        "occupied_volume_voxels": int(np.count_nonzero(volume)),
        "skeleton_voxels": int(np.count_nonzero(skeleton)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "skeletonize_ms": 1e3 * skeletonize_s,
        "adjacency_ms": 1e3 * adjacency_s,
        "topology_ms": 1e3 * topology_s,
        "extract_total_ms": 1e3 * total_s,
        "adjacency_fraction_of_extract": adjacency_s / total_s if total_s else 0.0,
        "topology_fraction_of_extract": topology_s / total_s if total_s else 0.0,
    }


def main():
    result = {
        "yamada": audit_yamada(),
        "projection": audit_projection(),
        "skeleton": audit_skeleton(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
