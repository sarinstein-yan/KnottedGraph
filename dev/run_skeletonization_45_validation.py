from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp
from skimage.morphology import ball, dilation, skeletonize

from knotted_graph.core import (
    contract_short_edges,
    remove_leaf_nodes,
    simplify_edges,
    smooth_edges,
)
from knotted_graph.extraction import skeleton_image_to_graph
from knotted_graph.invariants.yamada.native import native_available, native_import_error
from knotted_graph.projection import compute_yamada_polynomial


@dataclass
class Case:
    name: str
    graph: nx.MultiGraph
    radius_cap: float


A = sp.Symbol("A")
BOUND = 1.35
N = 200
RADII = [1, 2, 3]
TRANSFORMS = ["identity", "rotate", "affine"]
CLEARANCE_FRACTION = 0.40
DX = 2 * BOUND / (N - 1)


def normalize(points, scale=0.72):
    points = np.asarray(points, dtype=float)
    points -= points.mean(axis=0)
    return points * (scale / np.max(np.linalg.norm(points, axis=1)))


def embedded_graph(positions, edges):
    graph = nx.MultiGraph()
    for node, point in positions.items():
        graph.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v, points in edges:
        graph.add_edge(u, v, pts=np.asarray(points, dtype=float))
    return graph


def theta_case(name, bowed=False, samples=500):
    t = np.linspace(0, 1, samples)
    x = -0.72 + 1.44 * t
    if bowed:
        curves = [
            np.c_[x, -0.58 * np.sin(np.pi * t), 0.16 * np.sin(2 * np.pi * t)],
            np.c_[x, 0.10 * np.sin(2 * np.pi * t), -0.10 * np.sin(np.pi * t)],
            np.c_[x, 0.58 * np.sin(np.pi * t), -0.16 * np.sin(2 * np.pi * t)],
        ]
    else:
        curves = [
            np.c_[x, -0.58 * np.sin(np.pi * t), 0 * t],
            np.c_[x, 0 * t, 0 * t],
            np.c_[x, 0.58 * np.sin(np.pi * t), 0 * t],
        ]
    for points in curves:
        points[0] = [-0.72, 0, 0]
        points[-1] = [0.72, 0, 0]
    return Case(
        name,
        embedded_graph(
            {"u": curves[0][0], "v": curves[0][-1]},
            [("u", "v", points) for points in curves],
        ),
        0.060,
    )


def segment_distance(p1, q1, p2, q2):
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    determinant = a * c - b * b
    if determinant < 1e-14:
        s = 0.0
        t = np.clip(e / c if c > 1e-14 else 0.0, 0, 1)
    else:
        s = np.clip((b * e - c * d) / determinant, 0, 1)
        t = np.clip((a * e - b * d) / determinant, 0, 1)
    if a > 1e-14:
        s = np.clip((b * t - d) / a, 0, 1)
    if c > 1e-14:
        t = np.clip((b * s + e) / c, 0, 1)
    return float(np.linalg.norm(w + s * u - t * v))


def straight_clearance(graph, positions):
    edges = list(graph.edges())
    best = np.inf
    for index, (u, v) in enumerate(edges):
        for a, b in edges[index + 1 :]:
            if {u, v} & {a, b}:
                continue
            best = min(
                best,
                segment_distance(
                    positions[u],
                    positions[v],
                    positions[a],
                    positions[b],
                ),
            )
    return best


def cubic_case(name, graph, planar, seed, cap):
    graph = nx.Graph(graph)
    assert nx.is_connected(graph)
    assert not list(nx.bridges(graph))
    assert all(degree == 3 for _, degree in graph.degree())

    if planar:
        ok, _ = nx.check_planarity(graph)
        assert ok
        layout = nx.planar_layout(graph)
        xyz = normalize([[layout[node][0], layout[node][1], 0.0] for node in graph])
        positions = {node: xyz[index] for index, node in enumerate(graph)}
    else:
        positions = None
        for trial in range(250):
            layout = nx.spring_layout(
                graph,
                dim=3,
                seed=seed + trial,
                iterations=700,
            )
            xyz = normalize([layout[node] for node in graph])
            candidate = {node: xyz[index] for index, node in enumerate(graph)}
            if straight_clearance(graph, candidate) > 0.055:
                positions = candidate
                break
        if positions is None:
            raise RuntimeError(f"Could not find clear embedding for {name}")

    return Case(
        name,
        embedded_graph(
            positions,
            [
                (u, v, np.linspace(positions[u], positions[v], 100))
                for u, v in graph.edges()
            ],
        ),
        cap,
    )


def fixed_spring_case(name, graph, seed, cap=0.060):
    graph = nx.Graph(graph)
    assert nx.is_connected(graph)
    assert not list(nx.bridges(graph))
    assert all(degree == 3 for _, degree in graph.degree())
    layout = nx.spring_layout(graph, dim=3, seed=seed, iterations=700)
    xyz = normalize([layout[node] for node in graph])
    positions = {node: xyz[index] for index, node in enumerate(graph)}
    clearance = straight_clearance(graph, positions)
    if clearance <= 0.055:
        raise RuntimeError(f"Insufficient deterministic clearance for {name}: {clearance}")
    return Case(
        name,
        embedded_graph(
            positions,
            [
                (u, v, np.linspace(positions[u], positions[v], 100))
                for u, v in graph.edges()
            ],
        ),
        cap,
    )


ORIGINAL_CASES = [
    theta_case("theta3_planar"),
    theta_case("theta3_bowed", True),
    cubic_case("K4", nx.complete_graph(4), True, 11, 0.052),
    cubic_case("triangular_prism", nx.circular_ladder_graph(3), True, 12, 0.045),
    cubic_case("cube", nx.cubical_graph(), True, 13, 0.042),
    cubic_case("pentagonal_prism", nx.circular_ladder_graph(5), True, 14, 0.035),
    cubic_case("dodecahedral", nx.dodecahedral_graph(), True, 15, 0.027),
    cubic_case("K3_3", nx.complete_bipartite_graph(3, 3), False, 17, 0.032),
    cubic_case("petersen", nx.petersen_graph(), False, 18, 0.030),
    cubic_case("heawood", nx.heawood_graph(), False, 19, 0.024),
]

# Seven genuinely new graph families. Fixed seeds were selected solely for clear
# non-self-intersecting embeddings, not for extractor output.
CHALLENGE_CASES = [
    fixed_spring_case("frucht", nx.frucht_graph(), 702),
    fixed_spring_case("moebius_kantor", nx.moebius_kantor_graph(), 710),
    fixed_spring_case("desargues", nx.desargues_graph(), 706),
    fixed_spring_case("pappus", nx.pappus_graph(), 720),
    fixed_spring_case("truncated_tetrahedron", nx.truncated_tetrahedron_graph(), 722),
    fixed_spring_case("truncated_cube", nx.truncated_cube_graph(), 715),
    fixed_spring_case("tutte", nx.tutte_graph(), 712),
]


def rotation_xyz(a, b, c):
    a, b, c = np.deg2rad([a, b, c])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(name):
    if name == "identity":
        matrix, offset = np.eye(3), np.zeros(3)
    elif name == "rotate":
        matrix, offset = rotation_xyz(21, 34, 13), np.array([0.04, -0.03, 0.02])
    elif name == "affine":
        matrix = (
            rotation_xyz(17, -23, 31)
            @ np.diag([1.08, 0.91, 1.03])
            @ np.array([[1, 0.13, 0], [0, 1, 0.09], [0.05, 0, 1]])
        )
        offset = np.array([-0.03, 0.04, -0.02])
    else:
        raise ValueError(name)
    assert np.linalg.det(matrix) > 0
    return matrix, offset


def deform(graph, name):
    matrix, offset = transform(name)
    result = nx.MultiGraph()
    for node, data in graph.nodes(data=True):
        result.add_node(node, pos=np.asarray(data["pos"]) @ matrix.T + offset)
    for u, v, key, data in graph.edges(keys=True, data=True):
        result.add_edge(u, v, pts=np.asarray(data["pts"]) @ matrix.T + offset)
    return result


def trimmed(points, fraction=0.15):
    points = np.asarray(points, dtype=float)
    count = max(1, int(round(fraction * len(points))))
    return points[count:-count] if 2 * count < len(points) else points


def interior_separation(graph):
    edges = [
        (u, v, np.asarray(data["pts"], dtype=float))
        for u, v, _, data in graph.edges(keys=True, data=True)
    ]
    best = np.inf
    for index, (u, v, p0) in enumerate(edges):
        for a, b, q0 in edges[index + 1 :]:
            p, q = (trimmed(p0), trimmed(q0)) if {u, v} & {a, b} else (p0, q0)
            for start in range(0, len(p), 128):
                distances2 = np.sum((p[start : start + 128, None, :] - q[None, :, :]) ** 2, axis=-1)
                best = min(best, float(np.sqrt(distances2.min())))
    return best


def admissible(case, graph, radius):
    physical_radius = radius * DX
    separation = interior_separation(graph)
    limit = min(case.radius_cap, CLEARANCE_FRACTION * separation)
    return physical_radius <= limit


def resample(points, step):
    points = np.asarray(points, dtype=float)
    parts = []
    for p, q in zip(points[:-1], points[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(q - p) / step)) + 1)
        parts.append(np.linspace(p, q, count, endpoint=False))
    parts.append(points[-1:])
    return np.vstack(parts)


def voxelize(graph, radius):
    volume = np.zeros((N, N, N), dtype=bool)
    for _, _, _, data in graph.edges(keys=True, data=True):
        points = resample(data["pts"], DX / 3)
        indices = np.rint((points + BOUND) / (2 * BOUND) * (N - 1)).astype(int)
        indices = np.clip(indices, 0, N - 1)
        volume[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return dilation(volume, footprint=ball(radius))


def to_world(graph):
    graph = nx.MultiGraph(graph)
    origin = np.array([-BOUND] * 3, dtype=float)
    for _, data in graph.nodes(data=True):
        data["pos"] = origin + DX * np.asarray(data["pos"], dtype=float)
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["pts"] = origin + DX * np.asarray(data["pts"], dtype=float)
    return graph


def cleanup_baseline(graph):
    graph = remove_leaf_nodes(graph)
    graph = simplify_edges(graph)
    graph = contract_short_edges(graph, min_length=2.5 * DX, copy=False)
    graph = remove_leaf_nodes(graph)
    graph = simplify_edges(graph)
    return smooth_edges(graph, epsilon=2 * DX, copy=False)


def cleanup_optimized(graph):
    graph = remove_leaf_nodes(graph)
    graph = simplify_edges(graph)
    return smooth_edges(graph, epsilon=2 * DX, copy=False)


def run_reconstruction_suite():
    rows = []

    # Preserve the original admissibility logic exactly: this contributes 38.
    for case in ORIGINAL_CASES:
        for transform_name in TRANSFORMS:
            target = deform(case.graph, transform_name)
            for radius in RADII:
                if not admissible(case, target, radius):
                    continue
                rows.append((case, transform_name, radius, target, "original"))

    original_count = len(rows)
    assert original_count == 38, original_count

    # Add one fixed affine/r=2 reconstruction for each new graph family.
    for case in CHALLENGE_CASES:
        target = deform(case.graph, "affine")
        assert admissible(case, target, 2), case.name
        rows.append((case, "affine", 2, target, "challenge"))

    assert len(rows) == 45, len(rows)

    records = []
    for index, (case, transform_name, radius, target, group) in enumerate(rows, 1):
        volume = voxelize(target, radius)
        start = time.perf_counter()
        skeleton = skeletonize(volume, method="lee")
        skeleton_time = time.perf_counter() - start

        baseline_times = []
        optimized_times = []
        for _ in range(3):
            start = time.perf_counter()
            baseline = skeleton_image_to_graph(skeleton, backend="poly2graph")
            baseline_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            optimized = skeleton_image_to_graph(
                skeleton,
                backend="topology_aware",
                max_junction_degree=3,
            )
            optimized_times.append(time.perf_counter() - start)

        baseline_clean = cleanup_baseline(to_world(baseline))
        optimized_clean = cleanup_optimized(to_world(optimized))
        target_abstract = nx.MultiGraph(case.graph)
        baseline_ok = nx.is_isomorphic(target_abstract, baseline_clean)
        optimized_ok = nx.is_isomorphic(target_abstract, optimized_clean)

        record = {
            "index": index,
            "group": group,
            "case": case.name,
            "transform": transform_name,
            "radius": radius,
            "baseline": baseline_ok,
            "optimized": optimized_ok,
            "baseline_extract": statistics.median(baseline_times),
            "optimized_extract": statistics.median(optimized_times),
            "skeleton_time": skeleton_time,
        }
        records.append(record)
        print(record)

    baseline_pass = sum(row["baseline"] for row in records)
    optimized_pass = sum(row["optimized"] for row in records)
    challenge_pass = sum(
        row["optimized"] for row in records if row["group"] == "challenge"
    )
    baseline_time = statistics.median(row["baseline_extract"] for row in records)
    optimized_time = statistics.median(row["optimized_extract"] for row in records)
    speedup = baseline_time / optimized_time

    print(f"TOTAL=45 BASELINE={baseline_pass}/45 OPTIMIZED={optimized_pass}/45")
    print(f"NEW_CHALLENGE_CASES={challenge_pass}/7")
    print(
        "MEDIAN_EXTRACT "
        f"baseline={1e3 * baseline_time:.3f} ms "
        f"optimized={1e3 * optimized_time:.3f} ms "
        f"speedup={speedup:.3f}x"
    )

    assert optimized_pass == 45
    assert challenge_pass == 7
    assert optimized_pass > baseline_pass
    assert optimized_time < baseline_time
    assert speedup >= 1.50
    return records


def cycle_graph(offset=(0, 0, 0), radius=0.6, samples=180):
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=True)
    points = np.c_[
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta),
    ] + np.asarray(offset, dtype=float)
    points[-1] = points[0]
    graph = nx.MultiGraph()
    graph.add_node(0, pos=points[0].copy())
    graph.add_edge(0, 0, pts=points)
    return graph


def affine_embedded(graph):
    matrix = np.array([[1.05, 0.12, 0.03], [0.02, 0.94, 0.08], [0.04, 0.01, 1.02]]) @ rotation_xyz(13, 21, 8)
    offset = np.array([0.08, -0.05, 0.04])
    assert np.linalg.det(matrix) > 0
    result = nx.MultiGraph()
    for node, data in graph.nodes(data=True):
        result.add_node(node, pos=np.asarray(data["pos"]) @ matrix.T + offset)
    for u, v, key, data in graph.edges(keys=True, data=True):
        result.add_edge(u, v, pts=np.asarray(data["pts"]) @ matrix.T + offset)
    return result


def run_degree_two_yamada_check():
    print("Native Yamada backend:", native_available())
    print("Native import error:", native_import_error())
    assert native_available(), native_import_error()

    first = cycle_graph()
    left = cycle_graph(offset=(-0.8, 0, 0), radius=0.35)
    right = cycle_graph(offset=(0.8, 0, 0), radius=0.35)
    two = nx.disjoint_union(left, right)

    for label, graph in [("unknot_cycle", first), ("two_cycles", two)]:
        assert max(dict(graph.degree()).values()) <= 2
        deformed = affine_embedded(graph)
        assert max(dict(deformed.degree()).values()) <= 2
        old = sp.expand(compute_yamada_polynomial(graph, A, n_jobs=1))
        new = sp.expand(compute_yamada_polynomial(deformed, A, n_jobs=1))
        print(label, old, new)
        assert sp.expand(old - new) == 0


if __name__ == "__main__":
    run_reconstruction_suite()
    run_degree_two_yamada_check()
    print("PASS: 45/45 optimized reconstructions and degree<=2 Yamada checks.")
