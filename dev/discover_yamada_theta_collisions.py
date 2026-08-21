from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.native import native_available
from knotted_graph.projection import compute_yamada_polynomial

A = sp.Symbol("A")


@dataclass(frozen=True)
class Shadow:
    index: int
    rotation: dict[int, tuple[int, ...]]

    @property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        for vertex, neighbours in self.rotation.items():
            for neighbour in neighbours:
                graph.add_edge(vertex, neighbour)
        return graph

    @property
    def trivalent_vertices(self) -> tuple[int, int]:
        vertices = tuple(v for v, nbrs in self.rotation.items() if len(nbrs) == 3)
        if len(vertices) != 2:
            raise ValueError(f"shadow {self.index}: expected two trivalent vertices")
        return vertices

    @property
    def crossing_vertices(self) -> tuple[int, ...]:
        vertices = tuple(v for v, nbrs in self.rotation.items() if len(nbrs) == 4)
        expected = len(self.rotation) - 2
        if len(vertices) != expected:
            raise ValueError(f"shadow {self.index}: unexpected degree sequence")
        return vertices


def parse_plantri_ascii(line: str, index: int) -> Shadow:
    line = line.strip()
    if not line:
        raise ValueError("empty plantri line")
    size_text, payload = line.split(maxsplit=1)
    n = int(size_text)
    lists = payload.split(",")
    if len(lists) != n:
        raise ValueError(f"expected {n} neighbour lists, got {len(lists)}")
    rotation: dict[int, tuple[int, ...]] = {}
    for vertex, text in enumerate(lists):
        rotation[vertex] = tuple(ord(char) - ord("a") for char in text)
    return Shadow(index=index, rotation=rotation)


def generate_shadows(plantri: str, crossings: int) -> list[Shadow]:
    vertices = crossings + 2
    edges = 2 * crossings + 3
    command = [plantri, "-p", "-c3", "-m3", f"-e{edges}", "-a", str(vertices)]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    shadows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        shadow = parse_plantri_ascii(line, len(shadows))
        degrees = sorted(dict(shadow.graph.degree()).values())
        if degrees == [3, 3] + [4] * crossings:
            shadows.append(shadow)
    return shadows


def planar_positions(shadow: Shadow) -> dict[int, np.ndarray]:
    embedding = nx.PlanarEmbedding()
    embedding.set_data({v: list(nbrs) for v, nbrs in shadow.rotation.items()})
    embedding.check_structure()
    pos = nx.combinatorial_embedding_to_pos(embedding, fully_triangulate=True)
    points = {v: np.asarray(xy, dtype=float) for v, xy in pos.items()}
    all_xy = np.vstack(list(points.values()))
    center = all_xy.mean(axis=0)
    scale = np.max(np.linalg.norm(all_xy - center, axis=1))
    if scale <= 0:
        raise ValueError("degenerate planar layout")
    return {v: (xy - center) / scale for v, xy in points.items()}


def opposite(shadow: Shadow, crossing: int, incoming: int) -> int:
    cyclic = shadow.rotation[crossing]
    i = cyclic.index(incoming)
    return cyclic[(i + 2) % 4]


def trace_theta_edges(shadow: Shadow) -> list[list[int]]:
    u, v = shadow.trivalent_vertices
    traces: list[list[int]] = []
    for first in shadow.rotation[u]:
        path = [u, first]
        previous, current = u, first
        guard = 0
        while current not in (u, v):
            if len(shadow.rotation[current]) != 4:
                raise ValueError(f"shadow {shadow.index}: trace hit non-crossing vertex")
            nxt = opposite(shadow, current, previous)
            path.append(nxt)
            previous, current = current, nxt
            guard += 1
            if guard > 4 * len(shadow.rotation):
                raise ValueError(f"shadow {shadow.index}: strand trace did not terminate")
        if current != v:
            raise ValueError(f"shadow {shadow.index}: strand returns to same trivalent vertex")
        traces.append(path)

    if len(traces) != 3:
        raise ValueError(f"shadow {shadow.index}: expected three theta edges")

    # Every shadow edge must belong to exactly one passage of the traced theta graph.
    traversed = defaultdict(int)
    for trace in traces:
        for a, b in zip(trace, trace[1:]):
            traversed[tuple(sorted((a, b)))] += 1
    expected = {tuple(sorted(edge)) for edge in shadow.graph.edges()}
    if set(traversed) != expected or any(count != 1 for count in traversed.values()):
        raise ValueError(f"shadow {shadow.index}: strand decomposition is inconsistent")
    return traces


def passage_index(shadow: Shadow, crossing: int, before: int, after: int) -> int:
    cyclic = shadow.rotation[crossing]
    pair = frozenset((before, after))
    p0 = frozenset((cyclic[0], cyclic[2]))
    p1 = frozenset((cyclic[1], cyclic[3]))
    if pair == p0:
        return 0
    if pair == p1:
        return 1
    raise ValueError(f"shadow {shadow.index}: non-opposite strand at crossing {crossing}")


def expanded_trace_points(
    shadow: Shadow,
    trace: list[int],
    positions: dict[int, np.ndarray],
    assignment: dict[int, int],
    *,
    approach_fraction: float = 0.20,
    crossing_height: float = 0.08,
) -> np.ndarray:
    points: list[np.ndarray] = []

    def xyz(vertex: int, z: float = 0.0) -> np.ndarray:
        return np.array([positions[vertex][0], positions[vertex][1], z], dtype=float)

    points.append(xyz(trace[0]))
    for i in range(1, len(trace) - 1):
        before, crossing, after = trace[i - 1], trace[i], trace[i + 1]
        c = positions[crossing]
        p = positions[before]
        q = positions[after]
        pre = c + approach_fraction * (p - c)
        post = c + approach_fraction * (q - c)
        passage = passage_index(shadow, crossing, before, after)
        bit = assignment[crossing]
        z = crossing_height if passage == bit else -crossing_height
        points.append(np.array([pre[0], pre[1], z], dtype=float))
        points.append(np.array([post[0], post[1], z], dtype=float))
    points.append(xyz(trace[-1]))
    return np.vstack(points)


def spatial_theta(shadow: Shadow, bits: int) -> tuple[nx.MultiGraph, list[np.ndarray]]:
    crossings = shadow.crossing_vertices
    assignment = {crossing: (bits >> i) & 1 for i, crossing in enumerate(crossings)}
    positions = planar_positions(shadow)
    traces = trace_theta_edges(shadow)
    edge_points = [
        expanded_trace_points(shadow, trace, positions, assignment)
        for trace in traces
    ]
    u, v = shadow.trivalent_vertices
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([positions[u][0], positions[u][1], 0.0]))
    graph.add_node("v", pos=np.array([positions[v][0], positions[v][1], 0.0]))
    for role, pts in enumerate(edge_points):
        graph.add_edge("u", "v", role=role, pts=pts)
    return graph, edge_points


def polynomial_key(expr: sp.Expr) -> str:
    return sp.srepr(sp.expand(expr))


def constituent_cycle(edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
    # edge_a runs u->v; return along edge_b and let Topoly close the last tiny gap.
    if len(edge_b) < 3:
        raise ValueError("edge has too few points")
    return np.vstack([edge_a, edge_b[-2:0:-1]])


def topoly_constituent_signature(edge_points: list[np.ndarray]) -> tuple[str, ...]:
    import topoly

    signatures = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cycle = constituent_cycle(edge_points[i], edge_points[j])
        kwargs = dict(
            closure=0,
            tries=1,
            chiral=False,
            max_cross=20,
            run_parallel=False,
        )
        # Topology labels are obtained independently by Topoly's knot/link machinery.
        # Use two polynomial classifiers to reduce the chance of an ambiguous label.
        jones = topoly.jones(cycle.tolist(), **kwargs)
        homfly = topoly.homfly(cycle.tolist(), **kwargs)
        signatures.append(f"J:{jones}|H:{homfly}")
    return tuple(sorted(signatures))


def assignment_record(
    shadow: Shadow,
    bits: int,
    expected_crossings: int,
) -> tuple[dict, list[np.ndarray]]:
    graph, edge_points = spatial_theta(shadow, bits)
    result = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=True,
        n_jobs=1,
        method="negami",
        crossing_warning_threshold=None,
        return_result=True,
    )
    actual_crossings = int(result.projection.num_crossings)
    if actual_crossings != expected_crossings:
        raise AssertionError(
            f"shadow={shadow.index} bits={bits}: expected {expected_crossings} "
            f"crossings in canonical projection, got {actual_crossings}"
        )
    polynomial = sp.expand(result.polynomial)
    return (
        {
            "shadow": shadow.index,
            "bits": bits,
            "bitstring": format(bits, f"0{expected_crossings}b"),
            "crossings": actual_crossings,
            "yamada": str(polynomial),
            "yamada_key": polynomial_key(polynomial),
        },
        edge_points,
    )


def search(
    plantri: str,
    crossings: int,
    output: Path,
    *,
    limit_shadows: int | None,
    limit_assignments: int | None,
    certify_collisions: bool,
) -> dict:
    if not native_available():
        raise RuntimeError("The production native Yamada backend is required")

    shadows = generate_shadows(plantri, crossings)
    if crossings == 8 and len(shadows) != 39:
        raise AssertionError(
            f"Moriuchi reports 39 3-connected 8-crossing theta shadows; got {len(shadows)}"
        )
    if limit_shadows is not None:
        shadows = shadows[:limit_shadows]

    records: list[dict] = []
    edge_cache: dict[tuple[int, int], list[np.ndarray]] = {}
    assignment_count = 1 << crossings
    if limit_assignments is not None:
        assignment_count = min(assignment_count, limit_assignments)

    for shadow in shadows:
        for bits in range(assignment_count):
            record, edge_points = assignment_record(shadow, bits, crossings)
            records.append(record)
            edge_cache[(shadow.index, bits)] = edge_points
        print(
            f"shadow {shadow.index + 1}/{len(shadows)} complete; "
            f"records={len(records)}",
            flush=True,
        )

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[record["yamada_key"]].append(record)

    collision_buckets = [bucket for bucket in buckets.values() if len(bucket) > 1]
    certified_pairs: list[dict] = []
    if certify_collisions:
        signature_cache: dict[tuple[int, int], tuple[str, ...]] = {}
        for bucket in collision_buckets:
            # Compute independent constituent signatures only where Yamada collided.
            for record in bucket:
                key = (record["shadow"], record["bits"])
                if key not in signature_cache:
                    signature_cache[key] = topoly_constituent_signature(edge_cache[key])
                record["constituent_signature"] = signature_cache[key]

            for left, right in itertools.combinations(bucket, 2):
                left_sig = tuple(left["constituent_signature"])
                right_sig = tuple(right["constituent_signature"])
                if left_sig != right_sig:
                    certified_pairs.append(
                        {
                            "left": {
                                "shadow": left["shadow"],
                                "bits": left["bits"],
                                "bitstring": left["bitstring"],
                                "constituent_signature": left_sig,
                            },
                            "right": {
                                "shadow": right["shadow"],
                                "bits": right["bits"],
                                "bitstring": right["bitstring"],
                                "constituent_signature": right_sig,
                            },
                            "yamada": left["yamada"],
                            "reason": "same normalized Yamada, different Topoly constituent-knot multiset",
                        }
                    )

    result = {
        "crossings": crossings,
        "shadow_count": len(shadows),
        "assignments_per_shadow": assignment_count,
        "diagram_count": len(records),
        "distinct_yamada_count": len(buckets),
        "collision_bucket_count": len(collision_buckets),
        "largest_collision_bucket": max((len(bucket) for bucket in collision_buckets), default=1),
        "certified_nonisotopic_collision_pair_count": len(certified_pairs),
        "certified_pairs": certified_pairs,
        "collision_buckets": collision_buckets,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY=" + json.dumps({k: v for k, v in result.items() if k not in {"records", "collision_buckets", "certified_pairs"}}, sort_keys=True))
    if certified_pairs:
        print("CERTIFIED_COLLISION=" + json.dumps(certified_pairs[0], sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate theta shadows and search normalized-Yamada collisions."
    )
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--crossings", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-shadows", type=int)
    parser.add_argument("--limit-assignments", type=int)
    parser.add_argument("--certify-collisions", action="store_true")
    args = parser.parse_args()
    if args.crossings < 1:
        raise SystemExit("--crossings must be positive")
    search(
        args.plantri,
        args.crossings,
        args.output,
        limit_shadows=args.limit_shadows,
        limit_assignments=args.limit_assignments,
        certify_collisions=args.certify_collisions,
    )


if __name__ == "__main__":
    main()
