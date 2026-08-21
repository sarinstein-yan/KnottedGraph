from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp
from shapely.geometry import LineString

from knotted_graph.invariants.yamada.native import native_available
from knotted_graph.projection import compute_yamada_polynomial, select_projection

A = sp.Symbol("A")


@dataclass(frozen=True)
class Shadow:
    """A planar degree-(3,4) graph returned by plantri, with its rotation system."""

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
    size_text, payload = line.strip().split(maxsplit=1)
    n = int(size_text)
    lists = payload.split(",")
    if len(lists) != n:
        raise ValueError(f"expected {n} neighbour lists, got {len(lists)}")
    rotation = {
        vertex: tuple(ord(char) - ord("a") for char in text)
        for vertex, text in enumerate(lists)
    }
    return Shadow(index=index, rotation=rotation)


def generate_shadow_candidates(plantri: str, crossings: int) -> list[Shadow]:
    """Generate the 3-connected planar degree-sequence candidates.

    These are *candidates*, not automatically theta shadows.  A genuine
    crossing shadow must additionally have the transition system obtained by
    pairing opposite half-edges at every 4-valent vertex decompose into exactly
    three strands joining the two trivalent vertices.  That condition is
    checked separately by :func:`trace_theta_edges`.
    """
    vertices = crossings + 2
    edges = 2 * crossings + 3
    command = [plantri, "-p", "-c3", "-m3", f"-e{edges}", "-a", str(vertices)]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    candidates: list[Shadow] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        shadow = parse_plantri_ascii(line, len(candidates))
        if sorted(dict(shadow.graph.degree()).values()) == [3, 3] + [4] * crossings:
            candidates.append(shadow)
    return candidates


# Backward-compatible internal name used by the other research scripts on this branch.
generate_shadows = generate_shadow_candidates


def planar_positions(shadow: Shadow) -> dict[int, np.ndarray]:
    embedding = nx.PlanarEmbedding()
    embedding.set_data({v: list(nbrs) for v, nbrs in shadow.rotation.items()})
    embedding.check_structure()
    raw = nx.combinatorial_embedding_to_pos(embedding, fully_triangulate=False)
    points = {v: np.asarray(xy, dtype=float) for v, xy in raw.items()}
    all_xy = np.vstack(list(points.values()))
    center = all_xy.mean(axis=0)
    scale = np.max(np.linalg.norm(all_xy - center, axis=1))
    if scale <= 0:
        raise ValueError("degenerate planar layout")
    points = {v: (xy - center) / scale for v, xy in points.items()}
    verify_straight_line_shadow(shadow, points)
    return points


def verify_straight_line_shadow(
    shadow: Shadow,
    positions: dict[int, np.ndarray],
) -> None:
    """Require the selected straight-line realization to be genuinely planar."""
    edges = list(shadow.graph.edges())
    lines = {
        tuple(sorted(edge)): LineString([positions[edge[0]], positions[edge[1]]])
        for edge in edges
    }
    for i, edge_a in enumerate(edges):
        key_a = tuple(sorted(edge_a))
        for edge_b in edges[i + 1 :]:
            if set(edge_a) & set(edge_b):
                continue
            key_b = tuple(sorted(edge_b))
            if lines[key_a].intersects(lines[key_b]):
                raise ValueError(
                    f"shadow {shadow.index}: straight-line layout is not planar: "
                    f"{edge_a} intersects {edge_b}"
                )


def opposite(shadow: Shadow, crossing: int, incoming: int) -> int:
    cyclic = shadow.rotation[crossing]
    return cyclic[(cyclic.index(incoming) + 2) % 4]


def trace_theta_edges(shadow: Shadow) -> list[list[int]]:
    """Recover the three abstract theta edges from a candidate crossing shadow.

    At every 4-valent crossing vertex, a strand continues through the opposite
    half-edge in the planar cyclic order.  A valid theta shadow must therefore
    yield exactly three edge-disjoint u-to-v traces and must consume every
    shadow edge exactly once.  Degree sequence alone does not imply this.
    """
    u, v = shadow.trivalent_vertices
    traces: list[list[int]] = []
    for first in shadow.rotation[u]:
        path = [u, first]
        previous, current = u, first
        for _ in range(4 * len(shadow.rotation)):
            if current in (u, v):
                break
            if len(shadow.rotation[current]) != 4:
                raise ValueError(
                    f"shadow {shadow.index}: trace hit non-crossing vertex {current}"
                )
            nxt = opposite(shadow, current, previous)
            path.append(nxt)
            previous, current = current, nxt
        else:
            raise ValueError(f"shadow {shadow.index}: strand trace did not terminate")
        if current != v:
            raise ValueError(
                f"shadow {shadow.index}: opposite-edge transition returns a strand "
                "to the same trivalent vertex"
            )
        traces.append(path)

    traversed: dict[tuple[int, int], int] = defaultdict(int)
    for trace in traces:
        for a, b in zip(trace, trace[1:]):
            traversed[tuple(sorted((a, b)))] += 1

    expected = {tuple(sorted(edge)) for edge in shadow.graph.edges()}
    if len(traces) != 3:
        raise ValueError(f"shadow {shadow.index}: expected exactly three theta strands")
    if set(traversed) != expected:
        missing = sorted(expected - set(traversed))
        extra = sorted(set(traversed) - expected)
        raise ValueError(
            f"shadow {shadow.index}: transition system does not consume the shadow "
            f"edge-for-edge (missing={missing}, extra={extra})"
        )
    multiply_used = sorted(edge for edge, count in traversed.items() if count != 1)
    if multiply_used:
        raise ValueError(
            f"shadow {shadow.index}: transition system reuses shadow edges "
            f"{multiply_used}"
        )
    return traces


def classify_theta_candidates(
    candidates: list[Shadow],
) -> tuple[list[Shadow], list[dict[str, object]]]:
    """Separate genuine theta crossing shadows from degree-sequence candidates."""
    accepted: list[Shadow] = []
    rejected: list[dict[str, object]] = []
    for shadow in candidates:
        try:
            traces = trace_theta_edges(shadow)
        except ValueError as exc:
            rejected.append({"shadow": shadow.index, "reason": str(exc)})
            continue
        accepted.append(shadow)
        if len(traces) != 3:
            raise AssertionError("trace_theta_edges returned a non-theta decomposition")
    return accepted, rejected


def passage_index(shadow: Shadow, crossing: int, before: int, after: int) -> int:
    cyclic = shadow.rotation[crossing]
    pair = frozenset((before, after))
    if pair == frozenset((cyclic[0], cyclic[2])):
        return 0
    if pair == frozenset((cyclic[1], cyclic[3])):
        return 1
    raise ValueError(f"shadow {shadow.index}: non-opposite strand at crossing {crossing}")


def expanded_trace_points(
    shadow: Shadow,
    trace: list[int],
    positions: dict[int, np.ndarray],
    assignment: dict[int, int],
    *,
    approach_fraction: float,
    crossing_height: float = 0.05,
) -> np.ndarray:
    """Lift a theta strand while preserving the planar shadow edge-for-edge."""
    points: list[np.ndarray] = []

    def xyz(vertex: int, z: float = 0.0) -> np.ndarray:
        xy = positions[vertex]
        return np.array([xy[0], xy[1], z], dtype=float)

    points.append(xyz(trace[0]))
    for i in range(1, len(trace) - 1):
        before, crossing, after = trace[i - 1], trace[i], trace[i + 1]
        c = positions[crossing]
        pre = c + approach_fraction * (positions[before] - c)
        post = c + approach_fraction * (positions[after] - c)
        passage = passage_index(shadow, crossing, before, after)
        z = crossing_height if passage == assignment[crossing] else -crossing_height
        points.append(np.array([pre[0], pre[1], z], dtype=float))
        points.append(np.array([post[0], post[1], z], dtype=float))
    points.append(xyz(trace[-1]))
    return np.vstack(points)


def spatial_theta(
    shadow: Shadow,
    bits: int,
    *,
    approach_fraction: float,
) -> tuple[nx.MultiGraph, list[np.ndarray]]:
    crossings = shadow.crossing_vertices
    assignment = {crossing: (bits >> i) & 1 for i, crossing in enumerate(crossings)}
    positions = planar_positions(shadow)
    traces = trace_theta_edges(shadow)
    edge_points = [
        expanded_trace_points(
            shadow,
            trace,
            positions,
            assignment,
            approach_fraction=approach_fraction,
        )
        for trace in traces
    ]
    u, v = shadow.trivalent_vertices
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.r_[positions[u], 0.0])
    graph.add_node("v", pos=np.r_[positions[v], 0.0])
    for role, pts in enumerate(edge_points):
        graph.add_edge("u", "v", role=role, pts=pts)
    return graph, edge_points


def projection_crossing_count(
    shadow: Shadow,
    *,
    bits: int,
    approach_fraction: float,
) -> int:
    graph, _ = spatial_theta(shadow, bits, approach_fraction=approach_fraction)
    projection = select_projection(graph, rotation_angles=(0.0, 0.0, 0.0))
    return int(projection.num_crossings)


def choose_safe_approach_fraction(shadow: Shadow, expected_crossings: int) -> float:
    """Find a local crossing neighbourhood with no accidental XY crossings."""
    candidates = (0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005)
    all_over = (1 << expected_crossings) - 1
    diagnostics = []
    for fraction in candidates:
        counts = (
            projection_crossing_count(shadow, bits=0, approach_fraction=fraction),
            projection_crossing_count(
                shadow,
                bits=all_over,
                approach_fraction=fraction,
            ),
        )
        diagnostics.append((fraction, counts))
        if counts == (expected_crossings, expected_crossings):
            return fraction
    raise AssertionError(
        f"shadow={shadow.index}: no geometrically exact crossing neighbourhood; "
        f"tried {diagnostics}"
    )


def polynomial_key(expr: sp.Expr) -> str:
    return sp.srepr(sp.expand(expr))


def constituent_cycle(edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
    return np.vstack([edge_a, edge_b[-2:0:-1]])


def topoly_constituent_signature(edge_points: list[np.ndarray]) -> tuple[str, ...]:
    """Independent cycle-knot signature used only to certify Yamada collisions."""
    import topoly

    signatures = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cycle = constituent_cycle(edge_points[i], edge_points[j])
        kwargs = dict(closure=0, tries=1, chiral=False, max_cross=20, run_parallel=False)
        jones = topoly.jones(cycle.tolist(), **kwargs)
        homfly = topoly.homfly(cycle.tolist(), **kwargs)
        signatures.append(f"J:{jones}|H:{homfly}")
    return tuple(sorted(signatures))


def assignment_record(
    shadow: Shadow,
    bits: int,
    expected_crossings: int,
    approach_fraction: float,
) -> tuple[dict[str, object], list[np.ndarray]]:
    graph, edge_points = spatial_theta(
        shadow,
        bits,
        approach_fraction=approach_fraction,
    )
    result = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=True,
        n_jobs=1,
        crossing_warning_threshold=None,
        return_result=True,
    )
    actual_crossings = int(result.projection.num_crossings)
    if actual_crossings != expected_crossings:
        raise AssertionError(
            f"shadow={shadow.index} bits={bits}: expected {expected_crossings} "
            f"crossings, got {actual_crossings}"
        )
    polynomial = sp.expand(result.polynomial)
    return (
        {
            "shadow": shadow.index,
            "bits": bits,
            "bitstring": format(bits, f"0{expected_crossings}b"),
            "crossings": actual_crossings,
            "approach_fraction": approach_fraction,
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
) -> dict[str, object]:
    if not native_available():
        raise RuntimeError("The production native Yamada backend is required")

    all_candidates = generate_shadow_candidates(plantri, crossings)
    plantri_candidate_count = len(all_candidates)
    if crossings == 8 and plantri_candidate_count != 39:
        raise AssertionError(
            "Moriuchi reports 39 planar 3-connected degree-sequence candidates "
            f"with two trivalent and eight 4-valent vertices; got {plantri_candidate_count}"
        )

    candidates = all_candidates
    if limit_shadows is not None:
        candidates = candidates[:limit_shadows]

    shadows, rejected_candidates = classify_theta_candidates(candidates)
    for rejection in rejected_candidates:
        print(
            f"shadow {rejection['shadow']}: transition preflight SKIP -- "
            f"{rejection['reason']}",
            flush=True,
        )

    assignment_count = 1 << crossings
    if limit_assignments is not None:
        assignment_count = min(assignment_count, limit_assignments)

    records: list[dict[str, object]] = []
    edge_cache: dict[tuple[int, int], list[np.ndarray]] = {}
    geometry_fractions: dict[int, float] = {}
    for searched_index, shadow in enumerate(shadows, start=1):
        fraction = choose_safe_approach_fraction(shadow, crossings)
        geometry_fractions[shadow.index] = fraction
        print(
            f"shadow {shadow.index}: theta+geometry preflight PASS "
            f"at fraction={fraction}",
            flush=True,
        )
        for bits in range(assignment_count):
            record, edge_points = assignment_record(
                shadow,
                bits,
                crossings,
                fraction,
            )
            records.append(record)
            if certify_collisions:
                edge_cache[(shadow.index, bits)] = edge_points
        print(
            f"valid theta shadow {searched_index}/{len(shadows)} complete; "
            f"records={len(records)}",
            flush=True,
        )

    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        buckets[str(record["yamada_key"])].append(record)
    collision_buckets = [bucket for bucket in buckets.values() if len(bucket) > 1]

    certified_pairs: list[dict[str, object]] = []
    if certify_collisions:
        signature_cache: dict[tuple[int, int], tuple[str, ...]] = {}
        for bucket in collision_buckets:
            for record in bucket:
                key = (int(record["shadow"]), int(record["bits"]))
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
                            "reason": (
                                "same normalized Yamada, different independent "
                                "constituent-knot invariant multiset"
                            ),
                        }
                    )

    result: dict[str, object] = {
        "crossings": crossings,
        "plantri_candidate_count": plantri_candidate_count,
        "considered_candidate_count": len(candidates),
        "valid_theta_shadow_count": len(shadows),
        "rejected_non_theta_candidate_count": len(rejected_candidates),
        "rejected_candidates": rejected_candidates,
        "assignments_per_valid_shadow": assignment_count,
        "diagram_count": len(records),
        "distinct_yamada_count": len(buckets),
        "collision_bucket_count": len(collision_buckets),
        "largest_collision_bucket": max(
            (len(bucket) for bucket in collision_buckets), default=1
        ),
        "certified_nonisotopic_collision_pair_count": len(certified_pairs),
        "geometry_fractions": geometry_fractions,
        "certified_pairs": certified_pairs,
        "collision_buckets": collision_buckets,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    compact = {
        k: v
        for k, v in result.items()
        if k
        not in {
            "records",
            "collision_buckets",
            "certified_pairs",
            "rejected_candidates",
        }
    }
    print("SUMMARY=" + json.dumps(compact, sort_keys=True))
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
    parser.add_argument(
        "--limit-shadows",
        type=int,
        help=(
            "Limit plantri degree-sequence candidates before theta-transition "
            "filtering; intended only for smoke tests."
        ),
    )
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
