from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import statistics
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode

A = sp.Symbol("A")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_EMBEDDINGS = 10
DEFAULT_SEED = 20260818


@dataclass(frozen=True)
class Case:
    family: str
    size: int


def _kg_terms(poly: sp.Expr) -> dict[int, int]:
    out: dict[int, int] = {}
    for term in sp.expand(poly).as_ordered_terms():
        coeff, exponent = term.as_coeff_exponent(A)
        out[int(exponent)] = out.get(int(exponent), 0) + int(coeff)
    return {key: value for key, value in out.items() if value}


def _topoly_terms(poly) -> dict[int, int]:
    out: dict[int, int] = {}
    for term in poly.term:
        degree = getattr(term, "degree", {})
        exponent = int(next(iter(degree.values()))) if degree else 0
        out[exponent] = out.get(exponent, 0) + int(term.coef)
    return {key: value for key, value in out.items() if value}


def _sequence(terms: dict[int, int]) -> list[int]:
    if not terms:
        return [0]
    return [terms.get(i, 0) for i in range(min(terms), max(terms) + 1)]


def _validate_laurent_unit(kg_terms, topoly_terms):
    kg_seq = _sequence(kg_terms)
    tp_seq = _sequence(topoly_terms)
    candidates = [
        (1, 1, kg_seq),
        (-1, 1, [-value for value in kg_seq]),
        (1, -1, list(reversed(kg_seq))),
        (-1, -1, [-value for value in reversed(kg_seq)]),
    ]
    for sign, orientation, expected in candidates:
        if tp_seq == expected:
            if kg_terms and topoly_terms:
                anchor = min(kg_terms) if orientation == 1 else -max(kg_terms)
                shift = min(topoly_terms) - anchor
            else:
                shift = 0
            return sign, orientation, shift
    raise AssertionError(
        "Topoly and KnottedGraph differ beyond ±A^k and A<->A^-1: "
        f"KG={kg_seq}, Topoly={tp_seq}"
    )


def _seed_for(base_seed: int, family: str, size: int, embedding: int) -> int:
    payload = f"{base_seed}:{family}:{size}:{embedding}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _affine_xy(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a nonsingular orientation-preserving in-plane affine map."""
    theta = float(rng.uniform(-0.65, 0.65))
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    scales = np.diag(rng.uniform(0.82, 1.22, size=2))
    shear = np.array([[1.0, float(rng.uniform(-0.18, 0.18))], [0.0, 1.0]])
    matrix = rotation @ shear @ scales
    out = np.asarray(points, dtype=float).copy()
    out[:, :2] = out[:, :2] @ matrix.T
    out[:, :2] += rng.uniform(-0.35, 0.35, size=2)
    return out


def _crossing_theta_component(
    y_offset: float,
    sign: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    xscale = float(rng.uniform(0.88, 1.16))
    amp = float(rng.uniform(0.82, 1.18))
    zamp = float(rng.uniform(0.35, 0.72))
    top = float(rng.uniform(1.75, 2.35))
    curves = [
        np.array(
            [[-2*xscale, 0, 0], [-xscale, -amp, zamp * sign],
             [xscale, amp, zamp * sign], [2*xscale, 0, 0]],
            dtype=float,
        ),
        np.array(
            [[-2*xscale, 0, 0], [-xscale, amp, -zamp * sign],
             [xscale, -amp, -zamp * sign], [2*xscale, 0, 0]],
            dtype=float,
        ),
        np.array(
            [[-2*xscale, 0, 0], [-xscale, top, 0],
             [xscale, top, 0], [2*xscale, 0, 0]],
            dtype=float,
        ),
    ]
    packed = np.vstack(curves)
    packed = _affine_xy(packed, rng)
    shifted = [packed[4*i:4*(i+1)].copy() for i in range(3)]
    for points in shifted:
        points[:, 1] += y_offset
    left = shifted[0][0].copy()
    right = shifted[0][-1].copy()
    for points in shifted[1:]:
        points[0] = left
        points[-1] = right
    return tuple(shifted)


def _decomposable_crossings(
    crossing_count: int,
    rng: np.random.Generator,
) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    spacing = float(rng.uniform(6.5, 8.0))
    for component in range(crossing_count):
        y_offset = spacing * component
        sign = 1.0 if component % 2 == 0 else -1.0
        left = f"u{component}"
        right = f"v{component}"
        curves = _crossing_theta_component(y_offset, sign, rng)
        graph.add_node(left, pos=curves[0][0].copy())
        graph.add_node(right, pos=curves[0][-1].copy())
        for points in curves:
            graph.add_edge(left, right, pts=points)
    return graph


def _fixed_size_crossings(
    crossing_count: int,
    rng: np.random.Generator,
) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    span = float(crossing_count + 2) * float(rng.uniform(0.94, 1.08))
    left, right = -span, span
    graph.add_node("u", pos=np.array([left, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([right, 0.0, 0.0]))

    x = np.linspace(left, right, crossing_count + 3)
    y1 = np.zeros(crossing_count + 3)
    y2 = np.zeros(crossing_count + 3)
    amps = rng.uniform(0.82, 1.18, size=crossing_count + 1)
    for index in range(1, crossing_count + 2):
        sign = 1.0 if index % 2 else -1.0
        y1[index] = sign * amps[index - 1]
        y2[index] = -sign * amps[index - 1]
    zamp = float(rng.uniform(0.35, 0.72))
    z1 = np.full(crossing_count + 3, zamp)
    z2 = np.full(crossing_count + 3, -zamp)
    strand1 = np.column_stack([x, y1, z1])
    strand2 = np.column_stack([x, y2, z2])
    strand1[[0, -1], 2] = 0.0
    strand2[[0, -1], 2] = 0.0
    third_height = float(rng.uniform(2.7, 3.6))
    third = np.array(
        [[left, 0, 0], [left + 1, third_height, 0],
         [right - 1, third_height, 0], [right, 0, 0]],
        dtype=float,
    )
    all_points = np.vstack([strand1, strand2, third])
    transformed = _affine_xy(all_points, rng)
    n = len(strand1)
    strand1 = transformed[:n]
    strand2 = transformed[n:2*n]
    third = transformed[2*n:]
    left_pos, right_pos = strand1[0].copy(), strand1[-1].copy()
    for pts in (strand1, strand2, third):
        pts[0] = left_pos
        pts[-1] = right_pos
    graph.nodes["u"]["pos"] = left_pos
    graph.nodes["v"]["pos"] = right_pos
    graph.add_edge("u", "v", pts=strand1)
    graph.add_edge("u", "v", pts=strand2)
    graph.add_edge("u", "v", pts=third)
    return graph


def _edge_theta(edge_count: int, rng: np.random.Generator) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    offsets = np.linspace(-4.0, 4.0, edge_count)
    if edge_count > 2:
        jitter = rng.uniform(-0.18, 0.18, size=edge_count)
        offsets = np.sort(offsets + jitter)
    curves = []
    for offset in offsets:
        curves.append(
            np.array(
                [[-2, 0, 0], [-1, float(offset), 0],
                 [1, float(offset), 0], [2, 0, 0]],
                dtype=float,
            )
        )
    packed = np.vstack(curves)
    packed = _affine_xy(packed, rng)
    curves = [packed[4*i:4*(i+1)].copy() for i in range(edge_count)]
    left, right = curves[0][0].copy(), curves[0][-1].copy()
    graph.add_node("u", pos=left)
    graph.add_node("v", pos=right)
    for points in curves:
        points[0] = left
        points[-1] = right
        graph.add_edge("u", "v", pts=points)
    return graph


def _k4_components(vertex_count: int, rng: np.random.Generator) -> nx.MultiGraph:
    if vertex_count % 4:
        raise ValueError("vertex_count must be divisible by four")
    graph = nx.MultiGraph()
    base = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    spacing = float(rng.uniform(4.0, 5.0))
    for component in range(vertex_count // 4):
        local = _affine_xy(base, rng)
        local[:, 0] += spacing * component
        nodes = [4 * component + index for index in range(4)]
        for index, node in enumerate(nodes):
            graph.add_node(node, pos=local[index].copy())
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = nodes[i], nodes[j]
                graph.add_edge(
                    u, v,
                    pts=np.vstack([graph.nodes[u]["pos"], graph.nodes[v]["pos"]]),
                )
    return graph


def _prism(rung_count: int, rng: np.random.Generator) -> nx.MultiGraph:
    abstract = nx.circular_ladder_graph(rung_count)
    positions = nx.planar_layout(abstract, scale=5.0)
    nodes = list(abstract.nodes())
    coords = np.array(
        [[float(positions[node][0]), float(positions[node][1]), 0.0] for node in nodes]
    )
    coords = _affine_xy(coords, rng)
    graph = nx.MultiGraph()
    for node, xyz in zip(nodes, coords):
        graph.add_node(node, pos=xyz.copy())
    for u, v in abstract.edges():
        graph.add_edge(
            u, v,
            pts=np.vstack([graph.nodes[u]["pos"], graph.nodes[v]["pos"]]),
        )
    return graph


def _build_graph(case: Case, seed: int) -> nx.MultiGraph:
    rng = np.random.default_rng(seed)
    builders = {
        "crossings_fixed": _fixed_size_crossings,
        "crossings_throughput": _decomposable_crossings,
        "edges_theta": _edge_theta,
        "vertices_k4": _k4_components,
        "connected_prism": _prism,
    }
    return builders[case.family](case.size, rng)


def _expected_crossings(case: Case) -> int:
    if case.family in {"crossings_fixed", "crossings_throughput"}:
        return case.size
    return 0


def _embedding_hash(graph: nx.MultiGraph) -> str:
    digest = hashlib.sha256()
    for node in sorted(graph.nodes(), key=repr):
        digest.update(repr(node).encode())
        digest.update(np.asarray(graph.nodes[node]["pos"], dtype=np.float64).tobytes())
    for u, v, key, data in sorted(
        graph.edges(keys=True, data=True),
        key=lambda e: (repr(e[0]), repr(e[1]), repr(e[2])),
    ):
        digest.update(repr((u, v, key)).encode())
        digest.update(np.asarray(data["pts"], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _prepare(case: Case, seed: int):
    """Build one embedding and its PD input outside the timed Yamada section."""
    graph = _build_graph(case, seed)
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    crossings = len(processor.crossings)
    expected = _expected_crossings(case)
    if crossings != expected:
        raise AssertionError(
            f"{case.family}/{case.size}/seed={seed}: "
            f"expected {expected} crossings, got {crossings}"
        )
    if case.family == "crossings_fixed" and (
        graph.number_of_nodes(), graph.number_of_edges()
    ) != (2, 3):
        raise AssertionError("fixed crossing family must keep V=2, E=3")
    if case.family == "edges_theta" and graph.number_of_nodes() != 2:
        raise AssertionError("edge family must keep V=2")
    if case.family == "vertices_k4" and graph.number_of_nodes() != case.size:
        raise AssertionError("vertex family size must equal V")
    return graph, processor, pdcode


def _repeats(case: Case, profile: str) -> int:
    if profile == "smoke":
        return 1
    if case.family == "crossings_fixed":
        return 5 if case.size <= 4 else (3 if case.size <= 10 else 1)
    if case.family == "crossings_throughput":
        return 5 if case.size <= 10 else (3 if case.size <= 40 else 1)
    if case.family == "edges_theta":
        return 5 if case.size <= 30 else (3 if case.size <= 300 else 1)
    if case.family == "vertices_k4":
        return 5 if case.size <= 32 else (3 if case.size <= 512 else 1)
    return 3 if case.size <= 12 else 1


def _median_time(fn, repeats: int):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer, values


def _worker(
    framework: str,
    vertices,
    crossings,
    arcs,
    pdcode: str,
    case: Case,
    embedding: int,
    seed: int,
    profile: str,
    queue,
):
    try:
        repeats = _repeats(case, profile)

        if framework == "knottedgraph":
            def run():
                return Yamada(
                    list(vertices),
                    list(crossings),
                    list(arcs),
                ).compute(A, normalize=False, n_jobs=1, method="negami")

            elapsed, answer, timings = _median_time(run, repeats)
            terms = _kg_terms(answer)
        elif framework == "topoly":
            from topoly.invariants import Invariant, YamadaGraph

            def run():
                Invariant.known["Yamada"] = {}
                return YamadaGraph(pdcode).point(max_cross=5000)

            elapsed, answer, timings = _median_time(run, repeats)
            terms = _topoly_terms(answer)
        else:
            raise ValueError(framework)

        queue.put(
            {
                "status": "ok",
                "framework": framework,
                "time_s": elapsed,
                "timings_s": timings,
                "terms": terms,
                "repeats": repeats,
                "embedding": embedding,
                "embedding_seed": seed,
            }
        )
    except BaseException as exc:  # pragma: no cover - benchmark diagnostics
        queue.put(
            {
                "status": "error",
                "framework": framework,
                "embedding": embedding,
                "embedding_seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_with_timeout(
    framework: str,
    case: Case,
    embedding: int,
    seed: int,
    profile: str,
    timeout_s: float,
    processor: PDCode,
    pdcode: str,
):
    """Time only the Yamada evaluator; PD construction happened in the parent."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            framework,
            list(processor.vertices.values()),
            list(processor.crossings.values()),
            list(processor.arcs.values()),
            pdcode,
            case,
            embedding,
            seed,
            profile,
            queue,
        ),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {
            "status": "timeout",
            "framework": framework,
            "embedding": embedding,
            "embedding_seed": seed,
            "timeout_s": timeout_s,
        }
    if not queue.empty():
        return queue.get()
    return {
        "status": "error",
        "framework": framework,
        "embedding": embedding,
        "embedding_seed": seed,
        "error": f"worker exited with code {process.exitcode} without returning data",
    }


def _row(
    case: Case,
    embedding: int,
    seed: int,
    profile: str,
    timeout_s: float,
    active: dict[str, bool],
):
    graph, processor, pdcode = _prepare(case, seed)
    embedding_hash = _embedding_hash(graph)
    pd_hash = hashlib.sha256(pdcode.encode()).hexdigest()

    results = {}
    for framework in ("knottedgraph", "topoly"):
        if active[framework]:
            results[framework] = _run_with_timeout(
                framework,
                case,
                embedding,
                seed,
                profile,
                timeout_s,
                processor,
                pdcode,
            )
        else:
            results[framework] = {
                "status": "skipped_after_censor_frontier",
                "framework": framework,
                "embedding": embedding,
                "embedding_seed": seed,
            }

    kg = results["knottedgraph"]
    tp = results["topoly"]
    row = {
        "family": case.family,
        "size": case.size,
        "embedding": embedding,
        "embedding_seed": seed,
        "embedding_hash": embedding_hash,
        "pd_hash": pd_hash,
        "pd_length": len(pdcode),
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "crossings": len(processor.crossings),
        "timeout_s": timeout_s,
        "knottedgraph_status": kg["status"],
        "topoly_status": tp["status"],
        "knottedgraph_s": kg.get("time_s"),
        "topoly_s": tp.get("time_s"),
        "knottedgraph_repeats": kg.get("repeats"),
        "topoly_repeats": tp.get("repeats"),
    }

    for result in (kg, tp):
        if result["status"] == "error":
            row[f"{result['framework']}_error"] = result.get("error")

    if kg["status"] == "ok" and tp["status"] == "ok":
        sign, orientation, shift = _validate_laurent_unit(kg["terms"], tp["terms"])
        row.update(
            {
                "unit_sign_topoly_over_kg": sign,
                "variable_orientation": orientation,
                "monomial_shift_topoly_minus_kg": shift,
                "topoly_over_kg": tp["time_s"] / kg["time_s"],
                "kg_over_topoly": kg["time_s"] / tp["time_s"],
                "coefficient_count": len(_sequence(kg["terms"])),
                "correctness": "PASS",
            }
        )
    else:
        row["correctness"] = "not-evaluated-after-timeout-error-or-skip"
    return row


def _cases(profile: str):
    if profile == "smoke":
        fixed_c = [1, 2, 4, 8, 12]
        throughput_c = [1, 4, 8, 16, 32]
        edges = [3, 10, 30, 100, 200]
        vertices = [4, 16, 64, 256, 512]
        prisms = [3, 4, 6, 8, 10]
    elif profile == "paper":
        fixed_c = list(range(1, 21)) + [24, 28, 32, 36, 40, 48, 56, 64, 80]
        throughput_c = (
            list(range(1, 21))
            + [25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300,
               400, 500, 750, 1000]
        )
        edges = [
            3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 150, 200,
            300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000,
        ]
        vertices = [
            4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
            768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        ]
        prisms = [
            3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32,
            36, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256,
        ]
    else:
        raise ValueError(profile)

    return {
        "crossings_fixed": [Case("crossings_fixed", value) for value in fixed_c],
        "crossings_throughput": [
            Case("crossings_throughput", value) for value in throughput_c
        ],
        "edges_theta": [Case("edges_theta", value) for value in edges],
        "vertices_k4": [Case("vertices_k4", value) for value in vertices],
        "connected_prism": [Case("connected_prism", value) for value in prisms],
    }


def main(
    timeout_s: float,
    profile: str,
    embeddings: int,
    base_seed: int,
    censor_frontier: int,
):
    if embeddings < 1:
        raise ValueError("embeddings must be >= 1")
    if censor_frontier < 1:
        raise ValueError("censor_frontier must be >= 1")

    print(
        "CONFIG="
        + json.dumps(
            {
                "profile": profile,
                "embeddings_per_x": embeddings,
                "timeout_s": timeout_s,
                "base_seed": base_seed,
                "censor_frontier": censor_frontier,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    rows = []
    for family, cases in _cases(profile).items():
        active = {"knottedgraph": True, "topoly": True}
        consecutive_fully_censored = {"knottedgraph": 0, "topoly": 0}
        print(f"FAMILY={family}", flush=True)

        for case in cases:
            case_rows = []
            embedding_hashes = set()
            for embedding in range(embeddings):
                seed = _seed_for(base_seed, family, case.size, embedding)
                row = _row(case, embedding, seed, profile, timeout_s, active)
                rows.append(row)
                case_rows.append(row)
                embedding_hashes.add(row["embedding_hash"])
                print(json.dumps(row, separators=(",", ":")), flush=True)

            if len(embedding_hashes) != embeddings:
                raise AssertionError(
                    f"{family}/{case.size}: duplicate embedding geometry detected"
                )

            for framework in ("knottedgraph", "topoly"):
                if not active[framework]:
                    continue
                statuses = [row[f"{framework}_status"] for row in case_rows]
                if all(status in {"timeout", "error"} for status in statuses):
                    consecutive_fully_censored[framework] += 1
                else:
                    consecutive_fully_censored[framework] = 0
                if consecutive_fully_censored[framework] >= censor_frontier:
                    active[framework] = False
                    print(
                        f"CENSOR_FRONTIER={family}:{framework}:{case.size}",
                        flush=True,
                    )

            if not any(active.values()):
                break

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="hard wall-time limit per framework/embedding in seconds",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "paper"),
        default="paper",
        help="smoke is CI-sized; paper is the long-range publication configuration",
    )
    parser.add_argument(
        "--embeddings",
        type=int,
        default=DEFAULT_EMBEDDINGS,
        help="independent deterministic geometric embeddings per x-axis point",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="base seed used to generate deterministic independent embeddings",
    )
    parser.add_argument(
        "--censor-frontier",
        type=int,
        default=2,
        help=(
            "stop evaluating one framework after this many consecutive x points "
            "where every embedding timed out/errored"
        ),
    )
    args = parser.parse_args()
    main(
        args.timeout,
        args.profile,
        args.embeddings,
        args.seed,
        args.censor_frontier,
    )
