from __future__ import annotations

import argparse
from collections import Counter
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
DEFAULT_SAMPLES = 10
DEFAULT_SEED = 20260818


@dataclass(frozen=True)
class Sample:
    vertex_count: int
    sample_index: int
    topology_seed: int
    topology_attempt: int


def _paper_vertices() -> list[int]:
    return [
        10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40,
        48, 56, 64, 80, 100, 120, 160, 200,
    ]


def _smoke_vertices() -> list[int]:
    return [10, 14, 20]


def vertex_grid(profile: str) -> list[int]:
    if profile == "paper":
        return _paper_vertices()
    if profile == "smoke":
        return _smoke_vertices()
    raise ValueError(profile)


def _seed(*parts: object) -> int:
    payload = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


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


def _connected_cubic(vertex_count: int, seed: int) -> nx.Graph:
    if vertex_count < 4 or vertex_count % 2:
        raise ValueError("connected cubic graphs require even V >= 4")
    rng = np.random.default_rng(seed)
    for _ in range(20_000):
        graph = nx.random_regular_graph(
            3,
            vertex_count,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        if nx.is_connected(graph):
            return graph
    raise RuntimeError(f"could not generate a connected cubic graph at V={vertex_count}")


def _isomorphism_profile(graph: nx.Graph):
    """Return a relabeling-invariant distance profile and per-node colors.

    Different graph-level signatures prove non-isomorphism. Equal signatures
    are only a prefilter collision: callers must still run an exact graph
    isomorphism check. The per-node profiles are safe exact-match constraints
    because graph isomorphisms preserve all-pairs shortest-path distances.
    """
    raw_profiles = {}
    diameter = 0
    for node, distances in nx.all_pairs_shortest_path_length(graph):
        counts = Counter(distances.values())
        local_max = max(counts, default=0)
        diameter = max(diameter, local_max)
        raw_profiles[node] = counts

    node_profiles = {
        node: tuple(counts.get(distance, 0) for distance in range(diameter + 1))
        for node, counts in raw_profiles.items()
    }
    signature = tuple(sorted(node_profiles.values()))
    return signature, node_profiles


def _are_isomorphic_exact(
    left: nx.Graph,
    right: nx.Graph,
    left_profile=None,
    right_profile=None,
) -> bool:
    """Exact isomorphism with a safe distance-profile rejection fast path."""
    if left.number_of_nodes() != right.number_of_nodes():
        return False
    if left.number_of_edges() != right.number_of_edges():
        return False

    if left_profile is None:
        left_profile = _isomorphism_profile(left)
    if right_profile is None:
        right_profile = _isomorphism_profile(right)

    left_signature, left_node_profiles = left_profile
    right_signature, right_node_profiles = right_profile
    if left_signature != right_signature:
        return False

    # A signature collision is never treated as equality. Fall back to exact
    # VF2, but color vertices with their invariant distance profiles to avoid
    # the pathological uncolored search seen for large regular graphs.
    left_colored = left.copy()
    right_colored = right.copy()
    nx.set_node_attributes(left_colored, left_node_profiles, "_iso_profile")
    nx.set_node_attributes(right_colored, right_node_profiles, "_iso_profile")
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "_iso_profile",
        None,
    )
    return nx.is_isomorphic(left_colored, right_colored, node_match=node_match)


def topology_ensemble(
    vertex_count: int,
    n_samples: int,
    base_seed: int,
) -> list[tuple[Sample, nx.Graph]]:
    """Generate pairwise non-isomorphic connected cubic graph instances.

    A relabeling-invariant all-pairs-distance signature rejects graphs that are
    provably non-isomorphic. Signature collisions still use an exact NetworkX
    isomorphism check, so the acceptance criterion is unchanged while avoiding
    pathological VF2 searches on large 3-regular graphs. Deterministic candidate
    seeds make the ensemble reproducible.
    """
    accepted = []
    for sample_index in range(n_samples):
        for attempt in range(50_000):
            topology_seed = _seed(
                base_seed,
                "random_cubic",
                vertex_count,
                sample_index,
                attempt,
            )
            candidate = _connected_cubic(vertex_count, topology_seed)
            candidate_profile = _isomorphism_profile(candidate)
            if any(
                _are_isomorphic_exact(
                    candidate,
                    prior,
                    candidate_profile,
                    prior_profile,
                )
                for _, prior, prior_profile in accepted
            ):
                continue
            accepted.append(
                (
                    Sample(
                        vertex_count=vertex_count,
                        sample_index=sample_index,
                        topology_seed=topology_seed,
                        topology_attempt=attempt,
                    ),
                    candidate,
                    candidate_profile,
                )
            )
            break
        else:
            raise RuntimeError(
                f"could not obtain {n_samples} pairwise non-isomorphic connected "
                f"cubic graphs at V={vertex_count}"
            )
    return [(sample, graph) for sample, graph, _ in accepted]


def _abstract_hash(graph: nx.Graph) -> str:
    """Stable hash of the exact labeled instance; non-isomorphism is checked separately."""
    digest = hashlib.sha256()
    for node in sorted(graph.nodes()):
        digest.update(repr(node).encode())
    for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges()):
        digest.update(repr((u, v)).encode())
    return digest.hexdigest()


def _embedding_hash(graph: nx.MultiGraph) -> str:
    digest = hashlib.sha256()
    for node in sorted(graph.nodes()):
        digest.update(repr(node).encode())
        digest.update(np.asarray(graph.nodes[node]["pos"], dtype=np.float64).tobytes())
    for u, v, key, data in sorted(
        graph.edges(keys=True, data=True),
        key=lambda edge: (edge[0], edge[1], edge[2]),
    ):
        digest.update(repr((u, v, key)).encode())
        digest.update(np.asarray(data["pts"], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def embed_topology(
    graph: nx.Graph,
    *,
    base_seed: int,
    vertex_count: int,
    sample_index: int,
    embedding_attempt: int,
) -> nx.MultiGraph:
    """Give one topology a deterministic generic 3-D straight-edge embedding."""
    embedding_seed = _seed(
        base_seed,
        "random_cubic_embedding",
        vertex_count,
        sample_index,
        embedding_attempt,
    )
    rng = np.random.default_rng(embedding_seed)
    positions = nx.spring_layout(
        graph,
        dim=3,
        seed=int(rng.integers(0, 2**32 - 1)),
        iterations=160,
        scale=6.0,
    )
    rotation = _random_rotation(rng)

    embedded = nx.MultiGraph()
    for node in sorted(graph.nodes()):
        xyz = np.asarray(positions[node], dtype=float) @ rotation.T
        xyz += rng.normal(scale=1e-7, size=3)
        embedded.add_node(node, pos=xyz)
    for u, v in graph.edges():
        embedded.add_edge(
            u,
            v,
            pts=np.vstack([embedded.nodes[u]["pos"], embedded.nodes[v]["pos"]]),
        )
    return embedded


def prepare_sample(
    sample: Sample,
    abstract: nx.Graph,
    base_seed: int,
):
    """Prepare one valid PD instance; all preparation is outside timed Yamada work."""
    if not nx.is_connected(abstract):
        raise AssertionError("random cubic topology must be connected")
    if any(degree != 3 for _, degree in abstract.degree()):
        raise AssertionError("random cubic topology must be exactly 3-regular")
    if abstract.number_of_edges() != 3 * sample.vertex_count // 2:
        raise AssertionError("cubic topology must satisfy E=3V/2")

    last_error = None
    for embedding_attempt in range(25):
        embedded = embed_topology(
            abstract,
            base_seed=base_seed,
            vertex_count=sample.vertex_count,
            sample_index=sample.sample_index,
            embedding_attempt=embedding_attempt,
        )
        try:
            processor = PDCode(embedded)
            pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        except Exception as exc:  # retry only the geometric realization
            last_error = exc
            continue
        return embedded, processor, pdcode, embedding_attempt

    raise RuntimeError(
        f"could not prepare a nondegenerate PD for V={sample.vertex_count}, "
        f"sample={sample.sample_index}: {last_error}"
    )


def _repeats(vertex_count: int, profile: str) -> int:
    if profile == "smoke":
        return 1
    return 3 if vertex_count <= 20 else 1


def _median_time(fn, repeats: int):
    timings = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), answer, timings


def _worker(
    framework: str,
    vertices,
    crossings,
    arcs,
    pdcode: str,
    vertex_count: int,
    profile: str,
    queue,
):
    try:
        repeats = _repeats(vertex_count, profile)
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
                "repeats": repeats,
                "terms": terms,
            }
        )
    except BaseException as exc:  # pragma: no cover - benchmark diagnostics
        queue.put(
            {
                "status": "error",
                "framework": framework,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_with_timeout(
    framework: str,
    processor: PDCode,
    pdcode: str,
    vertex_count: int,
    profile: str,
    timeout_s: float,
):
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
            vertex_count,
            profile,
            queue,
        ),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {"status": "timeout", "framework": framework, "timeout_s": timeout_s}
    if not queue.empty():
        return queue.get()
    return {
        "status": "error",
        "framework": framework,
        "error": f"worker exited with code {process.exitcode} without returning data",
    }


def benchmark_sample(
    sample: Sample,
    abstract: nx.Graph,
    *,
    base_seed: int,
    profile: str,
    timeout_s: float,
    active: dict[str, bool],
) -> dict:
    embedded, processor, pdcode, embedding_attempt = prepare_sample(
        sample,
        abstract,
        base_seed,
    )
    pd_hash = hashlib.sha256(pdcode.encode()).hexdigest()

    results = {}
    for framework in ("knottedgraph", "topoly"):
        if active[framework]:
            results[framework] = _run_with_timeout(
                framework,
                processor,
                pdcode,
                sample.vertex_count,
                profile,
                timeout_s,
            )
        else:
            results[framework] = {
                "status": "skipped_after_censor_frontier",
                "framework": framework,
            }

    kg = results["knottedgraph"]
    tp = results["topoly"]
    row = {
        "family": "random_cubic",
        "sample_kind": "topology",
        "size": sample.vertex_count,
        "sample": sample.sample_index,
        "topology_seed": sample.topology_seed,
        "topology_attempt": sample.topology_attempt,
        "embedding_attempt": embedding_attempt,
        "topology_instance_hash": _abstract_hash(abstract),
        "embedding_hash": _embedding_hash(embedded),
        "nonisomorphic_ensemble_verified": True,
        "connected": True,
        "regular_degree": 3,
        "V": abstract.number_of_nodes(),
        "E": abstract.number_of_edges(),
        "crossings": len(processor.crossings),
        "pd_length": len(pdcode),
        "pd_hash": pd_hash,
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


def main(
    *,
    timeout_s: float,
    profile: str,
    samples_per_v: int,
    base_seed: int,
    censor_frontier: int,
) -> None:
    if samples_per_v < 1:
        raise ValueError("samples-per-v must be >= 1")
    vertices = vertex_grid(profile)
    print(
        "CONFIG="
        + json.dumps(
            {
                "family": "random_cubic",
                "profile": profile,
                "samples_per_v": samples_per_v,
                "timeout_s": timeout_s,
                "base_seed": base_seed,
                "censor_frontier": censor_frontier,
                "planned_samples": len(vertices) * samples_per_v,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    rows = []
    active = {"knottedgraph": True, "topoly": True}
    fully_censored = {"knottedgraph": 0, "topoly": 0}

    for vertex_count in vertices:
        print(f"FAMILY=random_cubic V={vertex_count}", flush=True)
        ensemble = topology_ensemble(vertex_count, samples_per_v, base_seed)
        # Repeat the exact ensemble-boundary guarantee using the same safe
        # invariant prefilter and exact collision fallback.
        profiles = [_isomorphism_profile(graph) for _, graph in ensemble]
        for left in range(len(ensemble)):
            for right in range(left):
                if _are_isomorphic_exact(
                    ensemble[left][1],
                    ensemble[right][1],
                    profiles[left],
                    profiles[right],
                ):
                    raise AssertionError(
                        f"random_cubic/V={vertex_count}: samples {left} and {right} "
                        "are isomorphic"
                    )

        point_rows = []
        for sample, abstract in ensemble:
            row = benchmark_sample(
                sample,
                abstract,
                base_seed=base_seed,
                profile=profile,
                timeout_s=timeout_s,
                active=active,
            )
            rows.append(row)
            point_rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)

        for framework in ("knottedgraph", "topoly"):
            if not active[framework]:
                continue
            statuses = [row[f"{framework}_status"] for row in point_rows]
            if all(status in {"timeout", "error"} for status in statuses):
                fully_censored[framework] += 1
            else:
                fully_censored[framework] = 0
            if fully_censored[framework] >= censor_frontier:
                active[framework] = False
                print(
                    f"CENSOR_FRONTIER=random_cubic:{framework}:V={vertex_count}",
                    flush=True,
                )

        if not any(active.values()):
            break

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--samples-per-v", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--censor-frontier", type=int, default=2)
    args = parser.parse_args()
    main(
        timeout_s=args.timeout,
        profile=args.profile,
        samples_per_v=args.samples_per_v,
        base_seed=args.seed,
        censor_frontier=args.censor_frontier,
    )
