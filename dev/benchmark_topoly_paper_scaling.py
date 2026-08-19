from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np
import sympy as sp

# This file is both executed directly and loaded with importlib by notebook 03.
# In the latter case Python does not automatically put dev/ on sys.path.
DEV = Path(__file__).resolve().parent
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from benchmark_topoly_extended_scaling import (  # noqa: E402
    Case,
    _embedding_hash,
    _k4_components,
    _kg_terms,
    _seed_for,
    _sequence,
    _topoly_terms,
    _validate_laurent_unit,
)
from knotted_graph.invariants.yamada.polynomial import Yamada  # noqa: E402
from knotted_graph.projection import PDCode  # noqa: E402

A = sp.Symbol("A")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_CROSSING_GRAPHS = 21
DEFAULT_K4_EMBEDDINGS = 10
DEFAULT_SEED = 20260818

CROSSING_FAMILY = "crossings_graph_ensemble"
K4_FAMILY = "vertices_k4"
PAPER_FAMILIES = (CROSSING_FAMILY, K4_FAMILY)


def crossing_grid(profile: str) -> list[int]:
    if profile == "smoke":
        return [1, 4, 8, 12]
    if profile == "paper":
        return [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
            24, 28, 32, 36, 40, 48, 56, 64, 80,
        ]
    raise ValueError(profile)


def k4_vertex_grid(profile: str) -> list[int]:
    if profile == "smoke":
        return [4, 16, 64, 256]
    if profile == "paper":
        return [
            4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
            384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        ]
    raise ValueError(profile)


def paper_plan(profile: str, crossing_graphs: int, k4_embeddings: int) -> dict:
    if crossing_graphs < 1 or k4_embeddings < 1:
        raise ValueError("sample counts must be >= 1")
    return {
        CROSSING_FAMILY: {
            "x_values": crossing_grid(profile),
            "samples_per_x": crossing_graphs,
        },
        K4_FAMILY: {
            "x_values": k4_vertex_grid(profile),
            "samples_per_x": k4_embeddings,
        },
    }


def _crossing_strands(crossing_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Two spatial strands whose xy projection has exactly crossing_count crossings."""
    if crossing_count < 1:
        raise ValueError("crossing_count must be >= 1")
    span = float(crossing_count + 2)
    left, right = -span, span
    x = np.linspace(left, right, crossing_count + 3)
    y1 = np.zeros(crossing_count + 3)
    y2 = np.zeros(crossing_count + 3)
    for index in range(1, crossing_count + 2):
        sign = 1.0 if index % 2 else -1.0
        y1[index] = sign
        y2[index] = -sign
    z1 = np.full(crossing_count + 3, 0.5)
    z2 = np.full(crossing_count + 3, -0.5)
    z1[[0, -1]] = 0.0
    z2[[0, -1]] = 0.0
    return np.column_stack([x, y1, z1]), np.column_stack([x, y2, z2])


def _prism_background(rung_count: int, y_center: float) -> tuple[nx.Graph, dict[int, np.ndarray]]:
    """Planar cubic prism with its lower outer edge reserved for the crossing splice."""
    if rung_count < 3:
        raise ValueError("prism rung_count must be >= 3")
    abstract = nx.circular_ladder_graph(rung_count)
    outer_radius = 3.0
    inner_radius = 1.7

    # Outer nodes 0 and 1 straddle the bottom direction, so edge (0,1) is on
    # the outer face and can be removed without creating projection crossings.
    angles = -np.pi / 2 + (np.arange(rung_count) - 0.5) * 2 * np.pi / rung_count
    positions: dict[int, np.ndarray] = {}
    for index, angle in enumerate(angles):
        positions[index] = np.array(
            [outer_radius * np.cos(angle), y_center + outer_radius * np.sin(angle), 0.0]
        )
        positions[index + rung_count] = np.array(
            [inner_radius * np.cos(angle), y_center + inner_radius * np.sin(angle), 0.0]
        )

    if not abstract.has_edge(0, 1):
        raise AssertionError("expected outer prism edge (0,1)")
    return abstract, positions


def _connector(start: np.ndarray, end: np.ndarray, *, side: float, clearance_y: float) -> np.ndarray:
    """Route from a theta endpoint outside the weave and then to an outer-face vertex."""
    outside_x = start[0] + side * 2.5
    return np.asarray(
        [
            start,
            [outside_x, start[1], 0.0],
            [outside_x, clearance_y, 0.0],
            [end[0], clearance_y, 0.0],
            end,
        ],
        dtype=float,
    )


def crossing_graph(crossing_count: int, graph_index: int) -> nx.MultiGraph:
    """Connected trivalent graph with controlled c and graph-dependent V/E.

    graph_index selects a prism background with rung_count=3+graph_index.  The
    same panel of graph sizes is used at every crossing count, so the V/E
    distribution is held fixed as c varies.
    """
    rung_count = 3 + graph_index
    strand1, strand2 = _crossing_strands(crossing_count)
    left = strand1[0].copy()
    right = strand1[-1].copy()
    y_center = 9.0
    abstract, positions = _prism_background(rung_count, y_center)

    graph = nx.MultiGraph()
    graph.add_node("theta_u", pos=left)
    graph.add_node("theta_v", pos=right)
    graph.add_edge("theta_u", "theta_v", pts=strand1, role="crossing_strand_1")
    graph.add_edge("theta_u", "theta_v", pts=strand2, role="crossing_strand_2")

    for node in abstract.nodes():
        graph.add_node(("p", int(node)), pos=positions[node].copy())
    for u, v in abstract.edges():
        if {u, v} == {0, 1}:
            continue
        graph.add_edge(
            ("p", int(u)),
            ("p", int(v)),
            pts=np.vstack([positions[u], positions[v]]),
            role="background",
        )

    # Replace one outer-face prism edge with two theta-to-prism connectors.
    # This edge splice preserves degree 3 at every vertex and makes the whole
    # graph connected without introducing a decomposable crossing component.
    p0 = positions[0]
    p1 = positions[1]
    clearance_y = min(p0[1], p1[1]) - 0.8
    graph.add_edge(
        "theta_u",
        ("p", 0),
        pts=_connector(left, p0, side=-1.0, clearance_y=clearance_y),
        role="splice",
    )
    graph.add_edge(
        "theta_v",
        ("p", 1),
        pts=_connector(right, p1, side=1.0, clearance_y=clearance_y),
        role="splice",
    )

    degrees = dict(graph.degree())
    if not nx.is_connected(nx.Graph(graph)):
        raise AssertionError("crossing benchmark graph must be connected")
    if not degrees or any(degree != 3 for degree in degrees.values()):
        raise AssertionError(f"crossing benchmark graph is not trivalent: {degrees}")
    if graph.number_of_edges() != 3 * graph.number_of_nodes() // 2:
        raise AssertionError("trivalent E=3V/2 identity failed")
    return graph


def _prepare_crossing(crossing_count: int, graph_index: int):
    graph = crossing_graph(crossing_count, graph_index)
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    actual = len(processor.crossings)
    if actual != crossing_count:
        raise AssertionError(
            f"crossing graph {graph_index}: expected c={crossing_count}, got {actual}"
        )
    return graph, processor, pdcode


def _prepare_k4(vertex_count: int, embedding: int, base_seed: int):
    seed = _seed_for(base_seed, K4_FAMILY, vertex_count, embedding)
    rng = np.random.default_rng(seed)
    graph = _k4_components(vertex_count, rng)
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    if len(processor.crossings) != 0:
        raise AssertionError(f"K4 input-size benchmark expected c=0, got {len(processor.crossings)}")
    if graph.number_of_nodes() != vertex_count:
        raise AssertionError("K4 benchmark V mismatch")
    return graph, processor, pdcode, seed


def _run_once(framework: str, vertices, crossings, arcs, pdcode: str):
    if framework == "knottedgraph":
        answer = Yamada(
            list(vertices), list(crossings), list(arcs)
        ).compute(A, normalize=False, n_jobs=1, method="negami")
        return _kg_terms(answer)

    if framework == "topoly":
        from topoly.invariants import Invariant, YamadaGraph

        Invariant.known["Yamada"] = {}
        answer = YamadaGraph(pdcode).point(max_cross=5000)
        return _topoly_terms(answer)

    raise ValueError(framework)


def _worker(framework: str, vertices, crossings, arcs, pdcode: str, queue):
    """Exactly one timed evaluation; replication comes from independent graph samples."""
    try:
        start = time.perf_counter()
        terms = _run_once(framework, vertices, crossings, arcs, pdcode)
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "ok",
                "framework": framework,
                "time_s": elapsed,
                "repeats": 1,
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


def _run_with_timeout(framework: str, processor: PDCode, pdcode: str, timeout_s: float):
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


def _evaluate_pair(processor: PDCode, pdcode: str, timeout_s: float, active: dict[str, bool]):
    results = {}
    for framework in ("knottedgraph", "topoly"):
        if active[framework]:
            results[framework] = _run_with_timeout(framework, processor, pdcode, timeout_s)
        else:
            results[framework] = {
                "status": "skipped_after_censor_frontier",
                "framework": framework,
            }
    return results


def _finalize_row(row: dict, results: dict) -> dict:
    kg = results["knottedgraph"]
    tp = results["topoly"]
    row.update(
        {
            "knottedgraph_status": kg["status"],
            "topoly_status": tp["status"],
            "knottedgraph_s": kg.get("time_s"),
            "topoly_s": tp.get("time_s"),
            "knottedgraph_repeats": kg.get("repeats"),
            "topoly_repeats": tp.get("repeats"),
        }
    )
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


def crossing_row(crossing_count: int, graph_index: int, timeout_s: float, active: dict[str, bool]):
    graph, processor, pdcode = _prepare_crossing(crossing_count, graph_index)
    row = {
        "family": CROSSING_FAMILY,
        "sample_kind": "topology",
        "size": crossing_count,
        "sample": graph_index,
        "background_rungs": 3 + graph_index,
        "embedding_hash": _embedding_hash(graph),
        "pd_hash": hashlib.sha256(pdcode.encode()).hexdigest(),
        "pd_length": len(pdcode),
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "crossings": len(processor.crossings),
        "connected": True,
        "regular_degree": 3,
        "timeout_s": timeout_s,
    }
    return _finalize_row(row, _evaluate_pair(processor, pdcode, timeout_s, active))


def k4_row(vertex_count: int, embedding: int, base_seed: int, timeout_s: float, active: dict[str, bool]):
    graph, processor, pdcode, seed = _prepare_k4(vertex_count, embedding, base_seed)
    row = {
        "family": K4_FAMILY,
        "sample_kind": "embedding",
        "size": vertex_count,
        "embedding": embedding,
        "embedding_seed": seed,
        "embedding_hash": _embedding_hash(graph),
        "pd_hash": hashlib.sha256(pdcode.encode()).hexdigest(),
        "pd_length": len(pdcode),
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "crossings": len(processor.crossings),
        "timeout_s": timeout_s,
    }
    return _finalize_row(row, _evaluate_pair(processor, pdcode, timeout_s, active))


def _update_censoring(
    family: str,
    x_value: int,
    rows: list[dict],
    active: dict[str, bool],
    consecutive: dict[str, int],
    frontier: int,
):
    for framework in ("knottedgraph", "topoly"):
        if not active[framework]:
            continue
        statuses = [row[f"{framework}_status"] for row in rows]
        if all(status in {"timeout", "error"} for status in statuses):
            consecutive[framework] += 1
        else:
            consecutive[framework] = 0
        if consecutive[framework] >= frontier:
            active[framework] = False
            print(f"CENSOR_FRONTIER={family}:{framework}:{x_value}", flush=True)


def main(
    timeout_s: float,
    profile: str,
    crossing_graphs: int,
    k4_embeddings: int,
    base_seed: int,
    censor_frontier: int,
):
    if censor_frontier < 1:
        raise ValueError("censor_frontier must be >= 1")
    plan = paper_plan(profile, crossing_graphs, k4_embeddings)
    print(
        "CONFIG=" + json.dumps(
            {
                "profile": profile,
                "families": list(plan),
                "crossing_graphs_per_c": crossing_graphs,
                "k4_embeddings_per_v": k4_embeddings,
                "timeout_s": timeout_s,
                "base_seed": base_seed,
                "censor_frontier": censor_frontier,
                "timed_repeats_per_sample": 1,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    all_rows: list[dict] = []

    # Figure 1: same panel of heterogeneous graph sizes at every c.
    family = CROSSING_FAMILY
    active = {"knottedgraph": True, "topoly": True}
    consecutive = {"knottedgraph": 0, "topoly": 0}
    print(f"FAMILY={family}", flush=True)
    expected_ve = None
    for c in plan[family]["x_values"]:
        x_rows = []
        for graph_index in range(crossing_graphs):
            row = crossing_row(c, graph_index, timeout_s, active)
            all_rows.append(row)
            x_rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)

        ve = [(int(row["V"]), int(row["E"])) for row in x_rows]
        if len(set(ve)) != crossing_graphs:
            raise AssertionError(f"{family}/c={c}: graph sizes are not all distinct")
        if expected_ve is None:
            expected_ve = ve
        elif ve != expected_ve:
            raise AssertionError("V/E panel changed with crossing count")

        _update_censoring(family, c, x_rows, active, consecutive, censor_frontier)
        if not any(active.values()):
            break

    # Figure 2: K4-component input-size scaling.
    family = K4_FAMILY
    active = {"knottedgraph": True, "topoly": True}
    consecutive = {"knottedgraph": 0, "topoly": 0}
    print(f"FAMILY={family}", flush=True)
    for vertex_count in plan[family]["x_values"]:
        x_rows = []
        hashes = set()
        for embedding in range(k4_embeddings):
            row = k4_row(vertex_count, embedding, base_seed, timeout_s, active)
            all_rows.append(row)
            x_rows.append(row)
            hashes.add(row["embedding_hash"])
            print(json.dumps(row, separators=(",", ":")), flush=True)
        if len(hashes) != k4_embeddings:
            raise AssertionError(f"{family}/V={vertex_count}: duplicate embedding geometry")
        _update_censoring(
            family,
            vertex_count,
            x_rows,
            active,
            consecutive,
            censor_frontier,
        )
        if not any(active.values()):
            break

    print("SUMMARY=" + json.dumps(all_rows, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--crossing-graphs", type=int, default=DEFAULT_CROSSING_GRAPHS)
    parser.add_argument("--k4-embeddings", type=int, default=DEFAULT_K4_EMBEDDINGS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--censor-frontier", type=int, default=2)
    args = parser.parse_args()
    main(
        args.timeout,
        args.profile,
        args.crossing_graphs,
        args.k4_embeddings,
        args.seed,
        args.censor_frontier,
    )
