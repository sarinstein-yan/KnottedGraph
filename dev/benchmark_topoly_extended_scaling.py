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
DEFAULT_TIMEOUT_S = 10.0


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


def _crossing_theta_component(
    y_offset: float, sign: float
) -> tuple[np.ndarray, ...]:
    curves = [
        np.array(
            [[-2, 0, 0], [-1, -1, 0.5 * sign], [1, 1, 0.5 * sign], [2, 0, 0]],
            dtype=float,
        ),
        np.array(
            [[-2, 0, 0], [-1, 1, -0.5 * sign], [1, -1, -0.5 * sign], [2, 0, 0]],
            dtype=float,
        ),
        np.array(
            [[-2, 0, 0], [-1, 2, 0], [1, 2, 0], [2, 0, 0]],
            dtype=float,
        ),
    ]
    shifted = []
    for points in curves:
        points = points.copy()
        points[:, 1] += y_offset
        shifted.append(points)
    return tuple(shifted)


def _decomposable_crossings(crossing_count: int) -> nx.MultiGraph:
    """Large-c throughput family: c independent one-crossing theta components."""
    graph = nx.MultiGraph()
    for component in range(crossing_count):
        y_offset = 5.0 * component
        sign = 1.0 if component % 2 == 0 else -1.0
        left = f"u{component}"
        right = f"v{component}"
        graph.add_node(left, pos=np.array([-2.0, y_offset, 0.0]))
        graph.add_node(right, pos=np.array([2.0, y_offset, 0.0]))
        for points in _crossing_theta_component(y_offset, sign):
            graph.add_edge(left, right, pts=points)
    return graph


def _fixed_size_crossings(crossing_count: int) -> nx.MultiGraph:
    """Connected theta with V=2, E=3 and exactly c projected crossings."""
    graph = nx.MultiGraph()
    left = -float(crossing_count + 2)
    right = float(crossing_count + 2)
    graph.add_node("u", pos=np.array([left, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([right, 0.0, 0.0]))

    x = np.linspace(left, right, crossing_count + 3)
    y1 = np.zeros(crossing_count + 3)
    y2 = np.zeros(crossing_count + 3)
    for index in range(1, crossing_count + 2):
        sign = 1.0 if index % 2 else -1.0
        y1[index] = sign
        y2[index] = -sign
    z1 = np.full(crossing_count + 3, 0.5)
    z2 = np.full(crossing_count + 3, -0.5)
    strand1 = np.column_stack([x, y1, z1])
    strand2 = np.column_stack([x, y2, z2])
    strand1[[0, -1], 2] = 0.0
    strand2[[0, -1], 2] = 0.0
    third = np.array(
        [
            [left, 0, 0],
            [left + 1, 3, 0],
            [right - 1, 3, 0],
            [right, 0, 0],
        ],
        dtype=float,
    )
    graph.add_edge("u", "v", pts=strand1)
    graph.add_edge("u", "v", pts=strand2)
    graph.add_edge("u", "v", pts=third)
    return graph


def _edge_theta(edge_count: int) -> nx.MultiGraph:
    """Edge scaling with V=2 and c=0."""
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    for offset in np.linspace(-4.0, 4.0, edge_count):
        graph.add_edge(
            "u",
            "v",
            pts=np.array(
                [
                    [-2, 0, 0],
                    [-1, float(offset), 0],
                    [1, float(offset), 0],
                    [2, 0, 0],
                ],
                dtype=float,
            ),
        )
    return graph


def _k4_components(vertex_count: int) -> nx.MultiGraph:
    """Trivalent c=0 input-size family, with E=3V/2."""
    if vertex_count % 4:
        raise ValueError("vertex_count must be divisible by four")
    graph = nx.MultiGraph()
    local = {
        0: np.array([-1.0, -1.0, 0.0]),
        1: np.array([1.0, -1.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([0.0, 0.0, 0.0]),
    }
    for component in range(vertex_count // 4):
        offset = np.array([4.0 * component, 0.0, 0.0])
        nodes = [4 * component + index for index in range(4)]
        for index, node in enumerate(nodes):
            graph.add_node(node, pos=local[index] + offset)
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = nodes[i], nodes[j]
                graph.add_edge(
                    u,
                    v,
                    pts=np.vstack(
                        [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
                    ),
                )
    return graph


def _prism(rung_count: int) -> nx.MultiGraph:
    abstract = nx.circular_ladder_graph(rung_count)
    positions = nx.planar_layout(abstract, scale=5.0)
    graph = nx.MultiGraph()
    for node, xy in positions.items():
        graph.add_node(
            node, pos=np.array([float(xy[0]), float(xy[1]), 0.0])
        )
    for u, v in abstract.edges():
        graph.add_edge(
            u,
            v,
            pts=np.vstack([graph.nodes[u]["pos"], graph.nodes[v]["pos"]]),
        )
    return graph


def _build_graph(case: Case) -> nx.MultiGraph:
    builders = {
        "crossings_fixed": _fixed_size_crossings,
        "crossings_throughput": _decomposable_crossings,
        "edges_theta": _edge_theta,
        "vertices_k4": _k4_components,
        "connected_prism": _prism,
    }
    return builders[case.family](case.size)


def _expected_crossings(case: Case) -> int:
    if case.family in {"crossings_fixed", "crossings_throughput"}:
        return case.size
    return 0


def _prepare(case: Case):
    graph = _build_graph(case)
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    crossings = len(processor.crossings)
    expected = _expected_crossings(case)
    if crossings != expected:
        raise AssertionError(
            f"{case.family}/{case.size}: expected {expected} crossings, got {crossings}"
        )
    if case.family == "crossings_fixed" and (
        graph.number_of_nodes(),
        graph.number_of_edges(),
    ) != (2, 3):
        raise AssertionError("fixed crossing family must keep V=2, E=3")
    if case.family == "edges_theta" and graph.number_of_nodes() != 2:
        raise AssertionError("edge family must keep V=2")
    if case.family == "vertices_k4" and graph.number_of_nodes() != case.size:
        raise AssertionError("vertex family size must equal V")
    return graph, processor, pdcode


def _repeats(case: Case, mode: str) -> int:
    # Pipeline repeats are deliberately no larger than invariant-only repeats:
    # every repetition rebuilds PDCode from the same spatial graph.
    if case.family == "crossings_fixed":
        base = 5 if case.size <= 4 else (3 if case.size <= 8 else 1)
    elif case.family == "crossings_throughput":
        base = 5 if case.size <= 10 else (3 if case.size <= 30 else 1)
    elif case.family == "edges_theta":
        base = 5 if case.size <= 20 else 3
    elif case.family == "vertices_k4":
        base = 5 if case.size <= 32 else 3
    else:
        base = 3 if case.size <= 8 else 1
    return min(base, 3) if mode == "pipeline" else base


def _median_time(fn, repeats: int):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def _kg_invariant(processor):
    return Yamada.from_PDCode(processor).compute(
        A, normalize=False, n_jobs=1, method="negami"
    )


def _topoly_invariant(pdcode: str):
    from topoly.invariants import Invariant, YamadaGraph

    Invariant.known["Yamada"] = {}
    return YamadaGraph(pdcode).point(max_cross=500)


def _kg_pipeline(graph: nx.MultiGraph):
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    answer = _kg_invariant(processor)
    return pdcode, len(processor.crossings), answer


def _topoly_pipeline(graph: nx.MultiGraph):
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    answer = _topoly_invariant(pdcode)
    return pdcode, len(processor.crossings), answer


def _worker(
    framework: str,
    mode: str,
    family: str,
    size: int,
    queue,
):
    try:
        case = Case(family, size)
        graph, processor, pdcode = _prepare(case)
        expected_pd_hash = hashlib.sha256(pdcode.encode()).hexdigest()
        repeats = _repeats(case, mode)

        if mode == "invariant":
            if framework == "knottedgraph":
                elapsed, answer = _median_time(
                    lambda: _kg_invariant(processor), repeats
                )
                terms = _kg_terms(answer)
            elif framework == "topoly":
                elapsed, answer = _median_time(
                    lambda: _topoly_invariant(pdcode), repeats
                )
                terms = _topoly_terms(answer)
            else:
                raise ValueError(framework)
            measured_pdcode = pdcode
            measured_crossings = len(processor.crossings)

        elif mode == "pipeline":
            if framework == "knottedgraph":
                elapsed, result = _median_time(
                    lambda: _kg_pipeline(graph), repeats
                )
                measured_pdcode, measured_crossings, answer = result
                terms = _kg_terms(answer)
            elif framework == "topoly":
                elapsed, result = _median_time(
                    lambda: _topoly_pipeline(graph), repeats
                )
                measured_pdcode, measured_crossings, answer = result
                terms = _topoly_terms(answer)
            else:
                raise ValueError(framework)
        else:
            raise ValueError(mode)

        measured_pd_hash = hashlib.sha256(measured_pdcode.encode()).hexdigest()
        if measured_pd_hash != expected_pd_hash:
            raise AssertionError(
                f"{framework}/{mode}: PDCode changed across repeated construction"
            )
        if measured_crossings != len(processor.crossings):
            raise AssertionError(
                f"{framework}/{mode}: crossing count changed across repeated construction"
            )

        queue.put(
            {
                "status": "ok",
                "framework": framework,
                "mode": mode,
                "time_s": elapsed,
                "terms": terms,
                "pd_hash": measured_pd_hash,
                "pd_length": len(measured_pdcode),
                "V": graph.number_of_nodes(),
                "E": graph.number_of_edges(),
                "crossings": measured_crossings,
                "repeats": repeats,
            }
        )
    except BaseException as exc:  # pragma: no cover - benchmark diagnostics
        queue.put(
            {
                "status": "error",
                "framework": framework,
                "mode": mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_with_timeout(
    framework: str,
    mode: str,
    case: Case,
    timeout_s: float,
):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(framework, mode, case.family, case.size, queue),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {
            "status": "timeout",
            "framework": framework,
            "mode": mode,
            "timeout_s": timeout_s,
        }
    if not queue.empty():
        return queue.get()
    return {
        "status": "error",
        "framework": framework,
        "mode": mode,
        "error": (
            f"worker exited with code {process.exitcode} without returning data"
        ),
    }


def _apply_pair(row, mode: str, kg, tp):
    prefix = f"{mode}_"
    row[f"{prefix}knottedgraph_status"] = kg["status"]
    row[f"{prefix}topoly_status"] = tp["status"]
    row[f"{prefix}knottedgraph_s"] = kg.get("time_s")
    row[f"{prefix}topoly_s"] = tp.get("time_s")

    for result in (kg, tp):
        if result["status"] == "error":
            row[f"{prefix}{result['framework']}_error"] = result.get("error")

    if kg["status"] == "ok" and tp["status"] == "ok":
        if kg["pd_hash"] != tp["pd_hash"]:
            raise AssertionError(
                f"{mode}: KnottedGraph and Topoly did not use the same PD string"
            )
        sign, orientation, shift = _validate_laurent_unit(
            kg["terms"], tp["terms"]
        )
        row.update(
            {
                f"{prefix}pd_hash": kg["pd_hash"],
                f"{prefix}unit_sign_topoly_over_kg": sign,
                f"{prefix}variable_orientation": orientation,
                f"{prefix}monomial_shift_topoly_minus_kg": shift,
                f"{prefix}topoly_over_kg": tp["time_s"] / kg["time_s"],
                f"{prefix}kg_over_topoly": kg["time_s"] / tp["time_s"],
                f"{prefix}coefficient_count": len(_sequence(kg["terms"])),
                f"{prefix}correctness": "PASS",
            }
        )
    else:
        row[f"{prefix}correctness"] = (
            "not-evaluated-after-timeout-or-error"
        )


def _row(case: Case, timeout_s: float):
    invariant_kg = _run_with_timeout(
        "knottedgraph", "invariant", case, timeout_s
    )
    invariant_tp = _run_with_timeout("topoly", "invariant", case, timeout_s)
    pipeline_kg = _run_with_timeout(
        "knottedgraph", "pipeline", case, timeout_s
    )
    pipeline_tp = _run_with_timeout("topoly", "pipeline", case, timeout_s)

    row = {
        "family": case.family,
        "size": case.size,
        "timeout_s": timeout_s,
    }

    metadata = next(
        (
            result
            for result in (
                invariant_kg,
                invariant_tp,
                pipeline_kg,
                pipeline_tp,
            )
            if result.get("status") == "ok"
        ),
        None,
    )
    if metadata is not None:
        for key in ("V", "E", "crossings", "pd_length"):
            row[key] = metadata[key]

    _apply_pair(row, "invariant", invariant_kg, invariant_tp)
    _apply_pair(row, "pipeline", pipeline_kg, pipeline_tp)

    # When both modes complete, each framework must also reproduce its own
    # invariant result exactly from the full graph->PD->Yamada workflow.
    if invariant_kg["status"] == "ok" and pipeline_kg["status"] == "ok":
        if invariant_kg["pd_hash"] != pipeline_kg["pd_hash"]:
            raise AssertionError("KnottedGraph invariant/pipeline PD mismatch")
        if invariant_kg["terms"] != pipeline_kg["terms"]:
            raise AssertionError(
                "KnottedGraph invariant-only and pipeline polynomials differ"
            )
        row["knottedgraph_cross_mode_correctness"] = "PASS"

    if invariant_tp["status"] == "ok" and pipeline_tp["status"] == "ok":
        if invariant_tp["pd_hash"] != pipeline_tp["pd_hash"]:
            raise AssertionError("Topoly invariant/pipeline PD mismatch")
        if invariant_tp["terms"] != pipeline_tp["terms"]:
            raise AssertionError(
                "Topoly invariant-only and pipeline polynomials differ"
            )
        row["topoly_cross_mode_correctness"] = "PASS"

    return row


def _cases():
    fixed_c = list(range(1, 16)) + [18, 20, 24, 30]
    throughput_c = (
        list(range(1, 21)) + [25, 30, 40, 50, 60, 80, 100]
    )
    edges = [
        3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 150, 200
    ]
    vertices = [
        4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512
    ]
    prisms = list(range(3, 11)) + [12, 14, 16, 18, 20, 24, 28, 30]
    return {
        "crossings_fixed": [Case("crossings_fixed", value) for value in fixed_c],
        "crossings_throughput": [
            Case("crossings_throughput", value) for value in throughput_c
        ],
        "edges_theta": [Case("edges_theta", value) for value in edges],
        "vertices_k4": [Case("vertices_k4", value) for value in vertices],
        "connected_prism": [
            Case("connected_prism", value) for value in prisms
        ],
    }


def _pair_completed(row, mode: str) -> bool:
    return (
        row[f"{mode}_knottedgraph_status"] == "ok"
        and row[f"{mode}_topoly_status"] == "ok"
    )


def main(timeout_s: float):
    rows = []
    for family, cases in _cases().items():
        consecutive_censored = 0
        print(f"FAMILY={family}", flush=True)
        for case in cases:
            row = _row(case, timeout_s)
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)

            # Stop only the difficult crossing/connected families after both
            # comparisons are repeatedly censored. Cheap control families run
            # their entire requested range.
            both_modes_paired = _pair_completed(
                row, "invariant"
            ) and _pair_completed(row, "pipeline")
            if both_modes_paired:
                consecutive_censored = 0
            else:
                consecutive_censored += 1

            if (
                family in {
                    "crossings_fixed",
                    "crossings_throughput",
                    "connected_prism",
                }
                and consecutive_censored >= 2
            ):
                break

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="hard wall-time limit per framework/mode/case in seconds",
    )
    args = parser.parse_args()
    main(args.timeout)
