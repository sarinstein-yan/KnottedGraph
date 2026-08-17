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

from benchmark_yamada_end_to_end import multi_crossing_theta
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
        if tp_seq != expected:
            continue
        if kg_terms and topoly_terms:
            kg_anchor = min(kg_terms) if orientation == 1 else -max(kg_terms)
            shift = min(topoly_terms) - kg_anchor
        else:
            shift = 0
        return sign, orientation, shift
    raise AssertionError(
        "Topoly and KnottedGraph differ beyond ±A^k and A<->A^-1: "
        f"KG={kg_seq}, Topoly={tp_seq}"
    )


def _embedded_theta(edge_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([-2.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    offsets = np.linspace(-4.0, 4.0, edge_count)
    for offset in offsets:
        pts = np.array(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, float(offset), 0.0],
                [1.0, float(offset), 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        graph.add_edge("u", "v", pts=pts)
    return graph


def _embedded_cycle(vertex_count: int) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    angles = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
    points = {
        i: np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
        for i, angle in enumerate(angles)
    }
    for node, point in points.items():
        graph.add_node(node, pos=point)
    for node in range(vertex_count):
        nxt = (node + 1) % vertex_count
        graph.add_edge(node, nxt, pts=np.vstack([points[node], points[nxt]]))
    return graph


def _embedded_prism(rung_count: int) -> nx.MultiGraph:
    abstract = nx.circular_ladder_graph(rung_count)
    if not nx.check_planarity(abstract)[0]:
        raise AssertionError("Circular ladder must be planar")
    positions_2d = nx.planar_layout(abstract, scale=5.0)
    graph = nx.MultiGraph()
    for node, xy in positions_2d.items():
        point = np.array([float(xy[0]), float(xy[1]), 0.0])
        graph.add_node(node, pos=point)
    for u, v in abstract.edges():
        graph.add_edge(
            u,
            v,
            pts=np.vstack([graph.nodes[u]["pos"], graph.nodes[v]["pos"]]),
        )
    return graph


def _build_graph(case: Case) -> nx.MultiGraph:
    if case.family == "crossings":
        return multi_crossing_theta(case.size)
    if case.family == "edges_theta":
        return _embedded_theta(case.size)
    if case.family == "vertices_cycle":
        return _embedded_cycle(case.size)
    if case.family == "connected_prism":
        return _embedded_prism(case.size)
    raise ValueError(case.family)


def _expected_crossings(case: Case) -> int:
    return case.size if case.family == "crossings" else 0


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
    return graph, processor, pdcode


def _repeats(case: Case) -> int:
    if case.family == "crossings":
        return 5 if case.size <= 6 else (3 if case.size <= 16 else 1)
    if case.family == "edges_theta":
        return 5 if case.size <= 20 else 3
    if case.family == "vertices_cycle":
        return 5 if case.size <= 64 else 3
    return 3 if case.size <= 8 else 1


def _median_time(fn, repeats: int):
    elapsed = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed), answer


def _worker(framework: str, family: str, size: int, queue):
    try:
        case = Case(family, size)
        graph, processor, pdcode = _prepare(case)
        pd_hash = hashlib.sha256(pdcode.encode("utf-8")).hexdigest()
        repeats = _repeats(case)

        if framework == "knottedgraph":
            def run():
                return Yamada.from_PDCode(processor).compute(
                    A, normalize=False, n_jobs=1, method="negami"
                )

            elapsed, answer = _median_time(run, repeats)
            terms = _kg_terms(answer)
        elif framework == "topoly":
            from topoly.invariants import Invariant, YamadaGraph

            def run():
                Invariant.known["Yamada"] = {}
                return YamadaGraph(pdcode).point(max_cross=500)

            elapsed, answer = _median_time(run, repeats)
            terms = _topoly_terms(answer)
        else:
            raise ValueError(framework)

        queue.put(
            {
                "status": "ok",
                "framework": framework,
                "time_s": elapsed,
                "terms": terms,
                "pd_hash": pd_hash,
                "pd_length": len(pdcode),
                "V": graph.number_of_nodes(),
                "E": graph.number_of_edges(),
                "crossings": len(processor.crossings),
                "repeats": repeats,
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


def _run_with_timeout(framework: str, case: Case, timeout_s: float):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(framework, case.family, case.size, queue),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {
            "status": "timeout",
            "framework": framework,
            "timeout_s": timeout_s,
        }
    if not queue.empty():
        return queue.get()
    return {
        "status": "error",
        "framework": framework,
        "error": f"worker exited with code {process.exitcode} without returning data",
    }


def _row(case: Case, timeout_s: float):
    kg = _run_with_timeout("knottedgraph", case, timeout_s)
    tp = _run_with_timeout("topoly", case, timeout_s)

    row = {
        "family": case.family,
        "size": case.size,
        "timeout_s": timeout_s,
        "knottedgraph_status": kg["status"],
        "topoly_status": tp["status"],
        "knottedgraph_s": kg.get("time_s"),
        "topoly_s": tp.get("time_s"),
    }

    metadata = kg if kg.get("status") == "ok" else tp
    for key in ("V", "E", "crossings", "pd_length"):
        if key in metadata:
            row[key] = metadata[key]

    if kg["status"] == "error":
        row["knottedgraph_error"] = kg.get("error")
    if tp["status"] == "error":
        row["topoly_error"] = tp.get("error")

    if kg["status"] == "ok" and tp["status"] == "ok":
        if kg["pd_hash"] != tp["pd_hash"]:
            raise AssertionError(
                f"{case}: frameworks reconstructed different PD inputs"
            )
        unit = _validate_laurent_unit(kg["terms"], tp["terms"])
        sign, orientation, shift = unit
        row.update(
            {
                "pd_hash": kg["pd_hash"],
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
        row["correctness"] = "not-evaluated-after-timeout-or-error"
    return row


def _cases():
    crossing_sizes = list(range(1, 21)) + [22, 24, 26, 28, 30]
    edge_sizes = [3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 150, 200]
    vertex_sizes = [3, 4, 5, 6, 8, 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    prism_sizes = list(range(3, 11)) + [12, 14, 16, 18, 20, 24, 28, 30]
    return {
        "crossings": [Case("crossings", value) for value in crossing_sizes],
        "edges_theta": [Case("edges_theta", value) for value in edge_sizes],
        "vertices_cycle": [Case("vertices_cycle", value) for value in vertex_sizes],
        "connected_prism": [Case("connected_prism", value) for value in prism_sizes],
    }


def main(timeout_s: float):
    rows = []
    for family, cases in _cases().items():
        consecutive_censored = 0
        print(f"FAMILY={family}")
        for case in cases:
            row = _row(case, timeout_s)
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)

            both_ok = (
                row["knottedgraph_status"] == "ok"
                and row["topoly_status"] == "ok"
            )
            if both_ok:
                consecutive_censored = 0
            else:
                consecutive_censored += 1

            # The structured control families are intentionally cheap and are
            # always sampled through their full requested range. The difficult
            # crossing/prism families stop after two consecutive censored cases
            # so asymptotic blow-up is recorded rather than hanging the suite.
            if family in {"crossings", "connected_prism"} and consecutive_censored >= 2:
                break

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="hard wall-time limit per framework/case in seconds",
    )
    args = parser.parse_args()
    main(args.timeout)
