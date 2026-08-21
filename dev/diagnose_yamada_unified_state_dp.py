from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import sympy as sp

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_frontier import compute_diagram_frontier_laurent
from knotted_graph.invariants.yamada.diagram_structural import _reduce_r1_queue, compute_structural_laurent
from knotted_graph.invariants.yamada.fast import shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode

A = sp.Symbol("A")
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "User_guide" / "benchmarks" / "03_knottedgraph_vs_topoly_scaling_final_push.ipynb"


def _constructors():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace: dict = {}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if all(name in source for name in (
            "def reference_dv_theta_graph",
            "def reference_lllv_cycle_graph",
            "def reference_lllv_theta_graph",
        )):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            return {
                "THETA_N_VALUES": namespace["reference_dv_theta_graph"],
                "LLLV_CYCLE_N_VALUES": namespace["reference_lllv_cycle_graph"],
                "LLLV_THETA_S_VALUES": namespace["reference_lllv_theta_graph"],
            }
    raise RuntimeError("reference constructor cell not found")


def _prepared(family: str, n: int):
    graph = _constructors()[family](n)
    processor = PDCode(graph)
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )
    return prepared


def _root_fixpoint(prepared):
    state = prepared
    shift_total = 0
    rii_total = 0
    r1_total = 0
    while True:
        state, rii = state.reduce_reidemeister_ii()
        state, delta, r1 = _reduce_r1_queue(state)
        shift_total += delta
        rii_total += rii
        r1_total += r1
        if not rii and not r1:
            return state, shift_total, rii_total, r1_total


def _time(fn, repeats: int):
    fn()
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), result


def main(family: str, n: int, repeats: int):
    prepared = _prepared(family, n)

    def bulk():
        ev = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        return ev.compute_prepared_bulk_laurent(prepared)

    structural_stats = {}
    def structural():
        nonlocal structural_stats
        structural_stats = {}
        ev = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        return compute_structural_laurent(prepared, ev, stats=structural_stats)

    frontier_stats = {}
    def frontier():
        nonlocal frontier_stats
        frontier_stats = {}
        return compute_diagram_frontier_laurent(prepared, stats=frontier_stats)

    reduced, exponent, rii_moves, r1_moves = _root_fixpoint(prepared)
    reduced_frontier_stats = {}
    def reduced_frontier():
        nonlocal reduced_frontier_stats
        reduced_frontier_stats = {}
        value = compute_diagram_frontier_laurent(reduced, stats=reduced_frontier_stats)
        return shift(value, exponent)

    production_stats = {}
    def production():
        nonlocal production_stats
        ev = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        value = ev.compute_prepared_laurent(prepared)
        production_stats = ev.last_structural_stats or {}
        return value

    rows = {}
    reference = None
    # Bulk is intentionally skipped beyond 11 crossings; it is only an oracle
    # for smaller diagnostics and is not part of the proposed algorithm.
    methods = [("structural", structural), ("frontier", frontier),
               ("reduced_frontier", reduced_frontier), ("production", production)]
    if len(prepared.crossing_ids) <= 11:
        methods.insert(0, ("bulk", bulk))

    for name, fn in methods:
        elapsed, value = _time(fn, repeats)
        if reference is None:
            reference = value
        assert value == reference, name
        rows[name] = elapsed

    result = {
        "family": family,
        "n": n,
        "crossings": len(prepared.crossing_ids),
        "root_reduced_crossings": len(reduced.crossing_ids),
        "root_rii_moves": rii_moves,
        "root_r1_moves": r1_moves,
        "root_shift": exponent,
        "times": rows,
        "structural_stats": structural_stats,
        "frontier_stats": frontier_stats,
        "reduced_frontier_stats": reduced_frontier_stats,
        "production_stats": production_stats,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("n", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    main(args.family, args.n, args.repeats)
