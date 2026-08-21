from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import sympy as sp

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_frontier import (
    compute_diagram_frontier_laurent,
    plan_diagram_frontier,
)
from knotted_graph.invariants.yamada.diagram_structural import _reduce_r1_queue, compute_structural_laurent
from knotted_graph.invariants.yamada.diagram_unified import (
    compute_unified_laurent,
    contract_frontier_laurent,
    native_frontier_available,
)
from knotted_graph.invariants.yamada.fast import shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode

try:
    from knotted_graph.invariants.yamada import _yamada_terminal_frontier
except Exception:
    _yamada_terminal_frontier = None

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
    return PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )


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

    native_frontier_stats = {}
    def native_frontier():
        nonlocal native_frontier_stats
        native_frontier_stats = {}
        return contract_frontier_laurent(prepared, stats=native_frontier_stats)

    terminal_frontier_stats = {}
    def terminal_frontier():
        nonlocal terminal_frontier_stats
        if _yamada_terminal_frontier is None:
            raise RuntimeError("terminal frontier extension was not built")
        plan = plan_diagram_frontier(prepared)
        value, max_states, max_terminals, transitions = (
            _yamada_terminal_frontier.compute_prepared_frontier(
                len(prepared.vertex_ids),
                len(prepared.crossing_ids),
                list(prepared.arc_partner),
                list(prepared.fixed_terminal_index),
                list(prepared.crossing_for_port),
                list(prepared.plus_partner),
                list(prepared.minus_partner),
                list(plan["factor_order"]),
            )
        )
        terminal_frontier_stats = {
            "max_states": int(max_states),
            "max_terminals": int(max_terminals),
            "transitions": int(transitions),
        }
        return tuple((int(power), int(coefficient)) for power, coefficient in value)

    reduced, exponent, rii_moves, r1_moves = _root_fixpoint(prepared)
    reduced_frontier_stats = {}
    def reduced_frontier():
        nonlocal reduced_frontier_stats
        reduced_frontier_stats = {}
        value = contract_frontier_laurent(reduced, stats=reduced_frontier_stats)
        return shift(value, exponent)

    unified_stats = {}
    def unified():
        nonlocal unified_stats
        unified_stats = {}
        return compute_unified_laurent(prepared, stats=unified_stats)

    production_stats = {}
    def production():
        nonlocal production_stats
        ev = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        value = ev.compute_prepared_laurent(prepared)
        production_stats = ev.last_structural_stats or {}
        return value

    rows = {}
    reference = None
    methods = [
        ("structural", structural),
        ("frontier_python", frontier),
        ("frontier_native", native_frontier),
        ("terminal_frontier", terminal_frontier),
        ("reduced_frontier_native", reduced_frontier),
        ("unified", unified),
        ("production", production),
    ]
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
        "native_frontier_available": native_frontier_available(),
        "terminal_frontier_available": _yamada_terminal_frontier is not None,
        "root_reduced_crossings": len(reduced.crossing_ids),
        "root_rii_moves": rii_moves,
        "root_r1_moves": r1_moves,
        "root_shift": exponent,
        "times": rows,
        "structural_stats": structural_stats,
        "frontier_stats": frontier_stats,
        "native_frontier_stats": native_frontier_stats,
        "terminal_frontier_stats": terminal_frontier_stats,
        "reduced_frontier_stats": reduced_frontier_stats,
        "unified_stats": unified_stats,
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
