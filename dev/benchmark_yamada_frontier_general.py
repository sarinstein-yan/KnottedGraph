from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics
import sys
import time

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_frontier import (
    FrontierLimitExceeded,
    compute_diagram_frontier_laurent,
    plan_diagram_frontier,
)
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator, native_available
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode

HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepared(processor: PDCode):
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices, yamada.crossings, yamada.arcs, _ordered_crossing_ports
    )
    return prepared.reduce_reidemeister_ii()[0]


def _mirror_graph(graph):
    mirrored = graph.copy()
    for _node, data in mirrored.nodes(data=True):
        if "pos" in data:
            data["pos"] = data["pos"].copy(); data["pos"][2] *= -1.0
    for _u, _v, _key, data in mirrored.edges(keys=True, data=True):
        if "pts" in data:
            data["pts"] = data["pts"].copy(); data["pts"][:, 2] *= -1.0
    return mirrored


def cases():
    controlled = _load("frontier_controlled", "benchmark_topoly_paper_scaling.py")
    for crossings in (8, 16, 32, 40):
        _g, processor, _pd = controlled._prepare_crossing(crossings, 4)
        yield f"controlled:{crossings}", _prepared(processor)

    torus = _load("frontier_torus", "benchmark_topoly_essential_torus_scaling.py")
    for n in (9, 11):
        _g, processor, _pd = torus.prepare_essential_torus(n)
        yield f"torus:{n}", _prepared(processor)
    graph = _mirror_graph(torus.essential_torus_graph(11))
    processor = PDCode(graph); processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    yield "torus:11:mirror", _prepared(processor)

    random_mod = _load("frontier_random", "benchmark_topoly_random_cubic_ensemble.py")
    ensemble = random_mod.topology_ensemble(20, 3, random_mod.DEFAULT_SEED)
    for sample_index, (sample, abstract) in enumerate(ensemble):
        _embedded, processor, _pd, _attempt = random_mod.prepare_sample(
            sample, abstract, random_mod.DEFAULT_SEED
        )
        yield f"random20:{sample_index}:c{len(processor.crossings)}", _prepared(processor)


def _median(fn, repeats):
    values=[]; answer=None; stats=None
    for _ in range(repeats):
        start=time.perf_counter(); answer, stats=fn(); values.append(time.perf_counter()-start)
    return statistics.median(values), values, answer, stats


def main():
    if not native_available():
        raise RuntimeError("native backend unavailable")
    for name, prepared in cases():
        plan = plan_diagram_frontier(prepared)
        print(
            f"PLAN {name} c={len(prepared.crossing_ids)} factors={plan['factor_count']} "
            f"peak_ports={plan['peak_ports']} boundary={plan['max_boundary_ports']}"
        )
        # Keep the probe bounded. High-width cases are expected to remain on the
        # production backend; this is testing adaptive selectivity, not forcing DP.
        if plan["peak_ports"] > 12:
            print(f"SKIP {name} frontier: structural width guard")
            continue

        def production():
            ev=NativeCompactEvaluator(PythonCompactYamadaEvaluator)
            return ev.compute_prepared_laurent(prepared), ev.last_structural_stats

        def frontier():
            stats={}
            value=compute_diagram_frontier_laurent(
                prepared, max_states=200_000, max_peak_ports=12, stats=stats
            )
            return value, stats

        prod_t, prod_ts, prod_v, prod_stats = _median(production, 1)
        try:
            front_t, front_ts, front_v, front_stats = _median(frontier, 2)
        except FrontierLimitExceeded as exc:
            print(f"ABORT {name} frontier: {exc}")
            continue
        if prod_v != front_v:
            raise AssertionError(
                f"{name}: frontier mismatch\nproduction={prod_v}\nfrontier={front_v}\n{front_stats}"
            )
        print(
            f"RESULT {name} production_s={prod_t:.9f} frontier_s={front_t:.9f} "
            f"speedup={prod_t/front_t:.6f}x exact=PASS"
        )
        print(f"  production_times={prod_ts}")
        print(f"  frontier_times={front_ts}")
        print(f"  frontier_states={front_stats['max_states']} transitions={front_stats['transitions']}")
        print(f"  production_stats={prod_stats}")


if __name__ == "__main__":
    main()
