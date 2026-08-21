from __future__ import annotations

import statistics
import time

import networkx as nx
import numpy as np

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_frontier import compute_diagram_frontier_laurent
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator, native_available
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode


def _line(a, b, samples=2):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    t = np.linspace(0.0, 1.0, int(samples))[:, None]
    return (1.0 - t) * a + t * b


def _resample(points, samples=121):
    points = np.asarray(points, dtype=float)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    target = np.linspace(0.0, cumulative[-1], int(samples))
    out = np.empty((len(target), 3), dtype=float)
    for dim in range(3): out[:, dim] = np.interp(target, cumulative, points[:, dim])
    return out


def _infinity_plus_pair(base_path, width, depth, samples=120):
    base = _resample(base_path, samples=samples)
    xy = base[:, :2]
    tangent = np.gradient(xy, axis=0)
    norm = np.linalg.norm(tangent, axis=1); norm[norm < 1e-12] = 1.0
    tangent = tangent / norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    t = np.linspace(0.0, 1.0, len(base))
    lateral = np.sin(2.0 * np.pi * t) * np.sin(np.pi * t)
    z_profile = np.exp(-((t - 0.5) / 0.070) ** 2) * np.sin(np.pi * t) ** 2
    over = base.copy(); under = base.copy()
    over[:, :2] += float(width) * lateral[:, None] * normal
    under[:, :2] -= float(width) * lateral[:, None] * normal
    over[:, 2] += float(depth) * z_profile
    under[:, 2] -= float(depth) * z_profile
    over[0] = under[0] = base[0]; over[-1] = under[-1] = base[-1]
    return over, under


def reference_lllv_cycle_graph(n):
    radius = 2.05
    angles = np.pi / 4 + 2.0 * np.pi * np.arange(n) / n
    positions = {k: np.array([radius*np.cos(a), radius*np.sin(a), 0.0]) for k, a in enumerate(angles)}
    graph = nx.MultiGraph()
    for k, pos in positions.items(): graph.add_node(k, pos=pos.copy())
    side = float(np.linalg.norm(positions[1] - positions[0]))
    width = min(0.26, 0.13 * side); depth = min(0.18, 0.085 * side)
    for k in range(n):
        j = (k + 1) % n
        over, under = _infinity_plus_pair(_line(positions[k], positions[j]), width, depth)
        graph.add_edge(k, j, pts=over, crossing_role="over")
        graph.add_edge(k, j, pts=under, crossing_role="under")
    return graph


def prepared_cycle(n):
    processor = PDCode(reference_lllv_cycle_graph(n))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    if len(processor.crossings) != n: raise AssertionError((n, len(processor.crossings)))
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices, yamada.crossings, yamada.arcs, _ordered_crossing_ports
    )
    prepared, _moves = prepared.reduce_reidemeister_ii()
    return prepared


def median_value(fn, repeats):
    times=[]; value=None; stats=None
    for _ in range(repeats):
        start=time.perf_counter(); value, stats = fn(); times.append(time.perf_counter()-start)
    return statistics.median(times), times, value, stats


def small_exact_gate():
    for n in (3, 4, 5, 6):
        prepared=prepared_cycle(n)
        evaluator=NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        expected=evaluator.compute_prepared_bulk_laurent(prepared)
        stats={}; actual=compute_diagram_frontier_laurent(prepared, stats=stats)
        if actual != expected:
            raise AssertionError(f"small exact gate n={n} failed: {stats}\nexpected={expected}\nactual={actual}")
        print(f"small_exact n={n} frontier={stats['max_frontier']} states={stats['max_states']} PASS")


def benchmark(n):
    prepared = prepared_cycle(n)
    def production():
        evaluator=NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        value=evaluator.compute_prepared_laurent(prepared)
        return value, evaluator.last_structural_stats
    def frontier():
        stats={}; value=compute_diagram_frontier_laurent(prepared, stats=stats)
        return value, stats
    prod_t, prod_ts, prod_v, prod_stats = median_value(production, 2)
    front_t, front_ts, front_v, front_stats = median_value(frontier, 5)
    if prod_v != front_v:
        raise AssertionError(
            f"cycle:{n}: direct diagram-frontier mismatch\nproduction={prod_v}\nfrontier={front_v}\nstats={front_stats}"
        )
    print(f"cycle={n} production_s={prod_t:.9f} frontier_s={front_t:.9f} speedup={prod_t/front_t:.6f}x")
    print(f"  production_times={prod_ts}")
    print(f"  frontier_times={front_ts}")
    print(f"  production_stats={prod_stats}")
    print(f"  frontier_stats={front_stats}")
    print("  exact=PASS")


def main():
    if not native_available(): raise RuntimeError("native backend unavailable")
    small_exact_gate()
    for n in (10, 11): benchmark(n)


if __name__ == "__main__": main()
