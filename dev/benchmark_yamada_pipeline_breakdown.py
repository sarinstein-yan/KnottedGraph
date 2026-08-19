from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def median(fn, repeats=5):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), result


def spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    pos = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    out = nx.MultiGraph()
    for node, point in pos.items():
        out.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        out.add_edge(u, v, pts=np.vstack([pos[u], pos[v]]))
    return out


def multi_crossing_theta(component_count=5):
    graph = nx.MultiGraph()
    for component in range(component_count):
        y_offset = 5.0 * component
        sign = 1.0 if component % 2 == 0 else -1.0
        left = f"u{component}"
        right = f"v{component}"
        graph.add_node(left, pos=np.array([-2.0, y_offset, 0.0]))
        graph.add_node(right, pos=np.array([2.0, y_offset, 0.0]))
        curves = [
            np.array([[-2, 0, 0], [-1, -1, 0.5 * sign], [1, 1, 0.5 * sign], [2, 0, 0]], dtype=float),
            np.array([[-2, 0, 0], [-1, 1, -0.5 * sign], [1, -1, -0.5 * sign], [2, 0, 0]], dtype=float),
            np.array([[-2, 0, 0], [-1, 2, 0], [1, 2, 0], [2, 0, 0]], dtype=float),
        ]
        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)
    return graph


def measure_case(name, graph, rotation, expected_crossings):
    def projection_only():
        processor = PDCode(graph)
        processor.compute(rotation_angles=rotation)
        return processor

    projection_s, processor = median(projection_only, 7)
    crossings = len(processor.crossings)
    if crossings != expected_crossings:
        raise AssertionError(f"{name}: expected {expected_crossings} crossings, got {crossings}")
    if len(Yamada.from_PDCode(processor)._diagram_blocks()) != (5 if name == "decomposable_c5" else 1):
        raise AssertionError(f"{name}: unexpected block factorization")

    invariant_s, polynomial = median(
        lambda: Yamada.from_PDCode(processor).compute(A, normalize=False, n_jobs=1, method="negami"),
        7,
    )

    def complete():
        p = PDCode(graph)
        p.compute(rotation_angles=rotation)
        return Yamada.from_PDCode(p).compute(A, normalize=False, n_jobs=1, method="negami")

    total_s, total_poly = median(complete, 7)
    if sp.expand(polynomial - total_poly) != 0:
        raise AssertionError("pipeline breakdown changed result")
    row = {
        "case": name,
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "crossings": crossings,
        "projection_s": projection_s,
        "invariant_s": invariant_s,
        "total_s": total_s,
        "projection_fraction": projection_s / total_s,
        "invariant_fraction": invariant_s / total_s,
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def _theta_generic_optimization_lab():
    """Profile exact generic structural candidates without theorem dispatch."""
    import benchmark_topoly_essential_torus_scaling as torus
    import knotted_graph.invariants.yamada.diagram_structural as ds
    import knotted_graph.invariants.yamada.skein_hybrid as sh
    from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
    from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
    from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder

    original_best_inversion = ds._best_inversion
    original_best_resolution = ds._best_resolution
    original_key = ds.diagram_key
    original_rii = PreparedCompactStateBuilder._find_reidemeister_ii_pair

    def first_inversion(prepared):
        for crossing_index in range(len(prepared.crossing_ids)):
            inverted = sh.invert_crossing(prepared, crossing_index)
            reduced, moves = inverted.reduce_reidemeister_ii()
            if moves:
                return moves, crossing_index, reduced
        return None

    def first_resolution(prepared):
        for crossing_index in range(len(prepared.crossing_ids)):
            children = []
            moves_total = 0
            try:
                for spin in (0, 1, 2):
                    child = sh.resolve_crossing(prepared, crossing_index, spin)
                    reduced, moves = child.reduce_reidemeister_ii()
                    children.append(reduced)
                    moves_total += moves
            except ValueError:
                continue
            if moves_total:
                return moves_total, crossing_index, tuple(children)
        return None

    def adjacency_rii(self):
        arc_partner = self.arc_partner
        crossing_for_port = self.crossing_for_port
        ordered_ports = self.ordered_ports
        for first, first_ports in enumerate(ordered_ports):
            by_second = {}
            for first_position, first_port in enumerate(first_ports):
                partner = arc_partner[first_port]
                second = crossing_for_port[partner]
                if second < 0 or second >= first:
                    continue
                by_second.setdefault(second, []).append((first_position, partner))
            for second, links in by_second.items():
                if len(links) != 2:
                    continue
                second_ports = ordered_ports[second]
                second_position = {port: i for i, port in enumerate(second_ports)}
                shared = []
                for first_position, partner in links:
                    pos = second_position.get(partner)
                    if pos is None:
                        break
                    shared.append((first_position, pos))
                if len(shared) != 2:
                    continue
                if (shared[0][0] - shared[1][0]) % 4 not in (1, 3):
                    continue
                if (shared[0][1] - shared[1][1]) % 4 not in (1, 3):
                    continue
                if any((x % 2) != (y % 2) for x, y in shared):
                    continue
                removed = set(first_ports) | set(second_ports)
                splices = []
                valid = True
                for first_position, second_position_index in shared:
                    first_external = first_ports[(first_position + 2) % 4]
                    second_external = second_ports[(second_position_index + 2) % 4]
                    remote_first = arc_partner[first_external]
                    remote_second = arc_partner[second_external]
                    if remote_first in removed or remote_second in removed or remote_first == remote_second:
                        valid = False
                        break
                    splices.append((remote_first, remote_second))
                if valid and len({p for pair in splices for p in pair}) == 4:
                    return first, second, tuple(splices)
        return None

    def terminal_canonical_key(prepared):
        # Graph vertices are unlabeled. Canonicalize only their integer terminal
        # indices; all crossing/port data remain exact and labeled, so equality
        # of this key is an exact diagram equality modulo vertex renaming.
        terminal_map = {}
        fixed = []
        next_terminal = 0
        for value in prepared.fixed_terminal_index:
            if value < 0:
                fixed.append(-1)
                continue
            if value not in terminal_map:
                terminal_map[value] = next_terminal
                next_terminal += 1
            fixed.append(terminal_map[value])
        return (
            len(prepared.vertex_ids),
            prepared.ordered_ports,
            prepared.arc_partner,
            tuple(fixed),
            prepared.crossing_for_port,
        )

    prepared_by_n = {}
    expected_by_n = {}
    for n in (9, 11, 13, 15, 17):
        _graph, processor, _pd = torus.prepare_essential_torus(n)
        yamada = Yamada.from_PDCode(processor)
        prepared = PreparedCompactStateBuilder.prepare(
            yamada.vertices, yamada.crossings, yamada.arcs, _ordered_crossing_ports
        )
        prepared_by_n[n] = prepared
        expected_by_n[n] = tuple(sorted(torus.independent_theta_terms(n).items()))

    configs = [
        ("baseline", original_best_inversion, original_best_resolution, original_rii, original_key),
        ("first_moves_adjacency_rii", first_inversion, first_resolution, adjacency_rii, original_key),
        ("first_moves_adjacency_rii_terminal_key", first_inversion, first_resolution, adjacency_rii, terminal_canonical_key),
    ]
    print("THETA_GENERIC_OPTIMIZATION_LAB")
    for label, inv, res, rii, key_fn in configs:
        ds._best_inversion = inv
        ds._best_resolution = res
        ds.diagram_key = key_fn
        PreparedCompactStateBuilder._find_reidemeister_ii_pair = rii
        for n, prepared in prepared_by_n.items():
            evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
            stats = {}
            start = time.perf_counter()
            value = ds.compute_structural_laurent(prepared, evaluator, stats=stats)
            elapsed = time.perf_counter() - start
            if value != expected_by_n[n]:
                raise AssertionError((label, n, value, expected_by_n[n]))
            print(json.dumps({"candidate": label, "n": n, "seconds": elapsed, "stats": stats}, separators=(",", ":")))

    # Profile where n=17 actually spends its fallback time using the best cheap
    # candidate. This guides the next algorithmic optimization.
    class TrackingNative(NativeCompactEvaluator):
        def __init__(self):
            super().__init__(PythonCompactYamadaEvaluator)
            self.fallback_counts = Counter()
            self.fallback_seconds = defaultdict(float)
        def compute_prepared_bulk_laurent(self, prepared):
            c = len(prepared.crossing_ids)
            start = time.perf_counter()
            value = super().compute_prepared_bulk_laurent(prepared)
            self.fallback_counts[c] += 1
            self.fallback_seconds[c] += time.perf_counter() - start
            return value

    ds._best_inversion = first_inversion
    ds._best_resolution = first_resolution
    ds.diagram_key = terminal_canonical_key
    PreparedCompactStateBuilder._find_reidemeister_ii_pair = adjacency_rii
    evaluator = TrackingNative()
    stats = {}
    start = time.perf_counter()
    value = ds.compute_structural_laurent(prepared_by_n[17], evaluator, stats=stats)
    elapsed = time.perf_counter() - start
    assert value == expected_by_n[17]
    print(json.dumps({
        "candidate": "n17_fallback_profile",
        "seconds": elapsed,
        "fallback_counts": dict(sorted(evaluator.fallback_counts.items())),
        "fallback_seconds": {str(k): v for k, v in sorted(evaluator.fallback_seconds.items())},
        "fallback_total_seconds": sum(evaluator.fallback_seconds.values()),
        "stats": stats,
    }, separators=(",", ":")))

    ds._best_inversion = original_best_inversion
    ds._best_resolution = original_best_resolution
    ds.diagram_key = original_key
    PreparedCompactStateBuilder._find_reidemeister_ii_pair = original_rii


def main():
    rows = []
    rows.append(measure_case("decomposable_c5", multi_crossing_theta(5), (0.0, 0.0, 0.0), 5))

    k4 = spring_embedding(nx.complete_graph(4), 7)
    rows.append(measure_case("connected_K4", k4, (0.0, 89.70158313251306, 0.0), 1))

    petersen = spring_embedding(nx.petersen_graph(), 9)
    rows.append(
        measure_case(
            "connected_petersen",
            petersen,
            (-134.58074129795634, 55.40942502382338, 0.0),
            6,
        )
    )
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))
    _theta_generic_optimization_lab()


if __name__ == "__main__":
    main()
