from __future__ import annotations

import json
import random
import statistics
import time

import networkx as nx
import numpy as np

from benchmark_topoly_random_cubic_ensemble import (
    DEFAULT_SEED,
    prepare_sample,
    topology_ensemble,
)
from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.fast import (
    ONE,
    ZERO,
    add,
    multiply,
    multiply_sigma,
    scale,
    shift,
)
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode


def _graph(rows: list[list[int]]) -> CompactGraph:
    return CompactGraph(tuple(tuple(int(value) for value in row) for row in rows))


def _remove_all_loops(graph: CompactGraph) -> tuple[CompactGraph, int]:
    loop_count = sum(graph.rows[i][i] for i in range(graph.n))
    if not loop_count:
        return graph, 0
    rows = [list(row) for row in graph.rows]
    for i in range(graph.n):
        rows[i][i] = 0
    return _graph(rows), loop_count


def _sigma_power(power: int, sign: int = 1):
    value = ONE
    for _ in range(power):
        value = multiply_sigma(value, sign=sign)
    return value


def _degrees(graph: CompactGraph) -> list[int]:
    return [graph.degree(i) for i in range(graph.n)]


def _suppress_degree_two(graph: CompactGraph, vertex: int) -> CompactGraph:
    if graph.rows[vertex][vertex]:
        raise ValueError("degree-two suppression expects a loopless vertex")
    neighbors: list[int] = []
    for other, multiplicity in enumerate(graph.rows[vertex]):
        if other == vertex:
            continue
        neighbors.extend([other] * multiplicity)
    if len(neighbors) != 2:
        raise ValueError("vertex is not degree two")

    left, right = neighbors
    kept = [index for index in range(graph.n) if index != vertex]
    remap = {old: new for new, old in enumerate(kept)}
    rows = [[graph.rows[i][j] for j in kept] for i in kept]
    left_new = remap[left]
    right_new = remap[right]
    if left_new == right_new:
        rows[left_new][left_new] += 1
    else:
        rows[left_new][right_new] += 1
        rows[right_new][left_new] += 1
    return _graph(rows)


def _delete_parallel_class(graph: CompactGraph, i: int, j: int) -> CompactGraph:
    rows = [list(row) for row in graph.rows]
    rows[i][j] = 0
    rows[j][i] = 0
    return _graph(rows)


def _add_edge(graph: CompactGraph, i: int, j: int) -> CompactGraph:
    rows = [list(row) for row in graph.rows]
    rows[i][j] += 1
    if i != j:
        rows[j][i] += 1
    return _graph(rows)


def _identify_vertices(graph: CompactGraph, i: int, j: int) -> CompactGraph:
    return _add_edge(graph, i, j).contract_edge(i, j)


def _parallel_factor(multiplicity: int):
    """Return sum_{j=0}^{k-1} (-sigma)^j exactly."""
    total = ZERO
    power = ONE
    for _ in range(multiplicity):
        total = add(total, power)
        power = multiply_sigma(power, sign=-1)
    return total


class StructuralFlowEvaluator:
    """Exact literature-backed reductions before ordinary deletion--contraction."""

    def __init__(self):
        self.memo: dict[CompactGraph, tuple[tuple[int, int], ...]] = {}
        self.stats = {
            "calls": 0,
            "memo_hits": 0,
            "loops_batched": 0,
            "degree_two_suppressions": 0,
            "low_cyclomatic_zeros": 0,
            "parallel_class_reductions": 0,
            "parallel_edges_collapsed": 0,
            "bridge_zeros": 0,
            "articulation_splits": 0,
            "ordinary_branches": 0,
        }

    def compute_laurent(self, graph: CompactGraph):
        return self._rec(graph)

    def _simplify_homeomorphism(self, graph: CompactGraph):
        factor = ONE
        while True:
            graph, loop_count = _remove_all_loops(graph)
            if loop_count:
                self.stats["loops_batched"] += loop_count
                factor = multiply(factor, _sigma_power(loop_count, sign=-1))

            degrees = _degrees(graph)
            degree_two = next(
                (
                    vertex
                    for vertex, degree in enumerate(degrees)
                    if degree == 2 and not graph.rows[vertex][vertex]
                ),
                None,
            )
            if degree_two is None:
                return graph, factor
            graph = _suppress_degree_two(graph, degree_two)
            self.stats["degree_two_suppressions"] += 1

    def _rec(self, graph: CompactGraph):
        self.stats["calls"] += 1
        graph, factor = self._simplify_homeomorphism(graph)
        if factor != ONE:
            return multiply(factor, self._rec(graph))

        cached = self.memo.get(graph)
        if cached is not None:
            self.stats["memo_hits"] += 1
            return cached

        edge_count, degrees, _loop, edge = graph.scan()
        if edge_count == 0:
            value = (((0, -1),) if graph.n % 2 else ONE)
            self.memo[graph] = value
            return value

        components = graph.components()
        if len(components) > 1:
            value = ONE
            for component in components:
                value = multiply(value, self._rec(graph.induced(component)))
            self.memo[graph] = value
            return value

        if graph.n == 2 and not graph.rows[0][0] and not graph.rows[1][1]:
            theta = graph.rows[0][1]
            if theta == edge_count:
                value = ZERO
                power = ONE
                for p in range(1, theta):
                    power = multiply_sigma(power)
                    value = add(value, scale(power, -1 if p % 2 == 0 else 1))
                self.memo[graph] = value
                return value

        # A connected loopless graph with cyclomatic number <= 1 is either a
        # cycle (removed by homeomorphic suppression) or contains a bridge.
        if edge_count <= graph.n:
            self.stats["low_cyclomatic_zeros"] += 1
            self.memo[graph] = ZERO
            return ZERO

        has_bridge, articulation = graph.bridge_and_articulation()
        if has_bridge:
            self.stats["bridge_zeros"] += 1
            self.memo[graph] = ZERO
            return ZERO

        if articulation is not None:
            parts = graph.articulation_parts_at(articulation)
            if parts is not None:
                self.stats["articulation_splits"] += 1
                value = ONE
                for part in parts:
                    value = multiply(value, self._rec(part))
                if (len(parts) - 1) % 2:
                    value = scale(value, -1)
                self.memo[graph] = value
                return value

        parallel = None
        for i in range(graph.n):
            for j in range(i + 1, graph.n):
                multiplicity = graph.rows[i][j]
                if multiplicity > 1:
                    parallel = (i, j, multiplicity)
                    break
            if parallel is not None:
                break

        if parallel is not None:
            i, j, multiplicity = parallel
            remainder = _delete_parallel_class(graph, i, j)
            contracted = _identify_vertices(remainder, i, j)
            value = add(
                self._rec(remainder),
                multiply(_parallel_factor(multiplicity), self._rec(contracted)),
            )
            self.stats["parallel_class_reductions"] += 1
            self.stats["parallel_edges_collapsed"] += multiplicity
            self.memo[graph] = value
            return value

        if edge is None:
            value = (((0, -1),) if graph.n % 2 else ONE)
        else:
            i, j = edge
            self.stats["ordinary_branches"] += 1
            value = add(
                self._rec(graph.delete_edge(i, j)),
                self._rec(graph.contract_edge(i, j)),
            )
        self.memo[graph] = value
        return value


def _med(fn, repeats: int = 3):
    timings = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), answer


def _fuzz_gate():
    rng = random.Random(20260818)
    checked = 0
    for _ in range(250):
        n = rng.randint(1, 8)
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(n))
        for _ in range(rng.randint(0, 16)):
            graph.add_edge(rng.randrange(n), rng.randrange(n))
        compact = CompactGraph.from_networkx(graph)
        expected = PythonCompactYamadaEvaluator().compute_laurent(compact)
        candidate = StructuralFlowEvaluator().compute_laurent(compact)
        if expected != candidate:
            raise AssertionError(
                "structural reduction changed exact Laurent output: "
                f"expected={expected}, candidate={candidate}, rows={compact.rows}"
            )
        checked += 1
    print(f"structural reduction fuzz equality gate: {checked} graphs PASS")


def _state_sum(calculator: Yamada, evaluator):
    total = ZERO
    for graph, exponent in calculator._iter_compact_states():
        total = add(total, shift(evaluator.compute_laurent(graph), exponent))
    return total


def _spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def _petersen_calculator():
    processor = PDCode(_spring_embedding(nx.petersen_graph(), 9))
    processor.compute(
        rotation_angles=(-134.58074129795634, 55.40942502382338, 0.0)
    )
    if len(processor.crossings) != 6:
        raise AssertionError(f"expected Petersen c=6, got {len(processor.crossings)}")
    return Yamada.from_PDCode(processor), len(processor.crossings)


def _random_cubic_calculator(vertex_count: int, sample_index: int = 0):
    sample, abstract = topology_ensemble(vertex_count, 2, DEFAULT_SEED)[sample_index]
    _, processor, _, _ = prepare_sample(sample, abstract, DEFAULT_SEED)
    return Yamada.from_PDCode(processor), len(processor.crossings)


def _benchmark_state_case(name: str, calculator: Yamada, crossings: int):
    baseline_evaluator = PythonCompactYamadaEvaluator()
    candidate_evaluator = StructuralFlowEvaluator()
    baseline_s, expected = _med(
        lambda: _state_sum(calculator, baseline_evaluator),
        1,
    )
    candidate_s, actual = _med(
        lambda: _state_sum(calculator, candidate_evaluator),
        1,
    )
    if expected != actual:
        raise AssertionError(f"state-sum mismatch for {name}")
    row = {
        "scope": "state_sum",
        "case": name,
        "crossings": crossings,
        "states": 3**crossings,
        "baseline_python_s": baseline_s,
        "structural_python_s": candidate_s,
        "speedup": baseline_s / candidate_s,
        "baseline_memo": len(baseline_evaluator.memo),
        "candidate_memo": len(candidate_evaluator.memo),
        "candidate_stats": candidate_evaluator.stats,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def _subdivided_graph_benchmark():
    graph = nx.MultiGraph(nx.complete_graph(4))
    next_node = max(graph.nodes) + 1
    for u, v in list(graph.edges()):
        graph.remove_edge(u, v)
        previous = u
        for _ in range(3):
            graph.add_edge(previous, next_node)
            previous = next_node
            next_node += 1
        graph.add_edge(previous, v)
    compact = CompactGraph.from_networkx(graph)
    baseline_eval = PythonCompactYamadaEvaluator()
    baseline_s, expected = _med(lambda: baseline_eval.compute_laurent(compact), 3)
    candidate_eval = StructuralFlowEvaluator()
    candidate_s, actual = _med(lambda: candidate_eval.compute_laurent(compact), 3)
    if expected != actual:
        raise AssertionError("subdivision reduction changed output")
    row = {
        "scope": "homeomorphism",
        "case": "K4_each_edge_subdivided_3x",
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "baseline_python_s": baseline_s,
        "structural_python_s": candidate_s,
        "speedup": baseline_s / candidate_s,
        "stats": candidate_eval.stats,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def main():
    _fuzz_gate()
    rows = [_subdivided_graph_benchmark()]
    calculator, crossings = _petersen_calculator()
    rows.append(_benchmark_state_case("petersen", calculator, crossings))
    calculator, crossings = _random_cubic_calculator(14, 0)
    rows.append(_benchmark_state_case("random_cubic_V14_s0", calculator, crossings))
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
