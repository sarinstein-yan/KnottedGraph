from __future__ import annotations

import json
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
    _theta_value,
)
from knotted_graph.invariants.yamada.fast import (
    ONE,
    SIGMA,
    ZERO,
    add,
    constant,
    multiply,
    multiply_sigma,
    scale,
    to_sympy,
)
from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    _evaluate_fast_state,
    _sum_laurent_states_raw,
)
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def median_time(fn, repeats=3):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def equal(left, right):
    return sp.simplify(sp.together(sp.expand(left - right))) == 0


def _edge_count_and_degrees(graph: CompactGraph):
    rows = graph.rows
    n = len(rows)
    edge_count = 0
    degrees = [0] * n
    first_loop = None
    first_edge = None
    for i in range(n):
        loop_count = rows[i][i]
        if loop_count:
            edge_count += loop_count
            degrees[i] += 2 * loop_count
            if first_loop is None:
                first_loop = i
        for j in range(i + 1, n):
            count = rows[i][j]
            if count:
                edge_count += count
                degrees[i] += count
                degrees[j] += count
                if first_edge is None:
                    first_edge = (i, j)
    return edge_count, tuple(degrees), first_loop, first_edge


def _components(graph: CompactGraph):
    rows = graph.rows
    n = len(rows)
    seen = bytearray(n)
    out = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        component = []
        while stack:
            u = stack.pop()
            component.append(u)
            row = rows[u]
            for v in range(n):
                if v != u and row[v] and not seen[v]:
                    seen[v] = 1
                    stack.append(v)
        out.append(tuple(sorted(component)))
    return out


def _bridge_and_articulation(graph: CompactGraph):
    rows = graph.rows
    n = len(rows)
    if n <= 1:
        return False, None
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    tick = 0
    bridge = False
    articulation = None

    def dfs(u):
        nonlocal tick, bridge, articulation
        disc[u] = low[u] = tick
        tick += 1
        child_count = 0
        for v, count in enumerate(rows[u]):
            if v == u or not count:
                continue
            if disc[v] == -1:
                parent[v] = u
                child_count += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u] and count == 1:
                    bridge = True
                if parent[u] == -1:
                    if child_count > 1 and articulation is None:
                        articulation = u
                elif low[v] >= disc[u] and articulation is None:
                    articulation = u
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for root in range(n):
        if disc[root] == -1:
            dfs(root)
    return bridge, articulation


def _split_at(graph: CompactGraph, cut: int):
    rows = graph.rows
    n = len(rows)
    remaining = [i for i in range(n) if i != cut]
    seen = set()
    components = []
    for start in remaining:
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in remaining:
                if v not in seen and rows[u][v]:
                    seen.add(v)
                    stack.append(v)
        components.append(tuple(sorted(comp)))
    if len(components) < 2:
        return None
    parts = []
    for part_index, component in enumerate(components):
        nodes = tuple(sorted((*component, cut)))
        matrix = [[rows[a][b] for b in nodes] for a in nodes]
        if part_index > 0:
            local_cut = nodes.index(cut)
            matrix[local_cut][local_cut] = 0
        parts.append(CompactGraph(tuple(tuple(row) for row in matrix)))
    return parts


def _choose_edge(graph: CompactGraph, degrees, strategy: str, first_edge):
    if strategy == "first":
        return first_edge
    best = None
    best_score = None
    n = graph.n
    for i in range(n):
        for j in range(i + 1, n):
            multiplicity = graph.rows[i][j]
            if not multiplicity:
                continue
            if strategy == "max_parallel":
                score = (multiplicity, degrees[i] + degrees[j], -i, -j)
            elif strategy == "max_degree_sum":
                score = (degrees[i] + degrees[j], multiplicity, -i, -j)
            else:
                raise ValueError(strategy)
            if best_score is None or score > best_score:
                best_score = score
                best = (i, j)
    return best


class CandidateEvaluator:
    def __init__(self, *, strategy="first", delete_first=True):
        self.strategy = strategy
        self.delete_first = delete_first
        self.memo = {}
        self.calls = 0

    def compute_laurent(self, graph):
        compact = graph if isinstance(graph, CompactGraph) else CompactGraph.from_networkx(graph)
        return self._rec(compact)

    def compute(self, graph):
        return to_sympy(self.compute_laurent(graph), A)

    def _rec(self, graph: CompactGraph):
        self.calls += 1
        cached = self.memo.get(graph)
        if cached is not None:
            return cached
        edge_count, degrees, loop, first_edge = _edge_count_and_degrees(graph)
        if edge_count == 0:
            value = constant((-1) ** graph.n)
            self.memo[graph] = value
            return value
        components = _components(graph)
        if len(components) > 1:
            value = ONE
            for component in components:
                value = multiply(value, self._rec(graph.induced(component)))
            self.memo[graph] = value
            return value
        if graph.n == 2 and not graph.rows[0][0] and not graph.rows[1][1]:
            theta = graph.rows[0][1]
            if theta == edge_count:
                value = _theta_value(theta)
                self.memo[graph] = value
                return value
        if graph.n and all(degree == 2 for degree in degrees):
            self.memo[graph] = SIGMA
            return SIGMA
        bridge, cut = _bridge_and_articulation(graph)
        if bridge:
            self.memo[graph] = ZERO
            return ZERO
        if loop is not None:
            value = multiply_sigma(self._rec(graph.delete_loop(loop)), sign=-1)
            self.memo[graph] = value
            return value
        if cut is not None:
            parts = _split_at(graph, cut)
            if parts is not None:
                value = ONE
                for part in parts:
                    value = multiply(value, self._rec(part))
                if (len(parts) - 1) % 2:
                    value = scale(value, -1)
                self.memo[graph] = value
                return value
        edge = _choose_edge(graph, degrees, self.strategy, first_edge)
        if edge is None:
            value = constant((-1) ** graph.n)
        else:
            i, j = edge
            if self.delete_first:
                first = self._rec(graph.delete_edge(i, j))
                second = self._rec(graph.contract_edge(i, j))
            else:
                first = self._rec(graph.contract_edge(i, j))
                second = self._rec(graph.delete_edge(i, j))
            value = add(first, second)
        self.memo[graph] = value
        return value


def kernel_cases():
    out = []
    for n in range(5, 8):
        out.append((f"wheel_{n}", nx.MultiGraph(nx.wheel_graph(n))))
    for n in range(3, 6):
        out.append((f"ladder_{n}", nx.MultiGraph(nx.circular_ladder_graph(n))))
    out.append(("K33", nx.MultiGraph(nx.complete_bipartite_graph(3, 3))))
    out.append(("K4", nx.MultiGraph(nx.complete_graph(4))))
    return out


def spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def connected_calculator():
    embedded = spring_embedding(nx.complete_graph(4), 7)
    processor = PDCode(embedded)
    # Find a deterministic nontrivial connected projection with at least two crossings.
    selected = None
    for ay in range(0, 180, 15):
        for ax in range(0, 180, 15):
            processor.compute(rotation_angles=(float(ax), float(ay), 0.0))
            crossings = len(processor.crossings)
            if 2 <= crossings <= 4:
                selected = (float(ax), float(ay), 0.0)
                break
        if selected is not None:
            break
    if selected is None:
        raise AssertionError("Could not find bounded connected K4 projection")
    processor.compute(rotation_angles=selected)
    calculator = Yamada.from_PDCode(processor)
    if len(calculator._diagram_blocks()) != 1:
        raise AssertionError("Connected benchmark unexpectedly factorized")
    return calculator, len(processor.crossings)


def candidate_state_sum(calculator, strategy, delete_first):
    evaluator = CandidateEvaluator(strategy=strategy, delete_first=delete_first)
    raw = _sum_laurent_states_raw(
        _evaluate_fast_state(evaluator, graph, exponent)
        for graph, exponent in calculator._iter_compact_states()
    )
    return to_sympy(raw, A), evaluator.calls, len(evaluator.memo)


def main():
    strategies = ["first", "max_parallel", "max_degree_sum"]
    rows = []
    print("KERNEL_CANDIDATES")
    for name, graph in kernel_cases():
        baseline_t, baseline = median_time(lambda: CompactYamadaEvaluator().compute(graph, A))
        for strategy in strategies:
            t, result = median_time(
                lambda strategy=strategy: CandidateEvaluator(strategy=strategy).compute(graph)
            )
            if not equal(baseline, result):
                raise AssertionError(f"Candidate output mismatch: {name} {strategy}")
            row = {
                "scope": "kernel",
                "case": name,
                "V": graph.number_of_nodes(),
                "E": graph.number_of_edges(),
                "strategy": strategy,
                "baseline_s": baseline_t,
                "candidate_s": t,
                "speedup": baseline_t / t,
            }
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")))

    ladder = nx.MultiGraph(nx.circular_ladder_graph(5))
    baseline_negami_t, baseline_negami = median_time(
        lambda: CompactNegamiSpecializedEvaluator().compute(ladder, A)
    )
    for delete_first in (False, True):
        t, result = median_time(
            lambda: CandidateEvaluator(strategy="first", delete_first=delete_first).compute(ladder)
        )
        if not equal(baseline_negami, result):
            raise AssertionError("Negami branch-order candidate changed output")
        row = {
            "scope": "negami_branch_order",
            "case": "ladder_5",
            "delete_first": delete_first,
            "baseline_s": baseline_negami_t,
            "candidate_s": t,
            "speedup": baseline_negami_t / t,
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))

    calculator, crossings = connected_calculator()
    baseline_t, baseline = median_time(
        lambda: calculator.compute(A, normalize=False, n_jobs=1, method="negami"), 2
    )
    print("CONNECTED_CANDIDATES")
    for strategy in strategies:
        t, result = median_time(
            lambda strategy=strategy: candidate_state_sum(calculator, strategy, True), 2
        )
        poly, calls, memo = result
        if not equal(baseline, poly):
            raise AssertionError(f"Connected candidate output mismatch: {strategy}")
        row = {
            "scope": "connected",
            "case": "K4",
            "V": 4,
            "E": 6,
            "crossings": crossings,
            "strategy": strategy,
            "baseline_s": baseline_t,
            "candidate_s": t,
            "speedup": baseline_t / t,
            "calls": calls,
            "memo": memo,
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
