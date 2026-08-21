from __future__ import annotations

from collections import defaultdict
import statistics
import time

import networkx as nx

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator


def _canonical(labels):
    remap = {}
    next_label = 0
    out = []
    for label in labels:
        if label not in remap:
            remap[label] = next_label
            next_label += 1
        out.append(remap[label])
    return tuple(out)


def _union(labels, left, right):
    a = labels[left]
    b = labels[right]
    if a == b:
        return labels, True
    low, high = (a, b) if a < b else (b, a)
    return _canonical(tuple(low if value == high else value for value in labels)), False


def _forget(labels, positions):
    remove = set(positions)
    return _canonical(tuple(value for index, value in enumerate(labels) if index not in remove))


def _q_poly_to_laurent(q_coefficients, vertex_count):
    total = ()
    sign = -1 if vertex_count % 2 else 1
    for beta, coefficient in sorted(q_coefficients.items()):
        term = ((0, sign * int(coefficient)),)
        for _ in range(beta):
            out = defaultdict(int)
            for power, coeff in term:
                out[power - 1] += coeff
                out[power] += 2 * coeff
                out[power + 1] += coeff
            term = tuple(sorted((p, c) for p, c in out.items() if c))
        merged = defaultdict(int)
        for power, coeff in total:
            merged[power] += coeff
        for power, coeff in term:
            merged[power] += coeff
        total = tuple(sorted((p, c) for p, c in merged.items() if c))
    return total


def frontier_flow_laurent(graph: nx.MultiGraph, vertex_order=None):
    """Exact H(G;A) using a path-frontier flow-polynomial state sum."""
    graph = nx.MultiGraph(graph)
    vertex_order = list(graph.nodes()) if vertex_order is None else list(vertex_order)
    if set(vertex_order) != set(graph.nodes()) or len(vertex_order) != graph.number_of_nodes():
        raise ValueError("vertex_order must contain every graph vertex exactly once")

    index = {vertex: i for i, vertex in enumerate(vertex_order)}
    backward = [[] for _ in vertex_order]
    last_use = list(range(len(vertex_order)))
    for u, v, _key in graph.edges(keys=True):
        iu = index[u]
        iv = index[v]
        if iu == iv:
            backward[iu].append((u, v))
            continue
        if iu > iv:
            iu, iv = iv, iu
            u, v = v, u
        backward[iv].append((u, v))
        last_use[iu] = max(last_use[iu], iv)
        last_use[iv] = max(last_use[iv], iv)

    active = []
    states = {((), 0): 1}
    max_frontier = 0
    max_states = 1

    for step, vertex in enumerate(vertex_order):
        active.append(vertex)
        introduced = {}
        for (labels, beta), coefficient in states.items():
            next_label = max(labels, default=-1) + 1
            introduced[(_canonical(labels + (next_label,)), beta)] = coefficient
        states = introduced

        positions = {node: i for i, node in enumerate(active)}
        for u, v in backward[step]:
            if u == v:
                updated = defaultdict(int)
                for (labels, beta), coefficient in states.items():
                    updated[(labels, beta)] += coefficient
                    updated[(labels, beta + 1)] -= coefficient
                states = {key: value for key, value in updated.items() if value}
                continue

            left = positions[u]
            right = positions[v]
            updated = defaultdict(int)
            for (labels, beta), coefficient in states.items():
                updated[(labels, beta)] += coefficient
                merged, closes_cycle = _union(labels, left, right)
                updated[(merged, beta + int(closes_cycle))] -= coefficient
            states = {key: value for key, value in updated.items() if value}

        forget_positions = [
            position
            for position, node in enumerate(active)
            if last_use[index[node]] <= step
        ]
        if forget_positions:
            forgotten = defaultdict(int)
            for (labels, beta), coefficient in states.items():
                forgotten[(_forget(labels, forget_positions), beta)] += coefficient
            states = {key: value for key, value in forgotten.items() if value}
            remove = set(forget_positions)
            active = [node for position, node in enumerate(active) if position not in remove]

        max_frontier = max(max_frontier, len(active))
        max_states = max(max_states, len(states))

    q_coefficients = defaultdict(int)
    for (labels, beta), coefficient in states.items():
        if labels:
            raise RuntimeError("frontier did not close")
        q_coefficients[beta] += coefficient
    q_coefficients = {beta: coefficient for beta, coefficient in q_coefficients.items() if coefficient}
    return _q_poly_to_laurent(q_coefficients, graph.number_of_nodes()), max_frontier, max_states


def ladder(rows: int, columns: int):
    base = nx.grid_2d_graph(rows, columns)
    graph = nx.MultiGraph(base)
    order = [(row, column) for column in range(columns) for row in range(rows)]
    return graph, order


def _median(fn, repeats=2):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), values, answer


def benchmark(rows, columns):
    graph, order = ladder(rows, columns)
    compact = CompactGraph.from_networkx(graph)
    native_time, native_times, native_value = _median(
        lambda: NativeCompactEvaluator(PythonCompactYamadaEvaluator).compute_laurent(compact)
    )
    frontier_time, frontier_times, frontier_result = _median(
        lambda: frontier_flow_laurent(graph, order)
    )
    frontier_value, width, states = frontier_result
    if frontier_value != native_value:
        raise AssertionError(
            f"frontier/native mismatch for {rows}x{columns}: "
            f"frontier={frontier_value}, native={native_value}"
        )
    print(
        f"grid={rows}x{columns} V={graph.number_of_nodes()} E={graph.number_of_edges()} "
        f"frontier={width} max_states={states} native_s={native_time:.9f} "
        f"frontier_s={frontier_time:.9f} native_over_frontier={native_time/frontier_time:.6f}"
    )
    print(f"  native_times={native_times}")
    print(f"  frontier_times={frontier_times}")


def main():
    for rows, columns in ((2, 10), (2, 30), (2, 100), (3, 10), (3, 20)):
        benchmark(rows, columns)


if __name__ == "__main__":
    main()
