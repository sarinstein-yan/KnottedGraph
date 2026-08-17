from __future__ import annotations

import json
import statistics
import time

import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada.compact import CompactGraph
from knotted_graph.invariants.yamada.fast import (
    ONE,
    ZERO,
    add as baseline_add,
    multiply as baseline_multiply,
    multiply_sigma as baseline_multiply_sigma,
    to_sympy,
)

A = sp.Symbol("A")


def candidate_add(left, right):
    if not left:
        return right
    if not right:
        return left
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        pe, pc = left[i]
        qe, qc = right[j]
        if pe < qe:
            out.append((pe, pc))
            i += 1
        elif qe < pe:
            out.append((qe, qc))
            j += 1
        else:
            value = pc + qc
            if value:
                out.append((pe, value))
            i += 1
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return tuple(out)


def candidate_multiply_sigma(poly, sign=1):
    if not poly:
        return ZERO
    minimum = poly[0][0] - 1
    maximum = poly[-1][0] + 1
    span = maximum - minimum + 1
    if span > 4 * len(poly) + 16:
        return baseline_multiply_sigma(poly, sign=sign)
    coeffs = [0] * span
    sign = int(sign)
    for exponent, coefficient in poly:
        value = sign * coefficient
        index = exponent - minimum
        coeffs[index - 1] += value
        coeffs[index] += value
        coeffs[index + 1] += value
    return tuple(
        (minimum + index, coefficient)
        for index, coefficient in enumerate(coeffs)
        if coefficient
    )


def candidate_multiply(left, right):
    if not left or not right:
        return ZERO
    if left == ONE:
        return right
    if right == ONE:
        return left
    left_span = left[-1][0] - left[0][0] + 1
    right_span = right[-1][0] - right[0][0] + 1
    if left_span == len(left) and right_span == len(right):
        coeffs = [0] * (left_span + right_span - 1)
        for i, (_, a) in enumerate(left):
            for j, (_, b) in enumerate(right):
                coeffs[i + j] += a * b
        minimum = left[0][0] + right[0][0]
        return tuple(
            (minimum + index, coefficient)
            for index, coefficient in enumerate(coeffs)
            if coefficient
        )
    return baseline_multiply(left, right)


def med(fn, repeats=2000):
    samples = []
    answer = None
    batch = max(1, repeats // 20)
    for _ in range(20):
        start = time.perf_counter()
        for _ in range(batch):
            answer = fn()
        samples.append((time.perf_counter() - start) / batch)
    return statistics.median(samples), answer


def polynomial_family(length):
    start = -(length // 2)
    return tuple((start + i, ((i * 17 + 5) % 23) - 11 or 1) for i in range(length))


def arithmetic_rows():
    rows = []
    for length in (5, 15, 40, 100):
        left = polynomial_family(length)
        right = tuple((power + 2, -coeff) for power, coeff in polynomial_family(length))
        for name, base_fn, candidate_fn, args in (
            ("add", baseline_add, candidate_add, (left, right)),
            ("multiply_sigma", baseline_multiply_sigma, candidate_multiply_sigma, (left,)),
            ("multiply", baseline_multiply, candidate_multiply, (left, right)),
        ):
            baseline_t, baseline = med(lambda: base_fn(*args), repeats=1200 if name == "multiply" else 4000)
            candidate_t, candidate = med(lambda: candidate_fn(*args), repeats=1200 if name == "multiply" else 4000)
            if baseline != candidate:
                raise AssertionError(f"Laurent arithmetic mismatch: {name} length={length}")
            row = {
                "operation": name,
                "terms": length,
                "baseline_s": baseline_t,
                "candidate_s": candidate_t,
                "speedup": baseline_t / candidate_t,
            }
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")))
    return rows


class CandidateEvaluator:
    def __init__(self):
        self.memo = {}

    def rec(self, graph):
        cached = self.memo.get(graph)
        if cached is not None:
            return cached
        edge_count, degrees, loop, edge = graph.scan()
        if edge_count == 0:
            value = () if False else ((0, (-1) ** graph.n),)
        else:
            components = graph.components()
            if len(components) > 1:
                value = ONE
                for component in components:
                    value = candidate_multiply(value, self.rec(graph.induced(component)))
            elif graph.n == 2 and not graph.rows[0][0] and not graph.rows[1][1] and graph.rows[0][1] == edge_count:
                value = ZERO
                power = ONE
                for p in range(1, edge_count):
                    power = candidate_multiply_sigma(power)
                    term = tuple((e, (-1 if p % 2 == 0 else 1) * c) for e, c in power)
                    value = candidate_add(value, term)
            elif graph.n and all(degree == 2 for degree in degrees):
                value = ((-1, 1), (0, 1), (1, 1))
            else:
                bridge, articulation = graph.bridge_and_articulation()
                if bridge:
                    value = ZERO
                elif loop is not None:
                    value = candidate_multiply_sigma(self.rec(graph.delete_loop(loop)), sign=-1)
                elif articulation is not None:
                    parts = graph.articulation_parts_at(articulation)
                    if parts is not None:
                        value = ONE
                        for part in parts:
                            value = candidate_multiply(value, self.rec(part))
                        if (len(parts) - 1) % 2:
                            value = tuple((e, -c) for e, c in value)
                    else:
                        value = self._branch(graph, edge)
                else:
                    value = self._branch(graph, edge)
        self.memo[graph] = value
        return value

    def _branch(self, graph, edge):
        if edge is None:
            return ((0, (-1) ** graph.n),)
        i, j = edge
        return candidate_add(self.rec(graph.delete_edge(i, j)), self.rec(graph.contract_edge(i, j)))


def real_kernel_rows():
    from knotted_graph.invariants.yamada.compact import CompactYamadaEvaluator

    cases = [
        ("wheel8", nx.MultiGraph(nx.wheel_graph(8))),
        ("ladder5", nx.MultiGraph(nx.circular_ladder_graph(5))),
        ("K33", nx.MultiGraph(nx.complete_bipartite_graph(3, 3))),
        ("K4", nx.MultiGraph(nx.complete_graph(4))),
    ]
    rows = []
    for name, graph in cases:
        compact = CompactGraph.from_networkx(graph)
        baseline_t, baseline = med(lambda: CompactYamadaEvaluator().compute_laurent(compact), repeats=300)
        candidate_t, candidate = med(lambda: CandidateEvaluator().rec(compact), repeats=300)
        if baseline != candidate:
            raise AssertionError(f"Real-kernel Laurent mismatch: {name}")
        if to_sympy(baseline, A) != to_sympy(candidate, A):
            raise AssertionError(f"Real-kernel SymPy mismatch: {name}")
        row = {
            "operation": "real_yamada_kernel",
            "case": name,
            "V": graph.number_of_nodes(),
            "E": graph.number_of_edges(),
            "baseline_s": baseline_t,
            "candidate_s": candidate_t,
            "speedup": baseline_t / candidate_t,
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))
    return rows


def main():
    rows = arithmetic_rows() + real_kernel_rows()
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
