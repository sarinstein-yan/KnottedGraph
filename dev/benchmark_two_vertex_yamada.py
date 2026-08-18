from __future__ import annotations

import json
import random
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
    shift,
)
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode

SIGMA = multiply_sigma(ONE)
SIGMA_PLUS_ONE = add(SIGMA, ONE)


def _divide_sigma_exact(poly):
    """Divide an exact sparse Laurent tuple by sigma=A^-1+1+A."""
    if not poly:
        return ZERO
    source = dict(poly)
    lo = min(source)
    hi = max(source)
    quotient: dict[int, int] = {}
    for exponent in range(lo, hi + 1):
        # [A^e](sigma*q) = q[e+1] + q[e] + q[e-1].
        value = source.get(exponent, 0)
        value -= quotient.get(exponent, 0)
        value -= quotient.get(exponent - 1, 0)
        if value:
            quotient[exponent + 1] = value
    out = tuple(sorted((power, coeff) for power, coeff in quotient.items() if coeff))
    if multiply_sigma(out) != poly:
        raise ArithmeticError(f"Laurent polynomial is not exactly divisible by sigma: {poly}")
    return out


def _components_without_pair(graph: CompactGraph, left: int, right: int):
    removed = {left, right}
    seen = bytearray(graph.n)
    seen[left] = 1
    seen[right] = 1
    components: list[tuple[int, ...]] = []
    for start in range(graph.n):
        if start in removed or seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for other, multiplicity in enumerate(graph.rows[node]):
                if other in removed or other == node or not multiplicity or seen[other]:
                    continue
                seen[other] = 1
                stack.append(other)
        components.append(tuple(sorted(component)))
    return components


def _find_clean_two_vertex_split(graph: CompactGraph):
    """Find a two-vertex cut with no separator edge, yielding an unambiguous 2-sum."""
    if graph.n < 4:
        return None
    best = None
    for left in range(graph.n):
        for right in range(left + 1, graph.n):
            if graph.rows[left][right]:
                continue
            components = _components_without_pair(graph, left, right)
            if len(components) < 2:
                continue
            components.sort(key=len)
            first = components[0]
            rest = tuple(node for comp in components[1:] for node in comp)
            # Prefer a balanced cut, which most strongly reduces recurrence size.
            score = max(len(first), len(rest))
            candidate = (score, left, right, first, rest)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return None
    _, left, right, first, rest = best
    return left, right, first, rest


def _two_vertex_parts(graph: CompactGraph, split):
    left, right, first, rest = split
    nodes1 = (left, right, *first)
    nodes2 = (left, right, *rest)
    graph1 = graph.induced(nodes1)
    graph2 = graph.induced(nodes2)
    # The separator vertices occupy local indices 0 and 1 by construction.
    identified1 = graph1.identify_vertices(0, 1)
    identified2 = graph2.identify_vertices(0, 1)
    return graph1, graph2, identified1, identified2


class TwoVertexEvaluator:
    """Exact Yamada two-vertex decomposition wrapped around production reductions."""

    def __init__(self):
        self.memo = {}
        self.base = PythonCompactYamadaEvaluator()
        self.splits = 0
        self.max_piece_n = 0

    def compute_laurent(self, graph: CompactGraph):
        cached = self.memo.get(graph)
        if cached is not None:
            return cached

        split = _find_clean_two_vertex_split(graph)
        if split is None:
            value = self.base.compute_laurent(graph)
            self.memo[graph] = value
            return value

        graph1, graph2, identified1, identified2 = _two_vertex_parts(graph, split)
        self.splits += 1
        self.max_piece_n = max(self.max_piece_n, graph1.n, graph2.n)
        h1 = self.compute_laurent(graph1)
        h2 = self.compute_laurent(graph2)
        k1 = self.compute_laurent(identified1)
        k2 = self.compute_laurent(identified2)

        numerator = add(
            add(multiply(k1, k2), multiply(SIGMA_PLUS_ONE, multiply(h1, h2))),
            add(multiply(k1, h2), multiply(k2, h1)),
        )
        value = _divide_sigma_exact(numerator)
        self.memo[graph] = value
        return value


def _random_multigraph(seed: int) -> CompactGraph:
    rng = random.Random(seed)
    n = rng.randint(2, 9)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n))
    for _ in range(rng.randint(0, 18)):
        graph.add_edge(rng.randrange(n), rng.randrange(n))
    return CompactGraph.from_networkx(graph)


def _fuzz_gate():
    splits = 0
    for seed in range(160):
        graph = _random_multigraph(31000 + seed)
        expected = PythonCompactYamadaEvaluator().compute_laurent(graph)
        evaluator = TwoVertexEvaluator()
        actual = evaluator.compute_laurent(graph)
        if expected != actual:
            raise AssertionError(
                f"two-vertex formula mismatch seed={seed}: expected={expected}, actual={actual}, rows={graph.rows}"
            )
        splits += evaluator.splits
    print(f"two-vertex exact fuzz gate: 160 graphs PASS; decompositions={splits}")


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


def _petersen():
    processor = PDCode(_spring_embedding(nx.petersen_graph(), 9))
    processor.compute(rotation_angles=(-134.58074129795634, 55.40942502382338, 0.0))
    return Yamada.from_PDCode(processor), len(processor.crossings)


def _random_cubic(vertex_count: int, sample_index: int):
    sample, abstract = topology_ensemble(vertex_count, 2, DEFAULT_SEED)[sample_index]
    _, processor, _, _ = prepare_sample(sample, abstract, DEFAULT_SEED)
    return Yamada.from_PDCode(processor), len(processor.crossings)


def _benchmark(name: str, calculator: Yamada, crossings: int):
    baseline = PythonCompactYamadaEvaluator()
    start = time.perf_counter()
    expected = _state_sum(calculator, baseline)
    baseline_s = time.perf_counter() - start

    candidate = TwoVertexEvaluator()
    start = time.perf_counter()
    actual = _state_sum(calculator, candidate)
    candidate_s = time.perf_counter() - start
    if expected != actual:
        raise AssertionError(f"two-vertex state-sum mismatch for {name}")

    row = {
        "case": name,
        "crossings": crossings,
        "states": 3**crossings,
        "baseline_structural_python_s": baseline_s,
        "two_vertex_python_s": candidate_s,
        "speedup": baseline_s / candidate_s,
        "baseline_memo": len(baseline.memo),
        "two_vertex_wrapper_memo": len(candidate.memo),
        "two_vertex_base_memo": len(candidate.base.memo),
        "two_vertex_splits": candidate.splits,
        "max_piece_n": candidate.max_piece_n,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def main():
    _fuzz_gate()
    rows = []
    calculator, crossings = _petersen()
    rows.append(_benchmark("petersen", calculator, crossings))
    calculator, crossings = _random_cubic(14, 0)
    rows.append(_benchmark("random_cubic_V14_s0", calculator, crossings))
    calculator, crossings = _random_cubic(20, 0)
    rows.append(_benchmark("random_cubic_V20_s0", calculator, crossings))
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
