from __future__ import annotations

from collections import defaultdict
import itertools
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
    scale,
    shift,
)
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode

# Bounded-treewidth Tutte algorithms motivate the frontier-connectivity state:
# A. Andrzejak, Discrete Mathematics 190 (1998), 39-54.
# https://doi.org/10.1016/S0012-365X(98)00113-7
# Here we specialize the random-cluster representation to x=0, y=-sigma.

Partition = tuple[tuple[int, ...], ...]
QPoly = tuple[tuple[int, int], ...]


def _q_add(left: QPoly, right: QPoly) -> QPoly:
    out = dict(left)
    for degree, coefficient in right:
        value = out.get(degree, 0) + coefficient
        if value:
            out[degree] = value
        else:
            out.pop(degree, None)
    return tuple(sorted(out.items()))


def _q_shift(poly: QPoly, amount: int, coefficient: int = 1) -> QPoly:
    if not poly or coefficient == 0:
        return ()
    return tuple((degree + amount, coefficient * value) for degree, value in poly)


def _normalize_partition(blocks) -> Partition:
    return tuple(sorted((tuple(sorted(block)) for block in blocks if block), key=lambda b: b[0]))


def _introduce(partition: Partition, vertex: int) -> Partition:
    if any(vertex in block for block in partition):
        return partition
    return _normalize_partition((*partition, (vertex,)))


def _union(partition: Partition, left: int, right: int) -> Partition:
    if left == right:
        return partition
    left_block = right_block = None
    others = []
    for block in partition:
        if left in block:
            left_block = block
        elif right in block:
            right_block = block
        else:
            others.append(block)
    if left_block is None or right_block is None:
        raise RuntimeError("frontier edge endpoint was not introduced")
    if left_block == right_block:
        return partition
    return _normalize_partition((*others, tuple(set(left_block) | set(right_block))))


def _forget(partition: Partition, vertex: int) -> tuple[Partition, bool]:
    blocks = []
    closed = False
    found = False
    for block in partition:
        if vertex not in block:
            blocks.append(block)
            continue
        found = True
        remainder = tuple(node for node in block if node != vertex)
        if remainder:
            blocks.append(remainder)
        else:
            closed = True
    if not found:
        raise RuntimeError("forgotten frontier vertex was not active")
    return _normalize_partition(blocks), closed


def _min_fill_order(graph: CompactGraph) -> tuple[list[int], int]:
    adjacency = [
        {v for v in range(graph.n) if v != u and graph.rows[u][v]}
        for u in range(graph.n)
    ]
    remaining = set(range(graph.n))
    order = []
    width = 0
    while remaining:
        def score(vertex: int):
            neighbors = sorted(adjacency[vertex] & remaining)
            missing = sum(
                1
                for i, left in enumerate(neighbors)
                for right in neighbors[i + 1 :]
                if right not in adjacency[left]
            )
            return missing, len(neighbors), vertex

        vertex = min(remaining, key=score)
        neighbors = sorted(adjacency[vertex] & remaining)
        width = max(width, len(neighbors))
        for i, left in enumerate(neighbors):
            for right in neighbors[i + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining.remove(vertex)
        order.append(vertex)
    return order, width


def frontier_z_q_minus_q(graph: CompactGraph) -> tuple[QPoly, int, int]:
    """Return Z_G(q,-q) exactly by frontier connectivity DP."""
    order, elimination_width = _min_fill_order(graph)
    position = {vertex: index for index, vertex in enumerate(order)}
    dp: dict[Partition, QPoly] = {(): ((0, 1),)}
    peak_states = 1

    for vertex in order:
        introduced: dict[Partition, QPoly] = {}
        for partition, weight in dp.items():
            key = _introduce(partition, vertex)
            introduced[key] = _q_add(introduced.get(key, ()), weight)
        dp = introduced

        # A loop is an ordinary subset edge whose inclusion multiplies by -q
        # without changing connectivity.
        for _ in range(graph.rows[vertex][vertex]):
            updated: dict[Partition, QPoly] = {}
            for partition, weight in dp.items():
                combined = _q_add(weight, _q_shift(weight, 1, -1))
                updated[partition] = _q_add(updated.get(partition, ()), combined)
            dp = updated

        later_neighbors = [
            other
            for other in range(graph.n)
            if other != vertex
            and graph.rows[vertex][other]
            and position[other] > position[vertex]
        ]
        for other in later_neighbors:
            for _ in range(graph.rows[vertex][other]):
                updated: dict[Partition, QPoly] = {}
                for partition, weight in dp.items():
                    active = _introduce(partition, other)
                    # Edge excluded.
                    updated[active] = _q_add(updated.get(active, ()), weight)
                    # Edge included: factor -q and union the endpoint blocks.
                    merged = _union(active, vertex, other)
                    included = _q_shift(weight, 1, -1)
                    updated[merged] = _q_add(updated.get(merged, ()), included)
                dp = updated
                peak_states = max(peak_states, len(dp))

        forgotten: dict[Partition, QPoly] = {}
        for partition, weight in dp.items():
            key, closed = _forget(partition, vertex)
            if closed:
                weight = _q_shift(weight, 1)
            forgotten[key] = _q_add(forgotten.get(key, ()), weight)
        dp = forgotten
        peak_states = max(peak_states, len(dp))

    if set(dp) != {()}:
        raise RuntimeError(f"frontier did not close: {set(dp)}")
    return dp[()], elimination_width, peak_states


def _qpoly_to_yamada(poly: QPoly, vertex_count: int):
    # Z_G(q,-q)=(-1)^|V| q^|V| H(G), q=sigma+1=A^-1+2+A.
    shifted = tuple(
        (degree - vertex_count, coefficient * (-1 if vertex_count % 2 else 1))
        for degree, coefficient in poly
        if coefficient
    )
    if any(degree < 0 for degree, _ in shifted):
        raise ArithmeticError(f"Z(q,-q) was not divisible by q^V: {poly}")
    q_laurent = ((-1, 1), (0, 2), (1, 1))
    powers = [ONE]
    max_degree = max((degree for degree, _ in shifted), default=0)
    for _ in range(max_degree):
        powers.append(multiply(powers[-1], q_laurent))
    total = ZERO
    for degree, coefficient in shifted:
        total = add(total, scale(powers[degree], coefficient))
    return total


class FrontierFlowEvaluator:
    def __init__(self):
        self.memo = {}
        self.peak_states = 0
        self.max_width = 0

    def compute_laurent(self, graph: CompactGraph):
        cached = self.memo.get(graph)
        if cached is not None:
            return cached
        z, width, peak = frontier_z_q_minus_q(graph)
        value = _qpoly_to_yamada(z, graph.n)
        self.max_width = max(self.max_width, width)
        self.peak_states = max(self.peak_states, peak)
        self.memo[graph] = value
        return value


def _fuzz_gate():
    rng = random.Random(20260818)
    widths = []
    for index in range(180):
        n = rng.randint(1, 8)
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(n))
        for _ in range(rng.randint(0, 13)):
            graph.add_edge(rng.randrange(n), rng.randrange(n))
        compact = CompactGraph.from_networkx(graph)
        expected = PythonCompactYamadaEvaluator().compute_laurent(compact)
        candidate = FrontierFlowEvaluator()
        actual = candidate.compute_laurent(compact)
        if expected != actual:
            raise AssertionError(
                f"frontier flow mismatch index={index}: expected={expected}, actual={actual}, rows={compact.rows}"
            )
        widths.append(candidate.max_width)
    print(
        "frontier random-multigraph equality: 180 PASS; "
        f"max_elimination_width={max(widths)}"
    )


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
    return Yamada.from_PDCode(processor)


def _random_cubic(vertex_count: int, sample_index: int):
    sample, abstract = topology_ensemble(vertex_count, 2, DEFAULT_SEED)[sample_index]
    _, processor, _, _ = prepare_sample(sample, abstract, DEFAULT_SEED)
    return Yamada.from_PDCode(processor)


def _state_sum(calculator: Yamada, evaluator, limit: int | None = None):
    total = ZERO
    count = 0
    start = time.perf_counter()
    for graph, exponent in calculator._iter_compact_states():
        total = add(total, shift(evaluator.compute_laurent(graph), exponent))
        count += 1
        if limit is not None and count >= limit:
            break
    return total, time.perf_counter() - start, count


def _benchmark(name: str, calculator: Yamada, limit: int | None = None):
    baseline = PythonCompactYamadaEvaluator()
    expected, baseline_s, count = _state_sum(calculator, baseline, limit)
    candidate = FrontierFlowEvaluator()
    actual, candidate_s, candidate_count = _state_sum(calculator, candidate, limit)
    if count != candidate_count or expected != actual:
        raise AssertionError(f"frontier state-sum mismatch for {name}")
    row = {
        "case": name,
        "states_checked": count,
        "baseline_python_s": baseline_s,
        "frontier_python_s": candidate_s,
        "speedup": baseline_s / candidate_s,
        "frontier_max_width": candidate.max_width,
        "frontier_peak_partition_states": candidate.peak_states,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def main():
    _fuzz_gate()
    rows = []
    rows.append(_benchmark("petersen", _petersen()))
    # Keep the Python prototype bounded while still sampling the real hard
    # topologies. A C++ promotion is considered only if per-state work wins.
    rows.append(_benchmark("random_cubic_V20_s0_first81", _random_cubic(20, 0), 81))
    rows.append(_benchmark("random_cubic_V20_s1_first81", _random_cubic(20, 1), 81))
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
