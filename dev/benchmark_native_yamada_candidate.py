from __future__ import annotations

import json
import random
import statistics
import time

import networkx as nx
import numpy as np

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactYamadaEvaluator,
    PythonCompactNegamiSpecializedEvaluator,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import (
    NativeCompactEvaluator,
    native_available,
)
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode


def med(fn, repeats=5):
    samples = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), answer


def python_state_sum(states):
    evaluator = PythonCompactNegamiSpecializedEvaluator()
    total = ()
    for graph, exponent in states:
        total = add(total, shift(evaluator.compute_laurent(graph), exponent))
    return total, len(evaluator.memo)


def production_native_state_sum(states):
    evaluator = CompactYamadaEvaluator()
    if not isinstance(evaluator, NativeCompactEvaluator):
        raise AssertionError("production compact evaluator did not select native backend")
    total = evaluator.compute_many_laurent(states)
    return total, evaluator.memo_size


def spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def fuzz_gate():
    if not native_available():
        raise AssertionError("native extension is not active")
    rng = random.Random(20260818)
    checked = 0
    for _ in range(120):
        n = rng.randint(1, 8)
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(n))
        for _ in range(rng.randint(0, 15)):
            graph.add_edge(rng.randrange(n), rng.randrange(n))
        compact = CompactGraph.from_networkx(graph)
        expected = PythonCompactYamadaEvaluator().compute_laurent(compact)
        selected = CompactYamadaEvaluator()
        actual = selected.compute_laurent(compact)
        if expected != actual:
            raise AssertionError(
                f"production native mismatch on fuzz graph {checked}: "
                f"expected={expected}, actual={actual}, rows={compact.rows}"
            )
        checked += 1
    print(f"production native fuzz equality gate: {checked} graphs PASS")


def kernel_benchmarks():
    cases = [
        ("wheel8", nx.MultiGraph(nx.wheel_graph(8))),
        ("ladder5", nx.MultiGraph(nx.circular_ladder_graph(5))),
        ("K33", nx.MultiGraph(nx.complete_bipartite_graph(3, 3))),
        ("K4", nx.MultiGraph(nx.complete_graph(4))),
    ]
    out = []
    for name, graph in cases:
        compact = CompactGraph.from_networkx(graph)
        python_t, expected = med(
            lambda: PythonCompactYamadaEvaluator().compute_laurent(compact), 9
        )
        native_t, actual = med(
            lambda: CompactYamadaEvaluator().compute_laurent(compact), 9
        )
        if expected != actual:
            raise AssertionError(f"production native kernel mismatch: {name}")
        row = {
            "scope": "crossing_free_kernel",
            "case": name,
            "V": graph.number_of_nodes(),
            "E": graph.number_of_edges(),
            "python_s": python_t,
            "native_s": native_t,
            "speedup": python_t / native_t,
        }
        out.append(row)
        print(json.dumps(row, separators=(",", ":")))
    return out


def connected_petersen_benchmark():
    embedded = spring_embedding(nx.petersen_graph(), 9)
    processor = PDCode(embedded)
    rotation = (-134.58074129795634, 55.40942502382338, 0.0)
    processor.compute(rotation_angles=rotation)
    if len(processor.crossings) != 6:
        raise AssertionError(f"expected six crossings, got {len(processor.crossings)}")
    calculator = Yamada.from_PDCode(processor)
    if len(calculator._diagram_blocks()) != 1:
        raise AssertionError("Petersen native benchmark unexpectedly factorized")
    states = list(calculator._iter_compact_states())
    if len(states) != 3**6:
        raise AssertionError(f"expected 729 states, got {len(states)}")

    python_t, python_result = med(lambda: python_state_sum(states), 3)
    native_t, native_result = med(lambda: production_native_state_sum(states), 5)
    if python_result[0] != native_result[0]:
        raise AssertionError(
            "production native connected state sum changed exact Laurent output: "
            f"python={python_result[0]}, native={native_result[0]}"
        )
    row = {
        "scope": "connected_state_sum",
        "case": "petersen_c6",
        "V": 10,
        "E": 15,
        "crossings": 6,
        "states": len(states),
        "python_s": python_t,
        "native_s": native_t,
        "speedup": python_t / native_t,
        "python_memo": python_result[1],
        "native_memo": native_result[1],
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def overflow_fallback_gate():
    compact = CompactGraph(((0, 70), (70, 0)))
    expected = PythonCompactYamadaEvaluator().compute_laurent(compact)
    evaluator = CompactYamadaEvaluator()
    actual = evaluator.compute_laurent(compact)
    if actual != expected:
        raise AssertionError("native overflow fallback changed exact theta_70 output")
    if not isinstance(evaluator, NativeCompactEvaluator) or evaluator.fallback_calls != 1:
        raise AssertionError("theta_70 did not exercise native overflow fallback")
    print("native int64 overflow -> arbitrary-precision Python fallback: PASS")


def main():
    fuzz_gate()
    overflow_fallback_gate()
    results = kernel_benchmarks()
    results.append(connected_petersen_benchmark())
    print("SUMMARY=" + json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    main()
