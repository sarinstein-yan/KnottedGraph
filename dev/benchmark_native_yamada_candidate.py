from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactNegamiSpecializedEvaluator,
    CompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode

A = sp.Symbol("A")
ROOT = Path(__file__).resolve().parents[1]


def compile_candidate() -> object:
    build = Path(tempfile.mkdtemp(prefix="kg-native-yamada-"))
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python extension suffix unavailable")
    output = build / f"_kg_native_candidate{suffix}"
    includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"], text=True
    ).strip()
    compiler = os.environ.get("CXX", "c++")
    command = [
        compiler,
        "-O3",
        "-DNDEBUG",
        "-shared",
        "-std=c++17",
        "-fPIC",
        *shlex.split(includes),
        str(ROOT / "dev" / "native_yamada_candidate.cpp"),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    spec = importlib.util.spec_from_file_location("_kg_native_candidate", output)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load compiled native candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(graph: CompactGraph) -> list[list[int]]:
    return [list(row) for row in graph.rows]


def med(fn, repeats=5):
    samples = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), answer


def python_state_sum(states):
    evaluator = CompactNegamiSpecializedEvaluator()
    total = ()
    for graph, exponent in states:
        total = add(total, shift(evaluator.compute_laurent(graph), exponent))
    return total, len(evaluator.memo)


def native_state_sum(module, states):
    evaluator = module.NativeEvaluator()
    total = tuple(
        tuple(term)
        for term in evaluator.compute_many(
            [rows(graph) for graph, _ in states],
            [exponent for _, exponent in states],
        )
    )
    return total, evaluator.memo_size


def spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def fuzz_gate(module):
    rng = random.Random(20260818)
    checked = 0
    for _ in range(120):
        n = rng.randint(1, 8)
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(n))
        for _ in range(rng.randint(0, 15)):
            u = rng.randrange(n)
            v = rng.randrange(n)
            graph.add_edge(u, v)
        compact = CompactGraph.from_networkx(graph)
        expected = CompactYamadaEvaluator().compute_laurent(compact)
        native = module.NativeEvaluator()
        actual = tuple(tuple(term) for term in native.compute(rows(compact)))
        if expected != actual:
            raise AssertionError(
                f"native candidate mismatch on fuzz graph {checked}: "
                f"expected={expected}, actual={actual}, rows={compact.rows}"
            )
        checked += 1
    print(f"native fuzz equality gate: {checked} graphs PASS")


def kernel_benchmarks(module):
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
            lambda: CompactYamadaEvaluator().compute_laurent(compact), 9
        )
        native_t, actual = med(
            lambda: tuple(
                tuple(term) for term in module.NativeEvaluator().compute(rows(compact))
            ),
            9,
        )
        if expected != actual:
            raise AssertionError(f"native kernel mismatch: {name}")
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


def connected_petersen_benchmark(module):
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
    native_t, native_result = med(lambda: native_state_sum(module, states), 5)
    if python_result[0] != native_result[0]:
        raise AssertionError(
            "native connected state sum changed exact Laurent output: "
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


def main():
    module = compile_candidate()
    fuzz_gate(module)
    results = kernel_benchmarks(module)
    results.append(connected_petersen_benchmark(module))
    print("SUMMARY=" + json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    main()
