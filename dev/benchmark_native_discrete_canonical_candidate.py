from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import time

import networkx as nx
import numpy as np

from benchmark_topoly_random_cubic_ensemble import (
    DEFAULT_SEED,
    prepare_sample,
    topology_ensemble,
)
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode


CANONICAL_CPP = r'''
Graph canonicalize_if_discrete(const Graph& graph) {
    if (graph.n <= 2) return graph;

    std::vector<int> colors(graph.n, 0);
    {
        std::vector<std::vector<int>> signatures(graph.n);
        for (int u = 0; u < graph.n; ++u) {
            int degree = 2 * graph.get(u, u);
            std::vector<int> multiplicities;
            for (int v = 0; v < graph.n; ++v) {
                if (v == u) continue;
                int m = graph.get(u, v);
                degree += m;
                if (m) multiplicities.push_back(m);
            }
            std::sort(multiplicities.begin(), multiplicities.end());
            signatures[u] = {graph.get(u, u), degree};
            signatures[u].insert(
                signatures[u].end(), multiplicities.begin(), multiplicities.end()
            );
        }
        auto unique = signatures;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (int u = 0; u < graph.n; ++u) {
            colors[u] = static_cast<int>(
                std::lower_bound(unique.begin(), unique.end(), signatures[u]) - unique.begin()
            );
        }
    }

    for (int iteration = 0; iteration < graph.n; ++iteration) {
        std::vector<std::vector<int>> signatures(graph.n);
        for (int u = 0; u < graph.n; ++u) {
            std::vector<std::pair<int, int>> neighbors;
            for (int v = 0; v < graph.n; ++v) {
                if (v == u) continue;
                int multiplicity = graph.get(u, v);
                if (multiplicity) neighbors.emplace_back(colors[v], multiplicity);
            }
            std::sort(neighbors.begin(), neighbors.end());
            auto& sig = signatures[u];
            sig.push_back(colors[u]);
            sig.push_back(graph.get(u, u));
            for (const auto& [color, multiplicity] : neighbors) {
                sig.push_back(color);
                sig.push_back(multiplicity);
            }
        }

        auto unique = signatures;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        std::vector<int> next(graph.n, 0);
        for (int u = 0; u < graph.n; ++u) {
            next[u] = static_cast<int>(
                std::lower_bound(unique.begin(), unique.end(), signatures[u]) - unique.begin()
            );
        }
        if (next == colors) break;
        colors.swap(next);
    }

    std::vector<int> order(graph.n);
    for (int i = 0; i < graph.n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (colors[a] != colors[b]) return colors[a] < colors[b];
        return a < b;
    });
    for (int i = 1; i < graph.n; ++i) {
        if (colors[order[i - 1]] == colors[order[i]]) {
            return graph;
        }
    }
    return induced(graph, order);
}

'''


def _compile(source: str, module_name: str):
    source = source.replace(
        "PYBIND11_MODULE(_yamada_native, module)",
        f"PYBIND11_MODULE({module_name}, module)",
        1,
    )
    # These are benchmark-only sibling extensions loaded into one interpreter.
    # Keep their NativeEvaluator Python registrations local to each module so
    # pybind11 does not reject the second temporary extension as a duplicate.
    binding = 'py::class_<NativeEvaluator>(module, "NativeEvaluator")'
    if binding not in source:
        raise RuntimeError("NativeEvaluator binding marker not found")
    source = source.replace(
        binding,
        'py::class_<NativeEvaluator>(module, "NativeEvaluator", py::module_local())',
        1,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix=f"kg-{module_name}-"))
    cpp = tmpdir / "candidate.cpp"
    cpp.write_text(source)
    extension = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so = tmpdir / f"{module_name}{extension}"
    includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"], text=True
    ).strip().split()
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-O3",
            "-DNDEBUG",
            "-shared",
            "-std=c++17",
            "-fPIC",
            *includes,
            str(cpp),
            "-o",
            str(so),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location(module_name, so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules():
    source = Path("src/knotted_graph/invariants/yamada/_yamada_native.cpp").read_text()
    baseline = _compile(source, "_yamada_canonical_baseline")

    marker = "PythonLaurent to_python(const Laurent& poly) {"
    assert source.count(marker) == 1
    candidate_source = source.replace(marker, CANONICAL_CPP + marker, 1)
    old = """    Laurent rec(const Graph& graph) {\n        auto found = memo_.find(graph);\n"""
    new = """    Laurent rec(const Graph& input_graph) {\n        Graph canonical_graph = canonicalize_if_discrete(input_graph);\n        const Graph& graph = canonical_graph;\n        auto found = memo_.find(graph);\n"""
    assert candidate_source.count(old) == 1
    candidate_source = candidate_source.replace(old, new, 1)
    candidate = _compile(candidate_source, "_yamada_discrete_canonical_candidate")
    return baseline, candidate


def _prepared(calculator: Yamada):
    prepared = PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )
    return prepared.reduce_reidemeister_ii()[0]


def _state_batch(prepared):
    graphs = []
    exponents = []
    for config in itertools.product((0, 1, 2), repeat=len(prepared.crossing_ids)):
        graph = prepared.build(config)
        graphs.append([list(row) for row in graph.rows])
        exponents.append(config.count(0) - config.count(1))
    return graphs, exponents


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


def _run(module, graphs, exponents):
    evaluator = module.NativeEvaluator()
    start = time.perf_counter()
    value = tuple(
        (int(power), int(coefficient))
        for power, coefficient in evaluator.compute_many(graphs, exponents)
    )
    return value, time.perf_counter() - start, int(evaluator.memo_size)


def _benchmark(baseline, candidate, name: str, calculator: Yamada):
    prepared = _prepared(calculator)
    graphs, exponents = _state_batch(prepared)
    expected, baseline_s, baseline_memo = _run(baseline, graphs, exponents)
    actual, candidate_s, candidate_memo = _run(candidate, graphs, exponents)
    if expected != actual:
        raise AssertionError(f"discrete canonicalization changed exact result for {name}")
    row = {
        "case": name,
        "crossings": len(prepared.crossing_ids),
        "states": len(graphs),
        "baseline_native_s": baseline_s,
        "discrete_canonical_s": candidate_s,
        "speedup": baseline_s / candidate_s,
        "baseline_memo": baseline_memo,
        "candidate_memo": candidate_memo,
        "memo_reduction_fraction": 1.0 - candidate_memo / baseline_memo,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def main():
    baseline, candidate = _modules()
    rows = [
        _benchmark(baseline, candidate, "petersen", _petersen()),
        _benchmark(baseline, candidate, "random_cubic_V20_s0", _random_cubic(20, 0)),
        _benchmark(baseline, candidate, "random_cubic_V20_s1", _random_cubic(20, 1)),
    ]
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
