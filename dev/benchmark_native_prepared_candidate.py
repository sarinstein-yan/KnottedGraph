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


BUILD_STATE_CPP = r'''
Graph build_prepared_state(
    int vertex_count,
    int crossing_count,
    const std::vector<int>& state,
    const std::vector<int>& arc_partner,
    const std::vector<int>& fixed_terminal_index,
    const std::vector<int>& crossing_for_port,
    const std::vector<int>& plus_partner,
    const std::vector<int>& minus_partner
) {
    const int port_count = static_cast<int>(arc_partner.size());
    if (
        static_cast<int>(state.size()) != crossing_count ||
        static_cast<int>(fixed_terminal_index.size()) != port_count ||
        static_cast<int>(crossing_for_port.size()) != port_count ||
        static_cast<int>(plus_partner.size()) != port_count ||
        static_cast<int>(minus_partner.size()) != port_count
    ) {
        throw std::invalid_argument("prepared Yamada table size mismatch");
    }

    std::vector<int> crossing_terminal_index(crossing_count, -1);
    int node_count = vertex_count;
    for (int crossing = 0; crossing < crossing_count; ++crossing) {
        int spin = state[crossing];
        if (spin == 2) {
            crossing_terminal_index[crossing] = node_count++;
        } else if (spin != 0 && spin != 1) {
            throw std::invalid_argument("invalid prepared Yamada state spin");
        }
    }

    auto terminal_index = [&](int port) -> int {
        int fixed = fixed_terminal_index[port];
        if (fixed >= 0) return fixed;
        int crossing = crossing_for_port[port];
        if (crossing >= 0 && state[crossing] == 2) {
            return crossing_terminal_index[crossing];
        }
        return -1;
    };

    std::vector<char> visited(port_count, 0);
    std::vector<std::pair<int, int>> graph_edges;
    graph_edges.reserve(port_count / 2);

    for (int start_port = 0; start_port < port_count; ++start_port) {
        int start_terminal = terminal_index(start_port);
        if (start_terminal < 0 || visited[start_port]) continue;

        int current = start_port;
        while (true) {
            int other = arc_partner[current];
            if (other < 0 || other >= port_count) {
                throw std::runtime_error("malformed prepared arc table");
            }
            visited[current] = 1;
            visited[other] = 1;

            int end_terminal = terminal_index(other);
            if (end_terminal >= 0) {
                graph_edges.emplace_back(start_terminal, end_terminal);
                break;
            }

            int crossing = crossing_for_port[other];
            if (crossing < 0 || crossing >= crossing_count) {
                throw std::runtime_error("resolved port has no crossing");
            }
            int spin = state[crossing];
            current = spin == 0 ? plus_partner[other] : minus_partner[other];
            if (current < 0 || current >= port_count) {
                throw std::runtime_error("malformed prepared resolution table");
            }
        }
    }

    int closed_loop_count = 0;
    for (int start_port = 0; start_port < port_count; ++start_port) {
        if (visited[start_port]) continue;
        ++closed_loop_count;
        int current = start_port;
        while (true) {
            int other = arc_partner[current];
            visited[current] = 1;
            visited[other] = 1;
            int crossing = crossing_for_port[other];
            if (crossing < 0 || crossing >= crossing_count || state[crossing] == 2) {
                throw std::runtime_error("malformed terminal-free prepared component");
            }
            current = state[crossing] == 0 ? plus_partner[other] : minus_partner[other];
            if (current < 0 || current >= port_count) {
                throw std::runtime_error("malformed prepared resolution table");
            }
            if (visited[current]) break;
        }
    }

    Graph graph;
    graph.n = node_count + closed_loop_count;
    graph.a.assign(
        static_cast<std::size_t>(graph.n) * static_cast<std::size_t>(graph.n), 0
    );
    for (const auto& [i, j] : graph_edges) {
        ++graph.at(i, j);
        if (i != j) ++graph.at(j, i);
    }
    for (int loop = 0; loop < closed_loop_count; ++loop) {
        int node = node_count + loop;
        graph.at(node, node) = 1;
    }
    return graph;
}

'''

COMPUTE_PREPARED_CPP = r'''
    PythonLaurent compute_prepared(
        int vertex_count,
        int crossing_count,
        const std::vector<int>& arc_partner,
        const std::vector<int>& fixed_terminal_index,
        const std::vector<int>& crossing_for_port,
        const std::vector<int>& plus_partner,
        const std::vector<int>& minus_partner
    ) {
        if (crossing_count < 0 || vertex_count < 0) {
            throw std::invalid_argument("negative prepared Yamada dimensions");
        }
        std::vector<int> state(static_cast<std::size_t>(crossing_count), 0);
        Laurent total;
        std::function<void(int, int)> enumerate = [&](int index, int exponent) {
            if (index == crossing_count) {
                Graph graph = build_prepared_state(
                    vertex_count,
                    crossing_count,
                    state,
                    arc_partner,
                    fixed_terminal_index,
                    crossing_for_port,
                    plus_partner,
                    minus_partner
                );
                total = add(total, shift(rec(graph), exponent));
                return;
            }
            state[index] = 0;
            enumerate(index + 1, exponent + 1);
            state[index] = 1;
            enumerate(index + 1, exponent - 1);
            state[index] = 2;
            enumerate(index + 1, exponent);
        };
        enumerate(0, 0);
        return to_python(total);
    }

'''

BINDING_CPP = r'''
        .def(
            "compute_prepared", &NativeEvaluator::compute_prepared,
            py::call_guard<py::gil_scoped_release>()
        )
'''


def _compile_candidate():
    source_path = Path("src/knotted_graph/invariants/yamada/_yamada_native.cpp")
    source = source_path.read_text()
    marker = "PythonLaurent to_python(const Laurent& poly) {"
    if marker not in source:
        raise RuntimeError("native source marker for prepared-state injection not found")
    source = source.replace(marker, BUILD_STATE_CPP + marker, 1)

    method_marker = "    std::size_t memo_size() const { return memo_.size(); }"
    if method_marker not in source:
        raise RuntimeError("native evaluator method marker not found")
    source = source.replace(method_marker, COMPUTE_PREPARED_CPP + method_marker, 1)

    binding_marker = '        .def("clear", &NativeEvaluator::clear)'
    if binding_marker not in source:
        raise RuntimeError("native evaluator binding marker not found")
    source = source.replace(binding_marker, BINDING_CPP + binding_marker, 1)
    source = source.replace(
        "PYBIND11_MODULE(_yamada_native, module)",
        "PYBIND11_MODULE(_yamada_prepared_candidate, module)",
        1,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="kg-yamada-prepared-"))
    candidate_cpp = tmpdir / "candidate.cpp"
    candidate_cpp.write_text(source)
    extension = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidate_so = tmpdir / f"_yamada_prepared_candidate{extension}"

    includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"],
        text=True,
    ).strip().split()
    compiler = os.environ.get("CXX", "c++")
    command = [
        compiler,
        "-O3",
        "-DNDEBUG",
        "-shared",
        "-std=c++17",
        "-fPIC",
        *includes,
        str(candidate_cpp),
        "-o",
        str(candidate_so),
    ]
    subprocess.run(command, check=True)

    spec = importlib.util.spec_from_file_location(
        "_yamada_prepared_candidate", candidate_so
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import temporary prepared candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepared(calculator: Yamada):
    prepared = PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )
    return prepared.reduce_reidemeister_ii()[0]


def _production(module, prepared):
    evaluator = module.NativeEvaluator()
    start = time.perf_counter()
    graphs = []
    exponents = []
    for config in itertools.product((0, 1, 2), repeat=len(prepared.crossing_ids)):
        graph = prepared.build(config)
        graphs.append([list(row) for row in graph.rows])
        exponents.append(config.count(0) - config.count(1))
    value = tuple(
        (int(power), int(coefficient))
        for power, coefficient in evaluator.compute_many(graphs, exponents)
    )
    elapsed = time.perf_counter() - start
    return value, elapsed, int(evaluator.memo_size)


def _candidate(module, prepared):
    evaluator = module.NativeEvaluator()
    start = time.perf_counter()
    value = tuple(
        (int(power), int(coefficient))
        for power, coefficient in evaluator.compute_prepared(
            len(prepared.vertex_ids),
            len(prepared.crossing_ids),
            list(prepared.arc_partner),
            list(prepared.fixed_terminal_index),
            list(prepared.crossing_for_port),
            list(prepared.plus_partner),
            list(prepared.minus_partner),
        )
    )
    elapsed = time.perf_counter() - start
    return value, elapsed, int(evaluator.memo_size)


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


def _benchmark(module, name: str, calculator: Yamada):
    prepared = _prepared(calculator)
    expected, production_s, production_memo = _production(module, prepared)
    actual, candidate_s, candidate_memo = _candidate(module, prepared)
    if expected != actual:
        raise AssertionError(
            f"native prepared candidate changed exact Laurent output for {name}: "
            f"production={expected}, candidate={actual}"
        )
    row = {
        "case": name,
        "crossings_after_rii": len(prepared.crossing_ids),
        "states": 3 ** len(prepared.crossing_ids),
        "production_python_state_build_plus_native_s": production_s,
        "native_prepared_s": candidate_s,
        "speedup": production_s / candidate_s,
        "production_memo": production_memo,
        "candidate_memo": candidate_memo,
        "correctness": "PASS",
    }
    print(json.dumps(row, separators=(",", ":")))
    return row


def main():
    module = _compile_candidate()
    rows = []
    rows.append(_benchmark(module, "petersen", _petersen()))
    rows.append(_benchmark(module, "random_cubic_V20_s0", _random_cubic(20, 0)))
    rows.append(_benchmark(module, "random_cubic_V20_s1", _random_cubic(20, 1)))
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
