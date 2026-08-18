from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUILD_STATE_CPP = r'''
// Build one resolved graph directly from the prepared PD port tables. This is
// an implementation optimization: the Yamada three-state definition is
// unchanged; only Python per-state allocation is removed.
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


def patch_cpp() -> None:
    path = ROOT / "src/knotted_graph/invariants/yamada/_yamada_native.cpp"
    text = path.read_text()
    if "compute_prepared(" not in text:
        marker = "PythonLaurent to_python(const Laurent& poly) {"
        assert text.count(marker) == 1
        text = text.replace(marker, BUILD_STATE_CPP + marker, 1)

        marker = "    std::size_t memo_size() const { return memo_.size(); }"
        assert text.count(marker) == 1
        text = text.replace(marker, COMPUTE_PREPARED_CPP + marker, 1)

        marker = '        .def("clear", &NativeEvaluator::clear)'
        assert text.count(marker) == 1
        binding = '''        .def(\n            "compute_prepared", &NativeEvaluator::compute_prepared,\n            py::call_guard<py::gil_scoped_release>()\n        )\n'''
        text = text.replace(marker, binding + marker, 1)

    provenance = '''// Exact structural identities used by the recurrence below are Yamada graph\n// polynomial identities, not numerical approximations. References:\n// S. Yamada, J. Graph Theory 13 (1989), 537-551.\n// https://doi.org/10.1002/jgt.3190130503\n// M. Li et al., J. Knot Theory Ramifications 27 (2018).\n// https://doi.org/10.1142/S021821651842004X\n// The dense compact representation, memo layout, int64 checked fast path and\n// prepared-state traversal are KnottedGraph implementation optimizations.\n\n'''
    anchor = "namespace py = pybind11;\n"
    if provenance not in text:
        assert text.count(anchor) == 1
        text = text.replace(anchor, provenance + anchor, 1)

    loop_anchor = "Laurent negative_sigma_power(int exponent) {"
    if "Yamada loop identity; Li et al." not in text:
        text = text.replace(
            loop_anchor,
            "// Yamada loop identity; Li et al., Sec. 2, property (3):\n"
            "// https://doi.org/10.1142/S021821651842004X\n" + loop_anchor,
            1,
        )
    parallel_anchor = "Laurent parallel_factor(int multiplicity) {"
    if "whole parallel class" not in text:
        text = text.replace(
            parallel_anchor,
            "// Closed whole-parallel-class factor obtained by repeated exact\n"
            "// deletion-contraction plus the loop identity (same reference).\n" + parallel_anchor,
            1,
        )
    homeo_anchor = "std::pair<Graph, Laurent> reduce_homeomorphic(const Graph& input) {"
    if "homeomorphism/subdivision" not in text:
        text = text.replace(
            homeo_anchor,
            "// Loop batching and degree-two homeomorphism/subdivision reduction;\n"
            "// see Li et al., Lemma 2.3 and Sec. 2 identities.\n" + homeo_anchor,
            1,
        )
    path.write_text(text)


def patch_native_wrapper() -> None:
    path = ROOT / "src/knotted_graph/invariants/yamada/native.py"
    text = path.read_text()
    if "def compute_prepared_laurent" not in text:
        marker = "    def compute(self, graph, variable):\n"
        assert text.count(marker) == 1
        method = '''    def compute_prepared_laurent(self, prepared):\n        """Evaluate a prepared diagram wholly in native code when available."""\n        if self._native is not None and hasattr(self._native, "compute_prepared"):\n            try:\n                self.native_calls += 1\n                return _as_laurent(\n                    self._native.compute_prepared(\n                        len(prepared.vertex_ids),\n                        len(prepared.crossing_ids),\n                        list(prepared.arc_partner),\n                        list(prepared.fixed_terminal_index),\n                        list(prepared.crossing_for_port),\n                        list(prepared.plus_partner),\n                        list(prepared.minus_partner),\n                    )\n                )\n            except OverflowError:\n                self.fallback_calls += 1\n\n        # Preserve exact arbitrary-precision behavior on non-native platforms or\n        # int64 overflow by evaluating the identical prepared state definition.\n        import itertools\n        from .fast import add, shift\n\n        evaluator = self._python()\n        total = ()\n        for config in itertools.product(\n            (0, 1, 2), repeat=len(prepared.crossing_ids)\n        ):\n            total = add(\n                total,\n                shift(\n                    evaluator.compute_laurent(prepared.build(config)),\n                    config.count(0) - config.count(1),\n                ),\n            )\n        return total\n\n'''
        text = text.replace(marker, method + marker, 1)
    path.write_text(text)


def patch_polynomial() -> None:
    path = ROOT / "src/knotted_graph/invariants/yamada/polynomial.py"
    text = path.read_text()
    old_iter = '''    def _iter_compact_states(self):\n        """Trace all resolutions directly into compact multigraphs."""\n        prepared = PreparedCompactStateBuilder.prepare(\n            self.vertices,\n            self.crossings,\n            self.arcs,\n            _ordered_crossing_ports,\n        )\n        prepared, _ = prepared.reduce_reidemeister_ii()\n        crossing_count = len(prepared.crossing_ids)\n        for config in itertools.product([0, 1, 2], repeat=crossing_count):\n            yield prepared.build(config), config.count(0) - config.count(1)\n'''
    if "def _prepare_compact_state_builder" not in text:
        assert text.count(old_iter) == 1
        new_iter = '''    def _prepare_compact_state_builder(self):\n        """Prepare and exactly RII-reduce the compact state tables once."""\n        prepared = PreparedCompactStateBuilder.prepare(\n            self.vertices,\n            self.crossings,\n            self.arcs,\n            _ordered_crossing_ports,\n        )\n        return prepared.reduce_reidemeister_ii()[0]\n\n    def _iter_compact_states(self):\n        """Trace all resolutions directly into compact multigraphs."""\n        prepared = self._prepare_compact_state_builder()\n        crossing_count = len(prepared.crossing_ids)\n        for config in itertools.product([0, 1, 2], repeat=crossing_count):\n            yield prepared.build(config), config.count(0) - config.count(1)\n'''
        text = text.replace(old_iter, new_iter, 1)

    old_block = '''    def _compute_laurent_block(self, evaluator):\n        states = self._iter_compact_states()\n        if hasattr(evaluator, "compute_many_laurent"):\n            return evaluator.compute_many_laurent(states)\n        evaluated_states = (\n            _evaluate_fast_state(evaluator, graph, exponent)\n            for graph, exponent in states\n        )\n        return _sum_laurent_states_raw(evaluated_states)\n'''
    if "compute_prepared_laurent(prepared)" not in text:
        assert text.count(old_block) == 1
        new_block = '''    def _compute_laurent_block(self, evaluator):\n        prepared = self._prepare_compact_state_builder()\n        if hasattr(evaluator, "compute_prepared_laurent"):\n            return evaluator.compute_prepared_laurent(prepared)\n\n        crossing_count = len(prepared.crossing_ids)\n        states = (\n            (prepared.build(config), config.count(0) - config.count(1))\n            for config in itertools.product([0, 1, 2], repeat=crossing_count)\n        )\n        if hasattr(evaluator, "compute_many_laurent"):\n            return evaluator.compute_many_laurent(states)\n        evaluated_states = (\n            _evaluate_fast_state(evaluator, graph, exponent)\n            for graph, exponent in states\n        )\n        return _sum_laurent_states_raw(evaluated_states)\n'''
        text = text.replace(old_block, new_block, 1)
    path.write_text(text)


def add_tests() -> None:
    path = ROOT / "tests/invariants/yamada/test_native_prepared_state_sum.py"
    if path.exists():
        return
    path.write_text(r'''from __future__ import annotations

import itertools

import sympy as sp

from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator, native_available
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.projection import PDCode
from knotted_graph.core import ThetaGraph


def _prepared_from_pd(processor):
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices, yamada.crossings, yamada.arcs, _ordered_crossing_ports
    )
    return prepared.reduce_reidemeister_ii()[0]


def _python_state_sum(prepared):
    evaluator = PythonCompactYamadaEvaluator()
    total = ()
    for config in itertools.product((0, 1, 2), repeat=len(prepared.crossing_ids)):
        total = add(
            total,
            shift(
                evaluator.compute_laurent(prepared.build(config)),
                config.count(0) - config.count(1),
            ),
        )
    return total


def test_native_prepared_path_matches_exact_python_state_sum():
    assert native_available()
    # The ordinary projection pipeline creates the prepared diagram; the test
    # deliberately uses a public graph constructor and no benchmark fixtures.
    processor = PDCode(ThetaGraph(3))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    prepared = _prepared_from_pd(processor)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    assert evaluator.compute_prepared_laurent(prepared) == _python_state_sum(prepared)


def test_public_result_is_identical_with_native_prepared_dispatch():
    A = sp.Symbol("A")
    processor = PDCode(ThetaGraph(3))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    yamada = Yamada.from_PDCode(processor)
    expected = yamada.compute(A, normalize=False, n_jobs=1, method="recursive")
    actual = yamada.compute(A, normalize=False, n_jobs=-1, method="negami")
    assert sp.expand(actual - expected) == 0
''')


def main() -> None:
    patch_cpp()
    patch_native_wrapper()
    patch_polynomial()
    add_tests()


if __name__ == "__main__":
    main()
