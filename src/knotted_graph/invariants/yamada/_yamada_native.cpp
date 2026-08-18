#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using Coeff = std::int64_t;
using Laurent = std::map<int, Coeff>;
using PythonLaurent = std::vector<std::pair<int, Coeff>>;

namespace {

[[noreturn]] void coefficient_overflow() {
    throw std::overflow_error(
        "native Yamada coefficient exceeded int64; use exact Python fallback"
    );
}

Coeff checked_add(Coeff a, Coeff b) {
    constexpr Coeff hi = std::numeric_limits<Coeff>::max();
    constexpr Coeff lo = std::numeric_limits<Coeff>::min();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) {
        coefficient_overflow();
    }
    return static_cast<Coeff>(a + b);
}

Coeff checked_negate(Coeff value) {
    if (value == std::numeric_limits<Coeff>::min()) {
        coefficient_overflow();
    }
    return static_cast<Coeff>(-value);
}

Coeff checked_mul(Coeff a, Coeff b) {
    constexpr Coeff hi = std::numeric_limits<Coeff>::max();
    constexpr Coeff lo = std::numeric_limits<Coeff>::min();
    if (a == 0 || b == 0) return 0;
    if (a == -1) return checked_negate(b);
    if (b == -1) return checked_negate(a);

    if (a > 0) {
        if (b > 0) {
            if (a > hi / b) coefficient_overflow();
        } else {
            if (b < lo / a) coefficient_overflow();
        }
    } else {
        if (b > 0) {
            if (a < lo / b) coefficient_overflow();
        } else {
            if (a < hi / b) coefficient_overflow();
        }
    }
    return static_cast<Coeff>(a * b);
}

void accumulate(Laurent& out, int power, Coeff value) {
    if (value == 0) return;
    auto it = out.find(power);
    if (it == out.end()) {
        out.emplace(power, value);
        return;
    }
    Coeff combined = checked_add(it->second, value);
    if (combined == 0) out.erase(it);
    else it->second = combined;
}

struct Graph {
    int n{};
    std::vector<int> a;

    int get(int i, int j) const {
        return a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + j];
    }
    int& at(int i, int j) {
        return a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + j];
    }
    bool operator==(const Graph& other) const {
        return n == other.n && a == other.a;
    }
};

struct GraphHash {
    std::size_t operator()(const Graph& graph) const noexcept {
        std::size_t h = static_cast<std::size_t>(graph.n) + 0x9e3779b97f4a7c15ULL;
        for (int value : graph.a) {
            h ^= static_cast<std::size_t>(value + 0x9e3779b9) + (h << 6) + (h >> 2);
        }
        return h;
    }
};

Graph from_rows(const std::vector<std::vector<int>>& rows) {
    const int n = static_cast<int>(rows.size());
    Graph graph;
    graph.n = n;
    graph.a.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (const auto& row : rows) {
        if (static_cast<int>(row.size()) != n) {
            throw std::invalid_argument("compact graph matrix must be square");
        }
        for (int value : row) {
            if (value < 0) {
                throw std::invalid_argument("compact graph multiplicities must be non-negative");
            }
            graph.a.push_back(value);
        }
    }
    return graph;
}

Laurent constant(int value) {
    Laurent out;
    if (value) out.emplace(0, static_cast<Coeff>(value));
    return out;
}

Laurent add(const Laurent& left, const Laurent& right) {
    Laurent out = left;
    for (const auto& [power, coefficient] : right) {
        accumulate(out, power, coefficient);
    }
    return out;
}

Laurent scale(const Laurent& poly, int coefficient) {
    if (coefficient == 0 || poly.empty()) return {};
    if (coefficient == 1) return poly;
    Laurent out;
    for (const auto& [power, value] : poly) {
        out.emplace(power, checked_mul(value, static_cast<Coeff>(coefficient)));
    }
    return out;
}

Laurent shift(const Laurent& poly, int exponent) {
    if (poly.empty() || exponent == 0) return poly;
    Laurent out;
    for (const auto& [power, value] : poly) {
        out.emplace(power + exponent, value);
    }
    return out;
}

Laurent multiply(const Laurent& left, const Laurent& right) {
    if (left.empty() || right.empty()) return {};
    Laurent out;
    for (const auto& [p, a] : left) {
        for (const auto& [q, b] : right) {
            accumulate(out, p + q, checked_mul(a, b));
        }
    }
    return out;
}

Laurent multiply_sigma(const Laurent& poly, int sign = 1) {
    if (poly.empty()) return {};
    Laurent out;
    for (const auto& [power, coefficient] : poly) {
        Coeff value = sign == 1 ? coefficient : checked_negate(coefficient);
        accumulate(out, power - 1, value);
        accumulate(out, power, value);
        accumulate(out, power + 1, value);
    }
    return out;
}

Laurent negative_sigma_power(int exponent) {
    Laurent out = constant(1);
    for (int i = 0; i < exponent; ++i) {
        out = multiply_sigma(out, -1);
    }
    return out;
}

Laurent parallel_factor(int multiplicity) {
    Laurent total;
    Laurent power = constant(1);
    for (int i = 0; i < multiplicity; ++i) {
        total = add(total, power);
        power = multiply_sigma(power, -1);
    }
    return total;
}

Laurent theta_value(int theta) {
    Laurent value;
    Laurent power = constant(1);
    for (int p = 1; p < theta; ++p) {
        power = multiply_sigma(power);
        value = add(value, scale(power, p % 2 == 0 ? -1 : 1));
    }
    return value;
}

struct Scan {
    int edge_count{};
    std::vector<int> degrees;
    int loop{-1};
    int edge_i{-1};
    int edge_j{-1};
};

Scan scan(const Graph& graph) {
    Scan result;
    result.degrees.assign(graph.n, 0);
    for (int i = 0; i < graph.n; ++i) {
        int loops = graph.get(i, i);
        if (loops) {
            result.edge_count += loops;
            result.degrees[i] += 2 * loops;
            if (result.loop < 0) result.loop = i;
        }
        for (int j = i + 1; j < graph.n; ++j) {
            int count = graph.get(i, j);
            if (!count) continue;
            result.edge_count += count;
            result.degrees[i] += count;
            result.degrees[j] += count;
            if (result.edge_i < 0) {
                result.edge_i = i;
                result.edge_j = j;
            }
        }
    }
    return result;
}

std::vector<std::vector<int>> components(const Graph& graph) {
    std::vector<char> seen(graph.n, 0);
    std::vector<std::vector<int>> result;
    for (int start = 0; start < graph.n; ++start) {
        if (seen[start]) continue;
        seen[start] = 1;
        std::vector<int> stack{start};
        std::vector<int> component;
        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();
            component.push_back(u);
            for (int v = 0; v < graph.n; ++v) {
                if (v != u && graph.get(u, v) && !seen[v]) {
                    seen[v] = 1;
                    stack.push_back(v);
                }
            }
        }
        std::sort(component.begin(), component.end());
        result.push_back(std::move(component));
    }
    return result;
}

Graph induced(const Graph& graph, const std::vector<int>& nodes) {
    Graph out;
    out.n = static_cast<int>(nodes.size());
    out.a.assign(
        static_cast<std::size_t>(out.n) * static_cast<std::size_t>(out.n), 0
    );
    for (int i = 0; i < out.n; ++i) {
        for (int j = 0; j < out.n; ++j) {
            out.at(i, j) = graph.get(nodes[i], nodes[j]);
        }
    }
    return out;
}

struct TarjanResult {
    bool bridge{false};
    int articulation{-1};
};

TarjanResult bridge_and_articulation(const Graph& graph) {
    TarjanResult result;
    if (graph.n <= 1) return result;
    std::vector<int> disc(graph.n, -1), low(graph.n, 0), parent(graph.n, -1);
    int tick = 0;
    std::function<void(int)> dfs = [&](int u) {
        disc[u] = low[u] = tick++;
        int children = 0;
        for (int v = 0; v < graph.n; ++v) {
            int count = graph.get(u, v);
            if (v == u || !count) continue;
            if (disc[v] == -1) {
                parent[v] = u;
                ++children;
                dfs(v);
                low[u] = std::min(low[u], low[v]);
                if (low[v] > disc[u] && count == 1) result.bridge = true;
                if (result.articulation < 0) {
                    if (parent[u] == -1 && children > 1) {
                        result.articulation = u;
                    } else if (parent[u] != -1 && low[v] >= disc[u]) {
                        result.articulation = u;
                    }
                }
            } else if (v != parent[u]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
    };
    for (int root = 0; root < graph.n; ++root) {
        if (disc[root] == -1) dfs(root);
    }
    return result;
}

Graph suppress_degree_two(const Graph& graph, int vertex) {
    std::vector<int> neighbors;
    for (int other = 0; other < graph.n; ++other) {
        if (other == vertex) continue;
        for (int count = 0; count < graph.get(vertex, other); ++count) {
            neighbors.push_back(other);
        }
    }
    if (graph.get(vertex, vertex) != 0 || neighbors.size() != 2) {
        throw std::invalid_argument("degree-two suppression precondition failed");
    }

    const int left = neighbors[0];
    const int right = neighbors[1];
    std::vector<int> kept;
    kept.reserve(graph.n - 1);
    for (int i = 0; i < graph.n; ++i) {
        if (i != vertex) kept.push_back(i);
    }
    Graph out = induced(graph, kept);
    auto local_index = [&](int old) {
        auto it = std::find(kept.begin(), kept.end(), old);
        return static_cast<int>(std::distance(kept.begin(), it));
    };
    int i = local_index(left);
    int j = local_index(right);
    if (i == j) {
        ++out.at(i, i);
    } else {
        ++out.at(i, j);
        ++out.at(j, i);
    }
    return out;
}

std::pair<Graph, Laurent> reduce_homeomorphic(const Graph& input) {
    Graph graph = input;
    Laurent factor = constant(1);
    while (true) {
        int loop_count = 0;
        for (int i = 0; i < graph.n; ++i) {
            loop_count += graph.get(i, i);
            graph.at(i, i) = 0;
        }
        if (loop_count) {
            factor = multiply(factor, negative_sigma_power(loop_count));
        }

        Scan info = scan(graph);
        int degree_two = -1;
        for (int i = 0; i < graph.n; ++i) {
            if (info.degrees[i] == 2 && graph.get(i, i) == 0) {
                degree_two = i;
                break;
            }
        }
        if (degree_two < 0) return {std::move(graph), std::move(factor)};
        graph = suppress_degree_two(graph, degree_two);
    }
}

Graph delete_edge(const Graph& graph, int i, int j) {
    Graph out = graph;
    --out.at(i, j);
    --out.at(j, i);
    return out;
}

Graph delete_parallel_class(const Graph& graph, int i, int j) {
    Graph out = graph;
    out.at(i, j) = 0;
    out.at(j, i) = 0;
    return out;
}

Graph identify_vertices(const Graph& graph, int i, int j) {
    if (i == j) return graph;
    if (i > j) std::swap(i, j);
    Graph matrix = graph;
    matrix.at(i, i) += matrix.get(j, j) + matrix.get(i, j);
    for (int k = 0; k < graph.n; ++k) {
        if (k == i || k == j) continue;
        matrix.at(i, k) += matrix.get(j, k);
        matrix.at(k, i) = matrix.get(i, k);
    }

    Graph out;
    out.n = graph.n - 1;
    out.a.assign(
        static_cast<std::size_t>(out.n) * static_cast<std::size_t>(out.n), 0
    );
    int oi = 0;
    for (int r = 0; r < graph.n; ++r) {
        if (r == j) continue;
        int oj = 0;
        for (int c = 0; c < graph.n; ++c) {
            if (c == j) continue;
            out.at(oi, oj) = matrix.get(r, c);
            ++oj;
        }
        ++oi;
    }
    return out;
}

Graph contract_edge(const Graph& graph, int i, int j) {
    if (i == j) throw std::invalid_argument("cannot contract loop");
    if (i > j) std::swap(i, j);
    Graph matrix = graph;
    --matrix.at(i, j);
    --matrix.at(j, i);
    matrix.at(i, i) += matrix.get(j, j) + matrix.get(i, j);
    for (int k = 0; k < graph.n; ++k) {
        if (k == i || k == j) continue;
        matrix.at(i, k) += matrix.get(j, k);
        matrix.at(k, i) = matrix.get(i, k);
    }

    Graph out;
    out.n = graph.n - 1;
    out.a.assign(
        static_cast<std::size_t>(out.n) * static_cast<std::size_t>(out.n), 0
    );
    int oi = 0;
    for (int r = 0; r < graph.n; ++r) {
        if (r == j) continue;
        int oj = 0;
        for (int c = 0; c < graph.n; ++c) {
            if (c == j) continue;
            out.at(oi, oj) = matrix.get(r, c);
            ++oj;
        }
        ++oi;
    }
    return out;
}

std::vector<Graph> articulation_parts(const Graph& graph, int cut) {
    std::vector<int> remaining;
    for (int i = 0; i < graph.n; ++i) {
        if (i != cut) remaining.push_back(i);
    }
    std::vector<char> seen(graph.n, 0);
    seen[cut] = 1;
    std::vector<std::vector<int>> comps;
    for (int start : remaining) {
        if (seen[start]) continue;
        seen[start] = 1;
        std::vector<int> stack{start}, component;
        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();
            component.push_back(u);
            for (int v : remaining) {
                if (!seen[v] && graph.get(u, v)) {
                    seen[v] = 1;
                    stack.push_back(v);
                }
            }
        }
        std::sort(component.begin(), component.end());
        comps.push_back(std::move(component));
    }
    if (comps.size() < 2) return {};

    std::vector<Graph> parts;
    for (std::size_t index = 0; index < comps.size(); ++index) {
        auto nodes = comps[index];
        nodes.push_back(cut);
        std::sort(nodes.begin(), nodes.end());
        Graph part = induced(graph, nodes);
        if (index > 0) {
            auto it = std::find(nodes.begin(), nodes.end(), cut);
            int local_cut = static_cast<int>(std::distance(nodes.begin(), it));
            part.at(local_cut, local_cut) = 0;
        }
        parts.push_back(std::move(part));
    }
    return parts;
}

PythonLaurent to_python(const Laurent& poly) {
    PythonLaurent out;
    out.reserve(poly.size());
    for (const auto& [power, coefficient] : poly) {
        out.emplace_back(power, coefficient);
    }
    return out;
}

}  // namespace

class NativeEvaluator {
public:
    PythonLaurent compute(const std::vector<std::vector<int>>& rows) {
        return to_python(rec(from_rows(rows)));
    }

    PythonLaurent compute_many(
        const std::vector<std::vector<std::vector<int>>>& graphs,
        const std::vector<int>& exponents
    ) {
        if (graphs.size() != exponents.size()) {
            throw std::invalid_argument("graphs/exponents size mismatch");
        }
        Laurent total;
        for (std::size_t i = 0; i < graphs.size(); ++i) {
            total = add(total, shift(rec(from_rows(graphs[i])), exponents[i]));
        }
        return to_python(total);
    }

    std::size_t memo_size() const { return memo_.size(); }
    void clear() { memo_.clear(); }

private:
    std::unordered_map<Graph, Laurent, GraphHash> memo_;

    Laurent rec(const Graph& graph) {
        auto found = memo_.find(graph);
        if (found != memo_.end()) return found->second;

        auto [reduced, factor] = reduce_homeomorphic(graph);
        if (!(reduced == graph) || factor != constant(1)) {
            Laurent value = multiply(factor, rec(reduced));
            memo_.emplace(graph, value);
            return value;
        }

        Scan info = scan(graph);
        Laurent value;
        if (info.edge_count == 0) {
            value = constant(graph.n % 2 ? -1 : 1);
        } else {
            auto comps = components(graph);
            if (comps.size() > 1) {
                value = constant(1);
                for (const auto& component : comps) {
                    value = multiply(value, rec(induced(graph, component)));
                }
            } else if (
                graph.n == 2 && graph.get(0, 0) == 0 && graph.get(1, 1) == 0 &&
                graph.get(0, 1) == info.edge_count
            ) {
                value = theta_value(info.edge_count);
            } else if (info.edge_count <= graph.n) {
                // With loops and degree-two vertices already removed, a connected
                // graph of cyclomatic number <= 1 must contain an isthmus.
                value = {};
            } else {
                TarjanResult tarjan = bridge_and_articulation(graph);
                if (tarjan.bridge) {
                    value = {};
                } else if (tarjan.articulation >= 0) {
                    auto parts = articulation_parts(graph, tarjan.articulation);
                    if (!parts.empty()) {
                        value = constant(1);
                        for (const auto& part : parts) {
                            value = multiply(value, rec(part));
                        }
                        if ((parts.size() - 1) % 2) value = scale(value, -1);
                    }
                }

                if (value.empty() && !tarjan.bridge && tarjan.articulation < 0) {
                    int best_i = -1;
                    int best_j = -1;
                    int best_multiplicity = 1;
                    for (int i = 0; i < graph.n; ++i) {
                        for (int j = i + 1; j < graph.n; ++j) {
                            int multiplicity = graph.get(i, j);
                            if (multiplicity > best_multiplicity) {
                                best_i = i;
                                best_j = j;
                                best_multiplicity = multiplicity;
                            }
                        }
                    }

                    if (best_i >= 0) {
                        Graph remainder = delete_parallel_class(graph, best_i, best_j);
                        Graph contracted = identify_vertices(remainder, best_i, best_j);
                        value = add(
                            rec(remainder),
                            multiply(
                                parallel_factor(best_multiplicity),
                                rec(contracted)
                            )
                        );
                    } else if (info.edge_i >= 0) {
                        value = add(
                            rec(delete_edge(graph, info.edge_i, info.edge_j)),
                            rec(contract_edge(graph, info.edge_i, info.edge_j))
                        );
                    } else {
                        value = constant(graph.n % 2 ? -1 : 1);
                    }
                }
            }
        }
        memo_.emplace(graph, value);
        return value;
    }
};

PYBIND11_MODULE(_yamada_native, module) {
    module.doc() = "Optional native exact-fast-path backend for Yamada evaluation";
    py::class_<NativeEvaluator>(module, "NativeEvaluator")
        .def(py::init<>())
        .def(
            "compute", &NativeEvaluator::compute,
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "compute_many", &NativeEvaluator::compute_many,
            py::call_guard<py::gil_scoped_release>()
        )
        .def("clear", &NativeEvaluator::clear)
        .def_property_readonly("memo_size", &NativeEvaluator::memo_size);
}
