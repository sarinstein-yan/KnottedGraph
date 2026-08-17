#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using BigInt = boost::multiprecision::cpp_int;
using Laurent = std::map<int, BigInt>;

struct Graph {
    int n{};
    std::vector<int> a;

    int get(int i, int j) const { return a[static_cast<std::size_t>(i) * n + j]; }
    int& at(int i, int j) { return a[static_cast<std::size_t>(i) * n + j]; }
    bool operator==(const Graph& other) const { return n == other.n && a == other.a; }
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

static Graph from_rows(const std::vector<std::vector<int>>& rows) {
    const int n = static_cast<int>(rows.size());
    Graph graph;
    graph.n = n;
    graph.a.reserve(static_cast<std::size_t>(n) * n);
    for (const auto& row : rows) {
        if (static_cast<int>(row.size()) != n) {
            throw std::invalid_argument("compact graph matrix must be square");
        }
        graph.a.insert(graph.a.end(), row.begin(), row.end());
    }
    return graph;
}

static Laurent constant(int value) {
    Laurent out;
    if (value) out.emplace(0, value);
    return out;
}

static Laurent add(const Laurent& left, const Laurent& right) {
    Laurent out = left;
    for (const auto& [power, coefficient] : right) {
        auto& value = out[power];
        value += coefficient;
        if (value == 0) out.erase(power);
    }
    return out;
}

static Laurent scale(const Laurent& poly, int coefficient) {
    if (coefficient == 0 || poly.empty()) return {};
    if (coefficient == 1) return poly;
    Laurent out;
    for (const auto& [power, value] : poly) out.emplace(power, value * coefficient);
    return out;
}

static Laurent shift(const Laurent& poly, int exponent) {
    if (poly.empty() || exponent == 0) return poly;
    Laurent out;
    for (const auto& [power, value] : poly) out.emplace(power + exponent, value);
    return out;
}

static Laurent multiply(const Laurent& left, const Laurent& right) {
    if (left.empty() || right.empty()) return {};
    Laurent out;
    for (const auto& [p, a] : left) {
        for (const auto& [q, b] : right) out[p + q] += a * b;
    }
    for (auto it = out.begin(); it != out.end();) {
        if (it->second == 0) it = out.erase(it);
        else ++it;
    }
    return out;
}

static Laurent multiply_sigma(const Laurent& poly, int sign = 1) {
    if (poly.empty()) return {};
    Laurent out;
    for (const auto& [power, coefficient] : poly) {
        BigInt value = coefficient * sign;
        out[power - 1] += value;
        out[power] += value;
        out[power + 1] += value;
    }
    for (auto it = out.begin(); it != out.end();) {
        if (it->second == 0) it = out.erase(it);
        else ++it;
    }
    return out;
}

static Laurent theta_value(int theta) {
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

static Scan scan(const Graph& graph) {
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

static std::vector<std::vector<int>> components(const Graph& graph) {
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

static Graph induced(const Graph& graph, const std::vector<int>& nodes) {
    Graph out;
    out.n = static_cast<int>(nodes.size());
    out.a.assign(static_cast<std::size_t>(out.n) * out.n, 0);
    for (int i = 0; i < out.n; ++i) {
        for (int j = 0; j < out.n; ++j) out.at(i, j) = graph.get(nodes[i], nodes[j]);
    }
    return out;
}

struct TarjanResult {
    bool bridge{false};
    int articulation{-1};
};

static TarjanResult bridge_and_articulation(const Graph& graph) {
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
                    if (parent[u] == -1 && children > 1) result.articulation = u;
                    else if (parent[u] != -1 && low[v] >= disc[u]) result.articulation = u;
                }
            } else if (v != parent[u]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
    };
    for (int root = 0; root < graph.n; ++root) if (disc[root] == -1) dfs(root);
    return result;
}

static Graph delete_loop(const Graph& graph, int i) {
    Graph out = graph;
    --out.at(i, i);
    return out;
}

static Graph delete_edge(const Graph& graph, int i, int j) {
    Graph out = graph;
    --out.at(i, j);
    --out.at(j, i);
    return out;
}

static Graph contract_edge(const Graph& graph, int i, int j) {
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
    out.a.assign(static_cast<std::size_t>(out.n) * out.n, 0);
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

static std::vector<Graph> articulation_parts(const Graph& graph, int cut) {
    std::vector<int> remaining;
    for (int i = 0; i < graph.n; ++i) if (i != cut) remaining.push_back(i);
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

class NativeEvaluator {
public:
    Laurent compute_graph(const Graph& graph) { return rec(graph); }

    py::list compute(const std::vector<std::vector<int>>& rows) {
        return to_python(rec(from_rows(rows)));
    }

    py::list compute_many(
        const std::vector<std::vector<std::vector<int>>>& graphs,
        const std::vector<int>& exponents
    ) {
        if (graphs.size() != exponents.size()) throw std::invalid_argument("graphs/exponents size mismatch");
        Laurent total;
        for (std::size_t i = 0; i < graphs.size(); ++i) {
            total = add(total, shift(rec(from_rows(graphs[i])), exponents[i]));
        }
        return to_python(total);
    }

    std::size_t memo_size() const { return memo_.size(); }

private:
    std::unordered_map<Graph, Laurent, GraphHash> memo_;

    Laurent rec(const Graph& graph) {
        auto found = memo_.find(graph);
        if (found != memo_.end()) return found->second;

        Scan info = scan(graph);
        Laurent value;
        if (info.edge_count == 0) {
            value = constant(graph.n % 2 ? -1 : 1);
        } else {
            auto comps = components(graph);
            if (comps.size() > 1) {
                value = constant(1);
                for (const auto& component : comps) value = multiply(value, rec(induced(graph, component)));
            } else if (
                graph.n == 2 && graph.get(0, 0) == 0 && graph.get(1, 1) == 0 &&
                graph.get(0, 1) == info.edge_count
            ) {
                value = theta_value(info.edge_count);
            } else if (graph.n > 0 && std::all_of(info.degrees.begin(), info.degrees.end(), [](int d) { return d == 2; })) {
                value = Laurent{{-1, 1}, {0, 1}, {1, 1}};
            } else {
                TarjanResult tarjan = bridge_and_articulation(graph);
                if (tarjan.bridge) {
                    value = {};
                } else if (info.loop >= 0) {
                    value = multiply_sigma(rec(delete_loop(graph, info.loop)), -1);
                } else if (tarjan.articulation >= 0) {
                    auto parts = articulation_parts(graph, tarjan.articulation);
                    if (!parts.empty()) {
                        value = constant(1);
                        for (const auto& part : parts) value = multiply(value, rec(part));
                        if ((parts.size() - 1) % 2) value = scale(value, -1);
                    } else if (info.edge_i >= 0) {
                        value = add(rec(delete_edge(graph, info.edge_i, info.edge_j)), rec(contract_edge(graph, info.edge_i, info.edge_j)));
                    } else {
                        value = constant(graph.n % 2 ? -1 : 1);
                    }
                } else if (info.edge_i >= 0) {
                    value = add(rec(delete_edge(graph, info.edge_i, info.edge_j)), rec(contract_edge(graph, info.edge_i, info.edge_j)));
                } else {
                    value = constant(graph.n % 2 ? -1 : 1);
                }
            }
        }
        memo_.emplace(graph, value);
        return value;
    }

    static py::list to_python(const Laurent& poly) {
        py::list out;
        py::object py_int = py::module_::import("builtins").attr("int");
        for (const auto& [power, coefficient] : poly) {
            out.append(py::make_tuple(power, py_int(coefficient.convert_to<std::string>())));
        }
        return out;
    }
};

PYBIND11_MODULE(_kg_native_candidate, module) {
    py::class_<NativeEvaluator>(module, "NativeEvaluator")
        .def(py::init<>())
        .def("compute", &NativeEvaluator::compute)
        .def("compute_many", &NativeEvaluator::compute_many)
        .def_property_readonly("memo_size", &NativeEvaluator::memo_size);
}
