#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct RelationGraph {
    int n{};
    std::vector<int> labels;
    std::vector<std::uint8_t> adjacency;
    std::vector<int> refined_colors;
    std::string fingerprint;

    std::uint8_t get(int i, int j) const {
        return adjacency[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + j];
    }
    void add(int i, int j, std::uint8_t bit) {
        adjacency[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + j] |= bit;
    }
};

std::vector<int> compress_signatures(const std::vector<std::vector<int>>& signatures) {
    std::vector<std::vector<int>> unique = signatures;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::vector<int> colors(signatures.size(), 0);
    for (std::size_t i = 0; i < signatures.size(); ++i) {
        auto it = std::lower_bound(unique.begin(), unique.end(), signatures[i]);
        colors[i] = static_cast<int>(std::distance(unique.begin(), it));
    }
    return colors;
}

std::vector<int> refine_colors(const RelationGraph& graph) {
    std::vector<int> colors = graph.labels;
    while (true) {
        std::vector<std::vector<int>> signatures(static_cast<std::size_t>(graph.n));
        for (int node = 0; node < graph.n; ++node) {
            std::vector<std::tuple<int, int, int>> neighborhood;
            neighborhood.reserve(static_cast<std::size_t>(2 * graph.n));
            for (int other = 0; other < graph.n; ++other) {
                std::uint8_t out = graph.get(node, other);
                if (out) neighborhood.emplace_back(0, static_cast<int>(out), colors[other]);
                std::uint8_t in = graph.get(other, node);
                if (in) neighborhood.emplace_back(1, static_cast<int>(in), colors[other]);
            }
            std::sort(neighborhood.begin(), neighborhood.end());
            auto& signature = signatures[static_cast<std::size_t>(node)];
            signature.reserve(1 + 3 * neighborhood.size());
            signature.push_back(colors[node]);
            for (const auto& [direction, mask, color] : neighborhood) {
                signature.push_back(direction);
                signature.push_back(mask);
                signature.push_back(color);
            }
        }
        std::vector<int> next = compress_signatures(signatures);
        if (next == colors) return colors;
        const int old_classes = colors.empty() ? 0 : 1 + *std::max_element(colors.begin(), colors.end());
        const int new_classes = next.empty() ? 0 : 1 + *std::max_element(next.begin(), next.end());
        colors.swap(next);
        if (new_classes == old_classes) return colors;
    }
}

std::string make_fingerprint(const RelationGraph& graph, const std::vector<int>& colors) {
    int class_count = colors.empty() ? 0 : 1 + *std::max_element(colors.begin(), colors.end());
    std::vector<int> counts(static_cast<std::size_t>(class_count), 0);
    for (int color : colors) ++counts[static_cast<std::size_t>(color)];

    std::map<std::tuple<int, int, int>, int> edge_counts;
    for (int i = 0; i < graph.n; ++i) {
        for (int j = 0; j < graph.n; ++j) {
            std::uint8_t mask = graph.get(i, j);
            if (!mask) continue;
            ++edge_counts[std::make_tuple(colors[i], static_cast<int>(mask), colors[j])];
        }
    }

    std::ostringstream stream;
    stream << graph.n << ':' << class_count << ':';
    for (int value : counts) stream << value << ',';
    stream << '|';
    for (const auto& [key, value] : edge_counts) {
        const auto& [left, mask, right] = key;
        stream << left << '/' << mask << '/' << right << '=' << value << ';';
    }
    return stream.str();
}

RelationGraph build_relation_graph(
    int vertex_count,
    const std::vector<std::vector<int>>& ordered_ports,
    const std::vector<int>& arc_partner,
    const std::vector<int>& fixed_terminal_index,
    const std::vector<int>& crossing_for_port
) {
    if (vertex_count < 0) throw std::invalid_argument("negative vertex count");
    const int port_count = static_cast<int>(arc_partner.size());
    if (
        static_cast<int>(fixed_terminal_index.size()) != port_count ||
        static_cast<int>(crossing_for_port.size()) != port_count
    ) {
        throw std::invalid_argument("prepared diagram table size mismatch");
    }

    RelationGraph graph;
    graph.n = port_count + vertex_count;
    graph.labels.assign(static_cast<std::size_t>(graph.n), 3);
    graph.adjacency.assign(
        static_cast<std::size_t>(graph.n) * static_cast<std::size_t>(graph.n), 0
    );

    std::vector<int> seen_crossing_port(static_cast<std::size_t>(port_count), 0);
    for (std::size_t crossing = 0; crossing < ordered_ports.size(); ++crossing) {
        const auto& ports = ordered_ports[crossing];
        if (ports.size() != 4) throw std::invalid_argument("crossing must have four ports");
        for (int position = 0; position < 4; ++position) {
            int port = ports[static_cast<std::size_t>(position)];
            if (port < 0 || port >= port_count) throw std::invalid_argument("crossing port out of range");
            graph.labels[static_cast<std::size_t>(port)] = position % 2;
            seen_crossing_port[static_cast<std::size_t>(port)] = 1;
            int next = ports[static_cast<std::size_t>((position + 1) % 4)];
            graph.add(port, next, 2);  // directed cyclic crossing order
        }
    }

    for (int port = 0; port < port_count; ++port) {
        int partner = arc_partner[static_cast<std::size_t>(port)];
        if (partner < 0 || partner >= port_count || arc_partner[static_cast<std::size_t>(partner)] != port) {
            throw std::invalid_argument("malformed arc pairing");
        }
        if (port < partner) {
            graph.add(port, partner, 1);
            graph.add(partner, port, 1);
        }

        int crossing = crossing_for_port[static_cast<std::size_t>(port)];
        int terminal = fixed_terminal_index[static_cast<std::size_t>(port)];
        if (crossing >= 0) {
            if (!seen_crossing_port[static_cast<std::size_t>(port)]) {
                throw std::invalid_argument("crossing_for_port disagrees with ordered_ports");
            }
        } else {
            if (terminal < 0 || terminal >= vertex_count) {
                throw std::invalid_argument("terminal port has invalid vertex index");
            }
            graph.labels[static_cast<std::size_t>(port)] = 2;
            int vnode = port_count + terminal;
            graph.add(port, vnode, 4);
            graph.add(vnode, port, 4);
        }
    }

    graph.refined_colors = refine_colors(graph);
    graph.fingerprint = make_fingerprint(graph, graph.refined_colors);
    return graph;
}

bool exact_isomorphic(const RelationGraph& left, const RelationGraph& right) {
    if (left.n != right.n || left.fingerprint != right.fingerprint) return false;
    const int n = left.n;
    if (left.refined_colors.size() != right.refined_colors.size()) return false;

    std::map<int, std::vector<int>> right_by_color;
    for (int node = 0; node < n; ++node) {
        right_by_color[right.refined_colors[static_cast<std::size_t>(node)]].push_back(node);
    }

    std::vector<int> mapping(static_cast<std::size_t>(n), -1);
    std::vector<char> used(static_cast<std::size_t>(n), 0);

    auto compatible = [&](int a, int b) {
        if (left.labels[static_cast<std::size_t>(a)] != right.labels[static_cast<std::size_t>(b)]) return false;
        if (left.refined_colors[static_cast<std::size_t>(a)] != right.refined_colors[static_cast<std::size_t>(b)]) return false;
        for (int other = 0; other < n; ++other) {
            int mapped = mapping[static_cast<std::size_t>(other)];
            if (mapped < 0) continue;
            if (left.get(a, other) != right.get(b, mapped)) return false;
            if (left.get(other, a) != right.get(mapped, b)) return false;
        }
        return true;
    };

    std::function<bool(int)> search = [&](int mapped_count) -> bool {
        if (mapped_count == n) return true;

        int best_a = -1;
        std::vector<int> best_candidates;
        for (int a = 0; a < n; ++a) {
            if (mapping[static_cast<std::size_t>(a)] >= 0) continue;
            auto bucket_it = right_by_color.find(left.refined_colors[static_cast<std::size_t>(a)]);
            if (bucket_it == right_by_color.end()) return false;
            std::vector<int> candidates;
            for (int b : bucket_it->second) {
                if (used[static_cast<std::size_t>(b)]) continue;
                if (compatible(a, b)) candidates.push_back(b);
            }
            if (candidates.empty()) return false;
            if (best_a < 0 || candidates.size() < best_candidates.size()) {
                best_a = a;
                best_candidates = std::move(candidates);
                if (best_candidates.size() == 1) break;
            }
        }

        for (int b : best_candidates) {
            mapping[static_cast<std::size_t>(best_a)] = b;
            used[static_cast<std::size_t>(b)] = 1;
            if (search(mapped_count + 1)) return true;
            used[static_cast<std::size_t>(b)] = 0;
            mapping[static_cast<std::size_t>(best_a)] = -1;
        }
        return false;
    };

    return search(0);
}

}  // namespace

class PreparedDiagramIndex {
public:
    PreparedDiagramIndex(
        int vertex_count,
        const std::vector<std::vector<int>>& ordered_ports,
        const std::vector<int>& arc_partner,
        const std::vector<int>& fixed_terminal_index,
        const std::vector<int>& crossing_for_port
    ) : graph_(build_relation_graph(
        vertex_count,
        ordered_ports,
        arc_partner,
        fixed_terminal_index,
        crossing_for_port
    )) {}

    const std::string& fingerprint() const { return graph_.fingerprint; }
    int node_count() const { return graph_.n; }
    bool isomorphic(const PreparedDiagramIndex& other) const {
        return exact_isomorphic(graph_, other.graph_);
    }

private:
    RelationGraph graph_;
};

PYBIND11_MODULE(_yamada_iso, module) {
    module.doc() = "Exact native isomorphism index for prepared Yamada diagrams";
    py::class_<PreparedDiagramIndex>(module, "PreparedDiagramIndex")
        .def(
            py::init<
                int,
                const std::vector<std::vector<int>>&,
                const std::vector<int>&,
                const std::vector<int>&,
                const std::vector<int>&
            >(),
            py::arg("vertex_count"),
            py::arg("ordered_ports"),
            py::arg("arc_partner"),
            py::arg("fixed_terminal_index"),
            py::arg("crossing_for_port")
        )
        .def_property_readonly("fingerprint", &PreparedDiagramIndex::fingerprint)
        .def_property_readonly("node_count", &PreparedDiagramIndex::node_count)
        .def("isomorphic", &PreparedDiagramIndex::isomorphic, py::call_guard<py::gil_scoped_release>());
}
