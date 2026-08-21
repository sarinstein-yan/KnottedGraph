#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
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
        "native Yamada frontier coefficient exceeded int64; use exact Python fallback"
    );
}

Coeff checked_add(Coeff a, Coeff b) {
    constexpr Coeff hi = std::numeric_limits<Coeff>::max();
    constexpr Coeff lo = std::numeric_limits<Coeff>::min();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) coefficient_overflow();
    return static_cast<Coeff>(a + b);
}

Coeff checked_negate(Coeff value) {
    if (value == std::numeric_limits<Coeff>::min()) coefficient_overflow();
    return static_cast<Coeff>(-value);
}

void accumulate(Laurent& out, int power, Coeff value) {
    if (!value) return;
    auto it = out.find(power);
    if (it == out.end()) {
        out.emplace(power, value);
        return;
    }
    const Coeff merged = checked_add(it->second, value);
    if (!merged) out.erase(it);
    else it->second = merged;
}

Laurent add(const Laurent& left, const Laurent& right) {
    Laurent out = left;
    for (const auto& [power, coefficient] : right) accumulate(out, power, coefficient);
    return out;
}

Laurent scale(const Laurent& poly, int coefficient) {
    if (!coefficient || poly.empty()) return {};
    if (coefficient == 1) return poly;
    Laurent out;
    for (const auto& [power, value] : poly) {
        if (coefficient == -1) out.emplace(power, checked_negate(value));
        else {
            if (value > std::numeric_limits<Coeff>::max() / coefficient ||
                value < std::numeric_limits<Coeff>::min() / coefficient) {
                coefficient_overflow();
            }
            out.emplace(power, value * coefficient);
        }
    }
    return out;
}

Laurent shift(const Laurent& poly, int exponent) {
    if (poly.empty() || !exponent) return poly;
    Laurent out;
    for (const auto& [power, coefficient] : poly) out.emplace(power + exponent, coefficient);
    return out;
}

Laurent negative_q(const Laurent& poly) {
    // -(A^-1 + 2 + A) * poly.
    Laurent out;
    for (const auto& [power, coefficient] : poly) {
        const Coeff neg = checked_negate(coefficient);
        accumulate(out, power - 1, neg);
        accumulate(out, power, neg);
        accumulate(out, power, neg);
        accumulate(out, power + 1, neg);
    }
    return out;
}

std::vector<int> canonical(const std::vector<int>& labels) {
    std::unordered_map<int, int> remap;
    std::vector<int> out;
    out.reserve(labels.size());
    int next = 0;
    for (int label : labels) {
        auto [it, inserted] = remap.emplace(label, next);
        if (inserted) ++next;
        out.push_back(it->second);
    }
    return out;
}

bool unite(std::vector<int>& labels, int left, int right) {
    const int a = labels.at(static_cast<std::size_t>(left));
    const int b = labels.at(static_cast<std::size_t>(right));
    if (a == b) return true;
    const int low = std::min(a, b);
    const int high = std::max(a, b);
    for (int& value : labels) if (value == high) value = low;
    labels = canonical(labels);
    return false;
}

struct VecHash {
    std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::size_t h = 0x9e3779b97f4a7c15ULL;
        for (int value : values) {
            h ^= static_cast<std::size_t>(value + 0x9e3779b9) + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using Table = std::unordered_map<std::vector<int>, Laurent, VecHash>;

void accumulate_state(Table& table, const std::vector<int>& key, const Laurent& value) {
    if (value.empty()) return;
    auto it = table.find(key);
    if (it == table.end()) {
        table.emplace(key, value);
        return;
    }
    Laurent merged = add(it->second, value);
    if (merged.empty()) table.erase(it);
    else it->second = std::move(merged);
}

PythonLaurent to_python(const Laurent& poly) {
    PythonLaurent out;
    out.reserve(poly.size());
    for (const auto& [power, coefficient] : poly) out.emplace_back(power, coefficient);
    return out;
}

PythonLaurent compute_frontier(
    int vertex_count,
    int crossing_count,
    const std::vector<int>& arc_partner,
    const std::vector<int>& fixed_terminal_index,
    const std::vector<int>& crossing_for_port,
    const std::vector<int>& plus_partner,
    const std::vector<int>& minus_partner,
    const std::vector<int>& factor_order
) {
    if (vertex_count < 0 || crossing_count < 0) {
        throw std::invalid_argument("negative prepared Yamada dimensions");
    }
    const int port_count = static_cast<int>(arc_partner.size());
    if (static_cast<int>(fixed_terminal_index.size()) != port_count ||
        static_cast<int>(crossing_for_port.size()) != port_count ||
        static_cast<int>(plus_partner.size()) != port_count ||
        static_cast<int>(minus_partner.size()) != port_count) {
        throw std::invalid_argument("prepared Yamada table size mismatch");
    }
    const int factor_count = vertex_count + crossing_count;
    if (static_cast<int>(factor_order.size()) != factor_count) {
        throw std::invalid_argument("factor order size mismatch");
    }

    std::vector<std::vector<int>> factor_ports(static_cast<std::size_t>(factor_count));
    std::vector<int> port_factor(static_cast<std::size_t>(port_count), -1);
    for (int port = 0; port < port_count; ++port) {
        int factor = fixed_terminal_index[port];
        if (factor < 0) {
            const int crossing = crossing_for_port[port];
            if (crossing < 0 || crossing >= crossing_count) {
                throw std::runtime_error("prepared port belongs to no factor");
            }
            factor = vertex_count + crossing;
        }
        if (factor < 0 || factor >= factor_count) {
            throw std::runtime_error("prepared factor index out of range");
        }
        factor_ports[static_cast<std::size_t>(factor)].push_back(port);
        port_factor[static_cast<std::size_t>(port)] = factor;
    }
    for (auto& ports : factor_ports) std::sort(ports.begin(), ports.end());

    std::vector<int> step_of(static_cast<std::size_t>(factor_count), -1);
    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        if (factor < 0 || factor >= factor_count || step_of[static_cast<std::size_t>(factor)] >= 0) {
            throw std::invalid_argument("factor_order must be a permutation");
        }
        step_of[static_cast<std::size_t>(factor)] = step;
    }

    std::vector<std::vector<std::pair<int, int>>> backward_arcs(
        static_cast<std::size_t>(factor_count)
    );
    for (int port = 0; port < port_count; ++port) {
        const int partner = arc_partner[static_cast<std::size_t>(port)];
        if (partner < 0 || partner >= port_count) throw std::runtime_error("malformed arc partner");
        if (port >= partner) continue;
        const int left_factor = port_factor[static_cast<std::size_t>(port)];
        const int right_factor = port_factor[static_cast<std::size_t>(partner)];
        const int step = std::max(
            step_of[static_cast<std::size_t>(left_factor)],
            step_of[static_cast<std::size_t>(right_factor)]
        );
        backward_arcs[static_cast<std::size_t>(step)].emplace_back(port, partner);
    }

    Table states;
    states.emplace(std::vector<int>{}, Laurent{{0, 1}});
    std::vector<int> active;
    std::vector<char> processed(static_cast<std::size_t>(factor_count), 0);

    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        const auto& ports = factor_ports[static_cast<std::size_t>(factor)];
        active.insert(active.end(), ports.begin(), ports.end());
        std::unordered_map<int, int> position;
        position.reserve(active.size());
        for (int i = 0; i < static_cast<int>(active.size()); ++i) position[active[i]] = i;

        Table introduced;
        const bool is_crossing = factor >= vertex_count;
        const int crossing = factor - vertex_count;

        for (const auto& [old_labels, poly] : states) {
            int next_label = -1;
            for (int label : old_labels) next_label = std::max(next_label, label);
            ++next_label;
            std::vector<int> base = old_labels;
            for (int ignored : ports) {
                (void)ignored;
                base.push_back(next_label++);
            }
            base = canonical(base);

            const int option_count = is_crossing ? 3 : 1;
            for (int option = 0; option < option_count; ++option) {
                std::vector<int> labels = base;
                if (is_crossing) {
                    if (option == 2) {
                        if (!ports.empty()) {
                            const int anchor = position.at(ports.front());
                            for (std::size_t i = 1; i < ports.size(); ++i) {
                                if (unite(labels, anchor, position.at(ports[i]))) {
                                    throw std::runtime_error("crossing vertex unexpectedly closed local cycle");
                                }
                            }
                        }
                    } else {
                        const auto& partner_table = option == 0 ? plus_partner : minus_partner;
                        std::vector<char> seen(static_cast<std::size_t>(port_count), 0);
                        for (int port : ports) {
                            if (seen[static_cast<std::size_t>(port)]) continue;
                            const int other = partner_table[static_cast<std::size_t>(port)];
                            if (position.find(other) == position.end()) {
                                throw std::runtime_error("crossing resolution partner escaped factor");
                            }
                            seen[static_cast<std::size_t>(port)] = 1;
                            seen[static_cast<std::size_t>(other)] = 1;
                            if (unite(labels, position.at(port), position.at(other))) {
                                throw std::runtime_error("crossing smoothing unexpectedly closed local cycle");
                            }
                        }
                    }
                } else if (!ports.empty()) {
                    const int anchor = position.at(ports.front());
                    for (std::size_t i = 1; i < ports.size(); ++i) {
                        if (unite(labels, anchor, position.at(ports[i]))) {
                            throw std::runtime_error("fixed vertex unexpectedly closed local cycle");
                        }
                    }
                }

                Laurent value = poly;
                if (is_crossing) {
                    if (option == 0) value = shift(value, 1);
                    else if (option == 1) value = shift(value, -1);
                    else value = scale(value, -1);
                } else {
                    value = scale(value, -1);
                }
                accumulate_state(introduced, labels, value);
            }
        }
        states = std::move(introduced);

        for (const auto& [left_port, right_port] : backward_arcs[static_cast<std::size_t>(step)]) {
            const int left = position.at(left_port);
            const int right = position.at(right_port);
            Table updated;
            for (const auto& [labels, poly] : states) {
                accumulate_state(updated, labels, poly);  // excluded physical edge
                std::vector<int> merged = labels;
                const bool closes = unite(merged, left, right);
                const Laurent included = closes ? negative_q(poly) : scale(poly, -1);
                accumulate_state(updated, merged, included);
            }
            states = std::move(updated);
        }

        processed[static_cast<std::size_t>(factor)] = 1;
        std::vector<int> kept_positions;
        kept_positions.reserve(active.size());
        for (int i = 0; i < static_cast<int>(active.size()); ++i) {
            const int partner = arc_partner[static_cast<std::size_t>(active[static_cast<std::size_t>(i)])];
            const int partner_factor = port_factor[static_cast<std::size_t>(partner)];
            if (!processed[static_cast<std::size_t>(partner_factor)]) kept_positions.push_back(i);
        }
        if (kept_positions.size() != active.size()) {
            Table forgotten;
            for (const auto& [labels, poly] : states) {
                std::vector<int> kept;
                kept.reserve(kept_positions.size());
                for (int index : kept_positions) kept.push_back(labels[static_cast<std::size_t>(index)]);
                kept = canonical(kept);
                accumulate_state(forgotten, kept, poly);
            }
            states = std::move(forgotten);
            std::vector<int> new_active;
            new_active.reserve(kept_positions.size());
            for (int index : kept_positions) new_active.push_back(active[static_cast<std::size_t>(index)]);
            active = std::move(new_active);
        }
    }

    if (!active.empty()) throw std::runtime_error("native Yamada frontier did not close");
    auto found = states.find(std::vector<int>{});
    if (found == states.end()) return {};
    if (states.size() != 1) throw std::runtime_error("closed frontier retained nonempty states");
    return to_python(found->second);
}

}  // namespace

PYBIND11_MODULE(_yamada_frontier, module) {
    module.doc() = "Native polynomial-valued connectivity frontier for exact Yamada evaluation";
    module.def(
        "compute_prepared_frontier",
        &compute_frontier,
        py::call_guard<py::gil_scoped_release>()
    );
}
