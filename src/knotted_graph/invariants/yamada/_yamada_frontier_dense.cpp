#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using Coeff = std::int64_t;
using PythonLaurent = std::vector<std::pair<int, Coeff>>;

namespace {

[[noreturn]] void coefficient_overflow() {
    throw std::overflow_error(
        "native dense Yamada frontier coefficient exceeded int64; use exact Python fallback"
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

struct DenseLaurent {
    int lo{0};
    std::vector<Coeff> coeff;

    bool empty() const { return coeff.empty(); }

    void trim() {
        std::size_t first = 0;
        while (first < coeff.size() && coeff[first] == 0) ++first;
        if (first == coeff.size()) {
            coeff.clear();
            lo = 0;
            return;
        }
        std::size_t last = coeff.size();
        while (last > first && coeff[last - 1] == 0) --last;
        if (first) lo += static_cast<int>(first);
        if (first || last != coeff.size()) {
            coeff = std::vector<Coeff>(coeff.begin() + static_cast<std::ptrdiff_t>(first),
                                       coeff.begin() + static_cast<std::ptrdiff_t>(last));
        }
    }

    void add_scaled_shifted(const DenseLaurent& other, int exponent_shift, int sign) {
        if (other.empty() || sign == 0) return;
        const int other_lo = other.lo + exponent_shift;
        const int other_hi = other_lo + static_cast<int>(other.coeff.size()) - 1;
        if (empty()) {
            lo = other_lo;
            coeff.resize(other.coeff.size());
            for (std::size_t i = 0; i < other.coeff.size(); ++i) {
                coeff[i] = sign == 1 ? other.coeff[i] : checked_negate(other.coeff[i]);
            }
            return;
        }
        const int old_hi = lo + static_cast<int>(coeff.size()) - 1;
        const int new_lo = std::min(lo, other_lo);
        const int new_hi = std::max(old_hi, other_hi);
        if (new_lo != lo || new_hi != old_hi) {
            std::vector<Coeff> expanded(static_cast<std::size_t>(new_hi - new_lo + 1), 0);
            const int offset = lo - new_lo;
            std::copy(coeff.begin(), coeff.end(), expanded.begin() + offset);
            coeff.swap(expanded);
            lo = new_lo;
        }
        const int offset = other_lo - lo;
        for (std::size_t i = 0; i < other.coeff.size(); ++i) {
            const Coeff value = sign == 1 ? other.coeff[i] : checked_negate(other.coeff[i]);
            Coeff& target = coeff[static_cast<std::size_t>(offset) + i];
            target = checked_add(target, value);
        }
    }
};

std::vector<int> canonical(const std::vector<int>& labels) {
    // Frontier labels are always small canonical integers. A vector remap avoids
    // hashing each label during the hottest state transition.
    int maximum = -1;
    for (int label : labels) maximum = std::max(maximum, label);
    std::vector<int> remap(static_cast<std::size_t>(maximum + 1), -1);
    std::vector<int> out;
    out.reserve(labels.size());
    int next = 0;
    for (int label : labels) {
        int& mapped = remap[static_cast<std::size_t>(label)];
        if (mapped < 0) mapped = next++;
        out.push_back(mapped);
    }
    return out;
}

bool unite(std::vector<int>& labels, int left, int right) {
    const int a = labels[static_cast<std::size_t>(left)];
    const int b = labels[static_cast<std::size_t>(right)];
    if (a == b) return true;
    const int low = std::min(a, b);
    const int high = std::max(a, b);
    for (int& value : labels) if (value == high) value = low;
    labels = canonical(labels);
    return false;
}

struct VecHash {
    std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::size_t h = 0xcbf29ce484222325ULL;
        for (int value : values) {
            h ^= static_cast<std::size_t>(value + 1);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

using Table = std::unordered_map<std::vector<int>, DenseLaurent, VecHash>;

void accumulate_transform(
    Table& table,
    const std::vector<int>& key,
    const DenseLaurent& poly,
    int exponent_shift,
    int sign
) {
    if (poly.empty()) return;
    auto [it, inserted] = table.try_emplace(key);
    it->second.add_scaled_shifted(poly, exponent_shift, sign);
    if (!inserted && it->second.empty()) table.erase(it);
}

void accumulate_negative_q(
    Table& table,
    const std::vector<int>& key,
    const DenseLaurent& poly
) {
    if (poly.empty()) return;
    auto [it, inserted] = table.try_emplace(key);
    DenseLaurent& target = it->second;
    target.add_scaled_shifted(poly, -1, -1);
    target.add_scaled_shifted(poly, 0, -1);
    target.add_scaled_shifted(poly, 0, -1);
    target.add_scaled_shifted(poly, 1, -1);
    if (!inserted && target.empty()) table.erase(it);
}

PythonLaurent to_python(DenseLaurent poly) {
    poly.trim();
    PythonLaurent out;
    out.reserve(poly.coeff.size());
    for (std::size_t i = 0; i < poly.coeff.size(); ++i) {
        if (poly.coeff[i]) out.emplace_back(poly.lo + static_cast<int>(i), poly.coeff[i]);
    }
    return out;
}

PythonLaurent compute_frontier_dense(
    int vertex_count,
    int crossing_count,
    const std::vector<int>& arc_partner,
    const std::vector<int>& fixed_terminal_index,
    const std::vector<int>& crossing_for_port,
    const std::vector<int>& plus_partner,
    const std::vector<int>& minus_partner,
    const std::vector<int>& factor_order
) {
    if (vertex_count < 0 || crossing_count < 0) throw std::invalid_argument("negative dimensions");
    const int port_count = static_cast<int>(arc_partner.size());
    if (static_cast<int>(fixed_terminal_index.size()) != port_count ||
        static_cast<int>(crossing_for_port.size()) != port_count ||
        static_cast<int>(plus_partner.size()) != port_count ||
        static_cast<int>(minus_partner.size()) != port_count) {
        throw std::invalid_argument("prepared table size mismatch");
    }
    const int factor_count = vertex_count + crossing_count;
    if (static_cast<int>(factor_order.size()) != factor_count) throw std::invalid_argument("factor order size mismatch");

    std::vector<std::vector<int>> factor_ports(static_cast<std::size_t>(factor_count));
    std::vector<int> port_factor(static_cast<std::size_t>(port_count), -1);
    for (int port = 0; port < port_count; ++port) {
        int factor = fixed_terminal_index[static_cast<std::size_t>(port)];
        if (factor < 0) {
            const int crossing = crossing_for_port[static_cast<std::size_t>(port)];
            if (crossing < 0 || crossing >= crossing_count) throw std::runtime_error("port has no factor");
            factor = vertex_count + crossing;
        }
        if (factor < 0 || factor >= factor_count) throw std::runtime_error("factor index out of range");
        factor_ports[static_cast<std::size_t>(factor)].push_back(port);
        port_factor[static_cast<std::size_t>(port)] = factor;
    }
    for (auto& ports : factor_ports) std::sort(ports.begin(), ports.end());

    std::vector<int> step_of(static_cast<std::size_t>(factor_count), -1);
    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        if (factor < 0 || factor >= factor_count || step_of[static_cast<std::size_t>(factor)] >= 0) {
            throw std::invalid_argument("factor order must be a permutation");
        }
        step_of[static_cast<std::size_t>(factor)] = step;
    }

    std::vector<std::vector<std::pair<int, int>>> arcs_at_step(static_cast<std::size_t>(factor_count));
    for (int port = 0; port < port_count; ++port) {
        const int partner = arc_partner[static_cast<std::size_t>(port)];
        if (partner < 0 || partner >= port_count) throw std::runtime_error("malformed arc partner");
        if (port >= partner) continue;
        const int step = std::max(
            step_of[static_cast<std::size_t>(port_factor[static_cast<std::size_t>(port)])],
            step_of[static_cast<std::size_t>(port_factor[static_cast<std::size_t>(partner)])]
        );
        arcs_at_step[static_cast<std::size_t>(step)].emplace_back(port, partner);
    }

    Table states;
    DenseLaurent one;
    one.coeff.push_back(1);
    states.emplace(std::vector<int>{}, std::move(one));
    std::vector<int> active;
    std::vector<char> processed(static_cast<std::size_t>(factor_count), 0);

    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        const auto& ports = factor_ports[static_cast<std::size_t>(factor)];
        active.insert(active.end(), ports.begin(), ports.end());
        std::unordered_map<int, int> position;
        position.reserve(active.size() * 2 + 1);
        for (int i = 0; i < static_cast<int>(active.size()); ++i) position.emplace(active[static_cast<std::size_t>(i)], i);

        Table introduced;
        introduced.reserve(states.size() * (factor >= vertex_count ? 3U : 1U));
        const bool is_crossing = factor >= vertex_count;
        for (const auto& entry : states) {
            const auto& old_labels = entry.first;
            const auto& poly = entry.second;
            int next_label = 0;
            for (int label : old_labels) next_label = std::max(next_label, label + 1);
            std::vector<int> base = old_labels;
            for (std::size_t i = 0; i < ports.size(); ++i) base.push_back(next_label++);
            base = canonical(base);

            const int options = is_crossing ? 3 : 1;
            for (int option = 0; option < options; ++option) {
                std::vector<int> labels = base;
                if (is_crossing && option < 2) {
                    const auto& partners = option == 0 ? plus_partner : minus_partner;
                    std::vector<char> seen(static_cast<std::size_t>(port_count), 0);
                    for (int port : ports) {
                        if (seen[static_cast<std::size_t>(port)]) continue;
                        const int other = partners[static_cast<std::size_t>(port)];
                        auto found = position.find(other);
                        if (found == position.end()) throw std::runtime_error("resolution escaped crossing factor");
                        seen[static_cast<std::size_t>(port)] = 1;
                        seen[static_cast<std::size_t>(other)] = 1;
                        if (unite(labels, position.at(port), found->second)) throw std::runtime_error("local smoothing cycle");
                    }
                } else if (!ports.empty()) {
                    const int anchor = position.at(ports.front());
                    for (std::size_t i = 1; i < ports.size(); ++i) {
                        if (unite(labels, anchor, position.at(ports[i]))) throw std::runtime_error("local vertex cycle");
                    }
                }

                if (!is_crossing) accumulate_transform(introduced, labels, poly, 0, -1);
                else if (option == 0) accumulate_transform(introduced, labels, poly, 1, 1);
                else if (option == 1) accumulate_transform(introduced, labels, poly, -1, 1);
                else accumulate_transform(introduced, labels, poly, 0, -1);
            }
        }
        states = std::move(introduced);

        for (const auto& [left_port, right_port] : arcs_at_step[static_cast<std::size_t>(step)]) {
            const int left = position.at(left_port);
            const int right = position.at(right_port);
            Table updated;
            updated.reserve(states.size() * 2U);
            for (const auto& entry : states) {
                const auto& labels = entry.first;
                const auto& poly = entry.second;
                accumulate_transform(updated, labels, poly, 0, 1);
                std::vector<int> merged = labels;
                const bool closes = unite(merged, left, right);
                if (closes) accumulate_negative_q(updated, merged, poly);
                else accumulate_transform(updated, merged, poly, 0, -1);
            }
            states = std::move(updated);
        }

        processed[static_cast<std::size_t>(factor)] = 1;
        std::vector<int> kept_positions;
        for (int i = 0; i < static_cast<int>(active.size()); ++i) {
            const int partner = arc_partner[static_cast<std::size_t>(active[static_cast<std::size_t>(i)])];
            const int partner_factor = port_factor[static_cast<std::size_t>(partner)];
            if (!processed[static_cast<std::size_t>(partner_factor)]) kept_positions.push_back(i);
        }
        if (kept_positions.size() != active.size()) {
            Table forgotten;
            forgotten.reserve(states.size());
            for (const auto& entry : states) {
                std::vector<int> kept;
                kept.reserve(kept_positions.size());
                for (int index : kept_positions) kept.push_back(entry.first[static_cast<std::size_t>(index)]);
                kept = canonical(kept);
                accumulate_transform(forgotten, kept, entry.second, 0, 1);
            }
            states = std::move(forgotten);
            std::vector<int> new_active;
            new_active.reserve(kept_positions.size());
            for (int index : kept_positions) new_active.push_back(active[static_cast<std::size_t>(index)]);
            active = std::move(new_active);
        }
    }

    if (!active.empty()) throw std::runtime_error("dense Yamada frontier did not close");
    auto found = states.find(std::vector<int>{});
    if (found == states.end()) return {};
    if (states.size() != 1) throw std::runtime_error("closed dense frontier retained states");
    return to_python(found->second);
}

}  // namespace

PYBIND11_MODULE(_yamada_frontier_dense, module) {
    module.doc() = "Fused dense exact Yamada connectivity-state contraction";
    module.def(
        "compute_prepared_frontier",
        &compute_frontier_dense,
        py::call_guard<py::gil_scoped_release>()
    );
}
