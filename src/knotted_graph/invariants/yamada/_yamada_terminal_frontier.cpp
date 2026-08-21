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
        "native terminal Yamada frontier coefficient exceeded int64; use exact Python fallback"
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
            coeff = std::vector<Coeff>(
                coeff.begin() + static_cast<std::ptrdiff_t>(first),
                coeff.begin() + static_cast<std::ptrdiff_t>(last)
            );
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
            std::copy(coeff.begin(), coeff.end(), expanded.begin() + (lo - new_lo));
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

struct VecHash {
    std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::size_t h = 0xcbf29ce484222325ULL;
        for (int value : values) {
            h ^= static_cast<std::size_t>(value + 0x10001);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

// A state key is [terminal_code_0, label_0, terminal_code_1, label_1, ...].
// Terminal codes are globally ordered. Labels are canonicalized in that order.
using Table = std::unordered_map<std::vector<int>, DenseLaurent, VecHash>;

std::vector<int> canonical_labels(const std::vector<int>& labels) {
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

void normalize_key(std::vector<int>& key) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(key.size() / 2);
    for (std::size_t i = 0; i < key.size(); i += 2) {
        pairs.emplace_back(key[i], key[i + 1]);
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<int> labels;
    labels.reserve(pairs.size());
    for (const auto& [code, label] : pairs) {
        (void)code;
        labels.push_back(label);
    }
    labels = canonical_labels(labels);
    key.clear();
    key.reserve(2 * pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        key.push_back(pairs[i].first);
        key.push_back(labels[i]);
    }
}

int max_label(const std::vector<int>& key) {
    int result = -1;
    for (std::size_t i = 1; i < key.size(); i += 2) result = std::max(result, key[i]);
    return result;
}

bool unite_positions(std::vector<int>& key, std::size_t left_pair, std::size_t right_pair) {
    const std::size_t li = 2 * left_pair + 1;
    const std::size_t ri = 2 * right_pair + 1;
    const int a = key[li];
    const int b = key[ri];
    if (a == b) return true;
    const int low = std::min(a, b);
    const int high = std::max(a, b);
    for (std::size_t i = 1; i < key.size(); i += 2) {
        if (key[i] == high) key[i] = low;
    }
    // Codes are unchanged/sorted; only labels need canonicalization.
    std::vector<int> labels;
    labels.reserve(key.size() / 2);
    for (std::size_t i = 1; i < key.size(); i += 2) labels.push_back(key[i]);
    labels = canonical_labels(labels);
    for (std::size_t p = 0; p < labels.size(); ++p) key[2 * p + 1] = labels[p];
    return false;
}

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
    if (!inserted) {
        it->second.trim();
        if (it->second.empty()) table.erase(it);
    }
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
    if (!inserted) {
        target.trim();
        if (target.empty()) table.erase(it);
    }
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

int terminal_code_fixed(int factor) {
    return factor * 32 + 16;
}

int terminal_code_crossing(int factor, int local_mask) {
    return factor * 32 + local_mask;
}

std::size_t find_pair_for_port(
    const std::vector<int>& key,
    int port,
    const std::vector<int>& port_factor,
    const std::vector<int>& port_local_bit,
    int vertex_count
) {
    const int factor = port_factor[static_cast<std::size_t>(port)];
    if (factor < vertex_count) {
        const int code = terminal_code_fixed(factor);
        for (std::size_t p = 0; p < key.size() / 2; ++p) {
            if (key[2 * p] == code) return p;
        }
    } else {
        const int bit = port_local_bit[static_cast<std::size_t>(port)];
        const int base = factor * 32;
        for (std::size_t p = 0; p < key.size() / 2; ++p) {
            const int code = key[2 * p];
            if (code / 32 == factor && ((code - base) & bit)) return p;
        }
    }
    throw std::runtime_error("active terminal for prepared port not found");
}

std::vector<int> crossing_group_masks(
    const std::vector<int>& ports,
    const std::vector<int>& partner_table
) {
    std::unordered_map<int, int> local_index;
    for (int i = 0; i < static_cast<int>(ports.size()); ++i) local_index.emplace(ports[i], i);
    std::vector<char> seen(ports.size(), 0);
    std::vector<int> masks;
    for (int i = 0; i < static_cast<int>(ports.size()); ++i) {
        if (seen[static_cast<std::size_t>(i)]) continue;
        const int other = partner_table[static_cast<std::size_t>(ports[static_cast<std::size_t>(i)])];
        auto found = local_index.find(other);
        if (found == local_index.end()) throw std::runtime_error("crossing partner escaped factor");
        const int j = found->second;
        seen[static_cast<std::size_t>(i)] = 1;
        seen[static_cast<std::size_t>(j)] = 1;
        masks.push_back((1 << i) | (1 << j));
    }
    std::sort(masks.begin(), masks.end());
    return masks;
}

PythonLaurent compute_terminal_frontier(
    int vertex_count,
    int crossing_count,
    const std::vector<int>& arc_partner,
    const std::vector<int>& fixed_terminal_index,
    const std::vector<int>& crossing_for_port,
    const std::vector<int>& plus_partner,
    const std::vector<int>& minus_partner,
    const std::vector<int>& factor_order,
    py::dict stats
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

    std::vector<int> port_local_bit(static_cast<std::size_t>(port_count), 0);
    for (int factor = vertex_count; factor < factor_count; ++factor) {
        const auto& ports = factor_ports[static_cast<std::size_t>(factor)];
        if (ports.size() != 4) throw std::runtime_error("prepared crossing must have four ports");
        for (int i = 0; i < 4; ++i) {
            port_local_bit[static_cast<std::size_t>(ports[static_cast<std::size_t>(i)])] = 1 << i;
        }
    }

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
        const int left_factor = port_factor[static_cast<std::size_t>(port)];
        const int right_factor = port_factor[static_cast<std::size_t>(partner)];
        const int step = std::max(
            step_of[static_cast<std::size_t>(left_factor)],
            step_of[static_cast<std::size_t>(right_factor)]
        );
        arcs_at_step[static_cast<std::size_t>(step)].emplace_back(port, partner);
    }

    std::vector<std::vector<int>> plus_masks(static_cast<std::size_t>(crossing_count));
    std::vector<std::vector<int>> minus_masks(static_cast<std::size_t>(crossing_count));
    for (int crossing = 0; crossing < crossing_count; ++crossing) {
        const int factor = vertex_count + crossing;
        plus_masks[static_cast<std::size_t>(crossing)] = crossing_group_masks(
            factor_ports[static_cast<std::size_t>(factor)], plus_partner
        );
        minus_masks[static_cast<std::size_t>(crossing)] = crossing_group_masks(
            factor_ports[static_cast<std::size_t>(factor)], minus_partner
        );
    }

    Table states;
    DenseLaurent one;
    one.coeff.push_back(1);
    states.emplace(std::vector<int>{}, std::move(one));
    std::vector<char> processed(static_cast<std::size_t>(factor_count), 0);
    std::size_t max_states = 1;
    std::size_t max_terminals = 0;
    std::uint64_t transitions = 0;

    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        const bool is_crossing = factor >= vertex_count;
        Table introduced;
        introduced.reserve(states.size() * (is_crossing ? 3U : 1U));

        for (const auto& entry : states) {
            const auto& old_key = entry.first;
            const auto& poly = entry.second;
            const int fresh_start = max_label(old_key) + 1;
            const int options = is_crossing ? 3 : 1;
            for (int option = 0; option < options; ++option) {
                std::vector<int> key = old_key;
                int fresh = fresh_start;
                if (!is_crossing) {
                    key.push_back(terminal_code_fixed(factor));
                    key.push_back(fresh++);
                } else {
                    const int crossing = factor - vertex_count;
                    if (option == 2) {
                        key.push_back(terminal_code_crossing(factor, 15));
                        key.push_back(fresh++);
                    } else {
                        const auto& masks = option == 0
                            ? plus_masks[static_cast<std::size_t>(crossing)]
                            : minus_masks[static_cast<std::size_t>(crossing)];
                        for (int mask : masks) {
                            key.push_back(terminal_code_crossing(factor, mask));
                            key.push_back(fresh++);
                        }
                    }
                }
                normalize_key(key);
                if (!is_crossing) accumulate_transform(introduced, key, poly, 0, -1);
                else if (option == 0) accumulate_transform(introduced, key, poly, 1, 1);
                else if (option == 1) accumulate_transform(introduced, key, poly, -1, 1);
                else accumulate_transform(introduced, key, poly, 0, -1);
                ++transitions;
            }
        }
        states = std::move(introduced);
        max_states = std::max(max_states, states.size());

        for (const auto& [left_port, right_port] : arcs_at_step[static_cast<std::size_t>(step)]) {
            Table updated;
            updated.reserve(states.size() * 2U);
            for (const auto& entry : states) {
                const auto& key = entry.first;
                const auto& poly = entry.second;
                accumulate_transform(updated, key, poly, 0, 1); // delete/exclude edge

                std::vector<int> merged = key;
                const std::size_t left = find_pair_for_port(
                    merged, left_port, port_factor, port_local_bit, vertex_count
                );
                const std::size_t right = find_pair_for_port(
                    merged, right_port, port_factor, port_local_bit, vertex_count
                );
                const bool closes = unite_positions(merged, left, right);
                if (closes) accumulate_negative_q(updated, merged, poly);
                else accumulate_transform(updated, merged, poly, 0, -1);
                transitions += 2;
            }
            states = std::move(updated);
            max_states = std::max(max_states, states.size());
        }

        processed[static_cast<std::size_t>(factor)] = 1;
        Table forgotten;
        forgotten.reserve(states.size());
        for (const auto& entry : states) {
            const auto& key = entry.first;
            std::vector<int> kept;
            kept.reserve(key.size());
            for (std::size_t p = 0; p < key.size() / 2; ++p) {
                const int code = key[2 * p];
                const int terminal_factor = code / 32;
                const int tag = code - terminal_factor * 32;
                bool live = false;
                if (terminal_factor < vertex_count) {
                    for (int port : factor_ports[static_cast<std::size_t>(terminal_factor)]) {
                        const int partner = arc_partner[static_cast<std::size_t>(port)];
                        const int partner_factor = port_factor[static_cast<std::size_t>(partner)];
                        if (!processed[static_cast<std::size_t>(partner_factor)]) {
                            live = true;
                            break;
                        }
                    }
                } else {
                    const auto& ports = factor_ports[static_cast<std::size_t>(terminal_factor)];
                    for (int i = 0; i < 4; ++i) {
                        if (!(tag & (1 << i))) continue;
                        const int port = ports[static_cast<std::size_t>(i)];
                        const int partner = arc_partner[static_cast<std::size_t>(port)];
                        const int partner_factor = port_factor[static_cast<std::size_t>(partner)];
                        if (!processed[static_cast<std::size_t>(partner_factor)]) {
                            live = true;
                            break;
                        }
                    }
                }
                if (live) {
                    kept.push_back(code);
                    kept.push_back(key[2 * p + 1]);
                }
            }
            normalize_key(kept);
            max_terminals = std::max(max_terminals, kept.size() / 2);
            accumulate_transform(forgotten, kept, entry.second, 0, 1);
            ++transitions;
        }
        states = std::move(forgotten);
        max_states = std::max(max_states, states.size());
    }

    auto found = states.find(std::vector<int>{});
    if (found == states.end()) return {};
    if (states.size() != 1) throw std::runtime_error("terminal frontier retained live terminal states");

    stats["max_states"] = py::int_(max_states);
    stats["max_terminals"] = py::int_(max_terminals);
    stats["transitions"] = py::int_(transitions);
    return to_python(found->second);
}

}  // namespace

PYBIND11_MODULE(_yamada_terminal_frontier, module) {
    module.doc() = "Exact quotient Yamada frontier over live local connectivity terminals";
    module.def(
        "compute_prepared_frontier",
        &compute_terminal_frontier,
        py::arg("vertex_count"),
        py::arg("crossing_count"),
        py::arg("arc_partner"),
        py::arg("fixed_terminal_index"),
        py::arg("crossing_for_port"),
        py::arg("plus_partner"),
        py::arg("minus_partner"),
        py::arg("factor_order"),
        py::arg("stats") = py::dict(),
        py::call_guard<py::gil_scoped_release>()
    );
}
