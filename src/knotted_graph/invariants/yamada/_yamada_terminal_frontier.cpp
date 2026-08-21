#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using Coeff = std::int64_t;
using PythonLaurent = std::vector<std::pair<int, Coeff>>;

namespace {

[[noreturn]] void overflow() {
    throw std::overflow_error("terminal Yamada coefficient exceeded int64");
}

Coeff add_checked(Coeff a, Coeff b) {
    constexpr Coeff hi = std::numeric_limits<Coeff>::max();
    constexpr Coeff lo = std::numeric_limits<Coeff>::min();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) overflow();
    return a + b;
}

Coeff neg_checked(Coeff a) {
    if (a == std::numeric_limits<Coeff>::min()) overflow();
    return -a;
}

struct Poly {
    int lo{0};
    std::vector<Coeff> c;

    bool empty() const { return c.empty(); }

    void trim() {
        std::size_t first = 0;
        while (first < c.size() && c[first] == 0) ++first;
        if (first == c.size()) {
            c.clear();
            lo = 0;
            return;
        }
        std::size_t last = c.size();
        while (last > first && c[last - 1] == 0) --last;
        if (first) lo += static_cast<int>(first);
        if (first || last != c.size()) {
            c = std::vector<Coeff>(
                c.begin() + static_cast<std::ptrdiff_t>(first),
                c.begin() + static_cast<std::ptrdiff_t>(last)
            );
        }
    }

    void add_from(const Poly& src, int shift, int sign) {
        if (src.empty()) return;
        const int src_lo = src.lo + shift;
        const int src_hi = src_lo + static_cast<int>(src.c.size()) - 1;
        if (empty()) {
            lo = src_lo;
            c.resize(src.c.size());
            for (std::size_t i = 0; i < src.c.size(); ++i) {
                c[i] = sign > 0 ? src.c[i] : neg_checked(src.c[i]);
            }
            return;
        }
        const int old_hi = lo + static_cast<int>(c.size()) - 1;
        const int new_lo = std::min(lo, src_lo);
        const int new_hi = std::max(old_hi, src_hi);
        if (new_lo != lo || new_hi != old_hi) {
            std::vector<Coeff> out(static_cast<std::size_t>(new_hi - new_lo + 1), 0);
            std::copy(c.begin(), c.end(), out.begin() + (lo - new_lo));
            c.swap(out);
            lo = new_lo;
        }
        const int off = src_lo - lo;
        for (std::size_t i = 0; i < src.c.size(); ++i) {
            const Coeff v = sign > 0 ? src.c[i] : neg_checked(src.c[i]);
            c[static_cast<std::size_t>(off) + i] =
                add_checked(c[static_cast<std::size_t>(off) + i], v);
        }
    }
};

struct KeyHash {
    std::size_t operator()(const std::vector<int>& key) const noexcept {
        std::size_t h = 0xcbf29ce484222325ULL;
        for (int x : key) {
            h ^= static_cast<std::size_t>(x + 0x10001);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

// Key = [terminal_code, connectivity_label, ...].
using Table = std::unordered_map<std::vector<int>, Poly, KeyHash>;

std::vector<int> canon_labels(const std::vector<int>& labels) {
    int max_label = -1;
    for (int x : labels) max_label = std::max(max_label, x);
    std::vector<int> remap(static_cast<std::size_t>(max_label + 1), -1);
    std::vector<int> out;
    out.reserve(labels.size());
    int next = 0;
    for (int x : labels) {
        int& y = remap[static_cast<std::size_t>(x)];
        if (y < 0) y = next++;
        out.push_back(y);
    }
    return out;
}

void normalize(std::vector<int>& key) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(key.size() / 2);
    for (std::size_t i = 0; i < key.size(); i += 2) pairs.emplace_back(key[i], key[i + 1]);
    std::sort(pairs.begin(), pairs.end());
    std::vector<int> labels;
    labels.reserve(pairs.size());
    for (const auto& p : pairs) labels.push_back(p.second);
    labels = canon_labels(labels);
    key.clear();
    key.reserve(2 * pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        key.push_back(pairs[i].first);
        key.push_back(labels[i]);
    }
}

int next_label(const std::vector<int>& key) {
    int m = -1;
    for (std::size_t i = 1; i < key.size(); i += 2) m = std::max(m, key[i]);
    return m + 1;
}

bool unite(std::vector<int>& key, std::size_t p, std::size_t q) {
    const int a = key[2 * p + 1];
    const int b = key[2 * q + 1];
    if (a == b) return true;
    const int low = std::min(a, b);
    const int high = std::max(a, b);
    for (std::size_t i = 1; i < key.size(); i += 2) if (key[i] == high) key[i] = low;
    std::vector<int> labels;
    labels.reserve(key.size() / 2);
    for (std::size_t i = 1; i < key.size(); i += 2) labels.push_back(key[i]);
    labels = canon_labels(labels);
    for (std::size_t i = 0; i < labels.size(); ++i) key[2 * i + 1] = labels[i];
    return false;
}

void acc(Table& table, const std::vector<int>& key, const Poly& p, int shift, int sign) {
    if (p.empty()) return;
    auto [it, inserted] = table.try_emplace(key);
    it->second.add_from(p, shift, sign);
    if (!inserted) {
        it->second.trim();
        if (it->second.empty()) table.erase(it);
    }
}

void acc_cycle(Table& table, const std::vector<int>& key, const Poly& p) {
    // -(A^-1 + 2 + A) p.
    auto [it, inserted] = table.try_emplace(key);
    it->second.add_from(p, -1, -1);
    it->second.add_from(p, 0, -1);
    it->second.add_from(p, 0, -1);
    it->second.add_from(p, 1, -1);
    if (!inserted) {
        it->second.trim();
        if (it->second.empty()) table.erase(it);
    }
}

PythonLaurent export_poly(Poly p) {
    p.trim();
    PythonLaurent out;
    for (std::size_t i = 0; i < p.c.size(); ++i) {
        if (p.c[i]) out.emplace_back(p.lo + static_cast<int>(i), p.c[i]);
    }
    return out;
}

int fixed_code(int factor) { return factor * 32 + 16; }
int crossing_code(int factor, int mask) { return factor * 32 + mask; }

std::vector<int> pair_masks(
    const std::vector<int>& ports,
    const std::vector<int>& partners
) {
    std::unordered_map<int, int> local;
    for (int i = 0; i < 4; ++i) local.emplace(ports[static_cast<std::size_t>(i)], i);
    std::vector<char> seen(4, 0);
    std::vector<int> masks;
    for (int i = 0; i < 4; ++i) {
        if (seen[static_cast<std::size_t>(i)]) continue;
        auto found = local.find(partners[static_cast<std::size_t>(ports[static_cast<std::size_t>(i)])]);
        if (found == local.end()) throw std::runtime_error("crossing partner escaped factor");
        const int j = found->second;
        seen[static_cast<std::size_t>(i)] = seen[static_cast<std::size_t>(j)] = 1;
        masks.push_back((1 << i) | (1 << j));
    }
    std::sort(masks.begin(), masks.end());
    return masks;
}

std::size_t terminal_for_port(
    const std::vector<int>& key,
    int port,
    const std::vector<int>& factor_of,
    const std::vector<int>& local_bit,
    int vertex_count
) {
    const int factor = factor_of[static_cast<std::size_t>(port)];
    if (factor < vertex_count) {
        const int code = fixed_code(factor);
        for (std::size_t i = 0; i < key.size() / 2; ++i) if (key[2 * i] == code) return i;
    } else {
        const int bit = local_bit[static_cast<std::size_t>(port)];
        for (std::size_t i = 0; i < key.size() / 2; ++i) {
            const int code = key[2 * i];
            if (code / 32 == factor && ((code % 32) & bit)) return i;
        }
    }
    throw std::runtime_error("live terminal for port not found");
}

using Result = std::tuple<PythonLaurent, std::size_t, std::size_t, std::uint64_t>;

Result compute(
    int vertex_count,
    int crossing_count,
    const std::vector<int>& arc_partner,
    const std::vector<int>& fixed_terminal_index,
    const std::vector<int>& crossing_for_port,
    const std::vector<int>& plus_partner,
    const std::vector<int>& minus_partner,
    const std::vector<int>& factor_order
) {
    const int port_count = static_cast<int>(arc_partner.size());
    const int factor_count = vertex_count + crossing_count;
    if (vertex_count < 0 || crossing_count < 0 ||
        static_cast<int>(fixed_terminal_index.size()) != port_count ||
        static_cast<int>(crossing_for_port.size()) != port_count ||
        static_cast<int>(plus_partner.size()) != port_count ||
        static_cast<int>(minus_partner.size()) != port_count ||
        static_cast<int>(factor_order.size()) != factor_count) {
        throw std::invalid_argument("malformed prepared Yamada tables");
    }

    std::vector<std::vector<int>> factor_ports(static_cast<std::size_t>(factor_count));
    std::vector<int> factor_of(static_cast<std::size_t>(port_count), -1);
    for (int port = 0; port < port_count; ++port) {
        int factor = fixed_terminal_index[static_cast<std::size_t>(port)];
        if (factor < 0) {
            const int crossing = crossing_for_port[static_cast<std::size_t>(port)];
            if (crossing < 0 || crossing >= crossing_count) throw std::runtime_error("port has no factor");
            factor = vertex_count + crossing;
        }
        factor_ports[static_cast<std::size_t>(factor)].push_back(port);
        factor_of[static_cast<std::size_t>(port)] = factor;
    }
    for (auto& ports : factor_ports) std::sort(ports.begin(), ports.end());

    std::vector<int> local_bit(static_cast<std::size_t>(port_count), 0);
    std::vector<std::vector<int>> plus_masks(static_cast<std::size_t>(crossing_count));
    std::vector<std::vector<int>> minus_masks(static_cast<std::size_t>(crossing_count));
    for (int crossing = 0; crossing < crossing_count; ++crossing) {
        const int factor = vertex_count + crossing;
        const auto& ports = factor_ports[static_cast<std::size_t>(factor)];
        if (ports.size() != 4) throw std::runtime_error("crossing must have four ports");
        for (int i = 0; i < 4; ++i) local_bit[static_cast<std::size_t>(ports[static_cast<std::size_t>(i)])] = 1 << i;
        plus_masks[static_cast<std::size_t>(crossing)] = pair_masks(ports, plus_partner);
        minus_masks[static_cast<std::size_t>(crossing)] = pair_masks(ports, minus_partner);
    }

    std::vector<int> step_of(static_cast<std::size_t>(factor_count), -1);
    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        if (factor < 0 || factor >= factor_count || step_of[static_cast<std::size_t>(factor)] >= 0) {
            throw std::invalid_argument("factor order is not a permutation");
        }
        step_of[static_cast<std::size_t>(factor)] = step;
    }

    std::vector<std::vector<std::pair<int, int>>> arcs_at(static_cast<std::size_t>(factor_count));
    for (int port = 0; port < port_count; ++port) {
        const int partner = arc_partner[static_cast<std::size_t>(port)];
        if (partner < 0 || partner >= port_count) throw std::runtime_error("malformed arc partner");
        if (port >= partner) continue;
        const int step = std::max(
            step_of[static_cast<std::size_t>(factor_of[static_cast<std::size_t>(port)])],
            step_of[static_cast<std::size_t>(factor_of[static_cast<std::size_t>(partner)])]
        );
        arcs_at[static_cast<std::size_t>(step)].emplace_back(port, partner);
    }

    Table states;
    Poly one;
    one.c.push_back(1);
    states.emplace(std::vector<int>{}, std::move(one));
    std::vector<char> processed(static_cast<std::size_t>(factor_count), 0);
    std::size_t max_states = 1;
    std::size_t max_terminals = 0;
    std::uint64_t transitions = 0;

    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        const bool crossing = factor >= vertex_count;
        Table introduced;
        introduced.reserve(states.size() * (crossing ? 3U : 1U));

        for (const auto& [old_key, poly] : states) {
            const int first_label = next_label(old_key);
            const int option_count = crossing ? 3 : 1;
            for (int option = 0; option < option_count; ++option) {
                std::vector<int> key = old_key;
                int label = first_label;
                if (!crossing) {
                    key.push_back(fixed_code(factor));
                    key.push_back(label++);
                } else if (option == 2) {
                    key.push_back(crossing_code(factor, 15));
                    key.push_back(label++);
                } else {
                    const auto& masks = option == 0
                        ? plus_masks[static_cast<std::size_t>(factor - vertex_count)]
                        : minus_masks[static_cast<std::size_t>(factor - vertex_count)];
                    for (int mask : masks) {
                        key.push_back(crossing_code(factor, mask));
                        key.push_back(label++);
                    }
                }
                normalize(key);
                if (!crossing) acc(introduced, key, poly, 0, -1);
                else if (option == 0) acc(introduced, key, poly, 1, 1);
                else if (option == 1) acc(introduced, key, poly, -1, 1);
                else acc(introduced, key, poly, 0, -1);
                ++transitions;
            }
        }
        states = std::move(introduced);
        max_states = std::max(max_states, states.size());

        for (const auto& [left_port, right_port] : arcs_at[static_cast<std::size_t>(step)]) {
            Table updated;
            updated.reserve(states.size() * 2U);
            for (const auto& [key, poly] : states) {
                acc(updated, key, poly, 0, 1);
                std::vector<int> merged = key;
                const std::size_t left = terminal_for_port(merged, left_port, factor_of, local_bit, vertex_count);
                const std::size_t right = terminal_for_port(merged, right_port, factor_of, local_bit, vertex_count);
                const bool cycle = unite(merged, left, right);
                if (cycle) acc_cycle(updated, merged, poly);
                else acc(updated, merged, poly, 0, -1);
                transitions += 2;
            }
            states = std::move(updated);
            max_states = std::max(max_states, states.size());
        }

        processed[static_cast<std::size_t>(factor)] = 1;
        Table forgotten;
        forgotten.reserve(states.size());
        for (const auto& [key, poly] : states) {
            std::vector<int> kept;
            kept.reserve(key.size());
            for (std::size_t p = 0; p < key.size() / 2; ++p) {
                const int code = key[2 * p];
                const int f = code / 32;
                const int tag = code % 32;
                bool live = false;
                if (f < vertex_count) {
                    for (int port : factor_ports[static_cast<std::size_t>(f)]) {
                        const int pf = factor_of[static_cast<std::size_t>(arc_partner[static_cast<std::size_t>(port)])];
                        if (!processed[static_cast<std::size_t>(pf)]) { live = true; break; }
                    }
                } else {
                    const auto& ports = factor_ports[static_cast<std::size_t>(f)];
                    for (int i = 0; i < 4; ++i) {
                        if (!(tag & (1 << i))) continue;
                        const int pf = factor_of[static_cast<std::size_t>(arc_partner[static_cast<std::size_t>(ports[static_cast<std::size_t>(i)])])];
                        if (!processed[static_cast<std::size_t>(pf)]) { live = true; break; }
                    }
                }
                if (live) {
                    kept.push_back(code);
                    kept.push_back(key[2 * p + 1]);
                }
            }
            normalize(kept);
            max_terminals = std::max(max_terminals, kept.size() / 2);
            acc(forgotten, kept, poly, 0, 1);
            ++transitions;
        }
        states = std::move(forgotten);
        max_states = std::max(max_states, states.size());
    }

    auto found = states.find(std::vector<int>{});
    if (found == states.end()) return {PythonLaurent{}, max_states, max_terminals, transitions};
    if (states.size() != 1) throw std::runtime_error("terminal frontier retained live states");
    return {export_poly(found->second), max_states, max_terminals, transitions};
}

}  // namespace

PYBIND11_MODULE(_yamada_terminal_frontier, module) {
    module.doc() = "Exact quotient Yamada frontier over live local connectivity terminals";
    module.def(
        "compute_prepared_frontier",
        &compute,
        py::call_guard<py::gil_scoped_release>()
    );
}
