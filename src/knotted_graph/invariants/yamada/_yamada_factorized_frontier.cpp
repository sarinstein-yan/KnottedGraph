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
using LaurentOut = std::vector<std::pair<int, Coeff>>;

namespace {
constexpr int EQ_NEG = 0;
constexpr int EQ_POS = 1;
constexpr int CROSSING = 2;
constexpr int PHYSICAL = 0;
constexpr int IDENTITY = 1;

[[noreturn]] void overflow() {
    throw std::overflow_error("factorized Yamada coefficient exceeded int64");
}

Coeff add_checked(Coeff a, Coeff b) {
    constexpr Coeff hi = std::numeric_limits<Coeff>::max();
    constexpr Coeff lo = std::numeric_limits<Coeff>::min();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b)) overflow();
    return a + b;
}

Coeff neg_checked(Coeff value) {
    if (value == std::numeric_limits<Coeff>::min()) overflow();
    return -value;
}

struct Poly {
    int lo{0};
    std::vector<Coeff> c;
    bool empty() const { return c.empty(); }

    void trim() {
        std::size_t first = 0;
        while (first < c.size() && c[first] == 0) ++first;
        if (first == c.size()) { c.clear(); lo = 0; return; }
        std::size_t last = c.size();
        while (last > first && c[last - 1] == 0) --last;
        if (first) lo += static_cast<int>(first);
        if (first || last != c.size()) {
            c = std::vector<Coeff>(c.begin() + static_cast<std::ptrdiff_t>(first),
                                   c.begin() + static_cast<std::ptrdiff_t>(last));
        }
    }

    void add_from(const Poly& src, int shift, int sign) {
        if (src.empty() || sign == 0) return;
        const int src_lo = src.lo + shift;
        const int src_hi = src_lo + static_cast<int>(src.c.size()) - 1;
        if (empty()) {
            lo = src_lo;
            c.resize(src.c.size());
            for (std::size_t i = 0; i < src.c.size(); ++i)
                c[i] = sign > 0 ? src.c[i] : neg_checked(src.c[i]);
            return;
        }
        const int old_hi = lo + static_cast<int>(c.size()) - 1;
        const int new_lo = std::min(lo, src_lo);
        const int new_hi = std::max(old_hi, src_hi);
        if (new_lo != lo || new_hi != old_hi) {
            std::vector<Coeff> expanded(static_cast<std::size_t>(new_hi - new_lo + 1), 0);
            std::copy(c.begin(), c.end(), expanded.begin() + (lo - new_lo));
            c.swap(expanded);
            lo = new_lo;
        }
        const int offset = src_lo - lo;
        for (std::size_t i = 0; i < src.c.size(); ++i) {
            const Coeff value = sign > 0 ? src.c[i] : neg_checked(src.c[i]);
            Coeff& target = c[static_cast<std::size_t>(offset) + i];
            target = add_checked(target, value);
        }
    }
};

void canonicalize(std::vector<int>& labels) {
    if (labels.empty()) return;
    int maximum = -1;
    for (int label : labels) maximum = std::max(maximum, label);
    std::vector<int> remap(static_cast<std::size_t>(maximum + 1), -1);
    int next = 0;
    for (int& label : labels) {
        int& mapped = remap[static_cast<std::size_t>(label)];
        if (mapped < 0) mapped = next++;
        label = mapped;
    }
}

std::vector<int> canonical(std::vector<int> labels) {
    canonicalize(labels);
    return labels;
}

bool unite(std::vector<int>& labels, int left, int right) {
    const int a = labels[static_cast<std::size_t>(left)];
    const int b = labels[static_cast<std::size_t>(right)];
    if (a == b) return true;
    const int low = std::min(a, b), high = std::max(a, b);
    for (int& value : labels) if (value == high) value = low;
    canonicalize(labels);
    return false;
}

struct KeyHash {
    std::size_t operator()(const std::vector<int>& key) const noexcept {
        std::size_t h = 0xcbf29ce484222325ULL;
        for (int value : key) {
            h ^= static_cast<std::size_t>(value + 1);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};
using Table = std::unordered_map<std::vector<int>, Poly, KeyHash>;

void accumulate(Table& table, const std::vector<int>& key, const Poly& poly, int shift, int sign) {
    if (poly.empty()) return;
    auto [it, inserted] = table.try_emplace(key);
    it->second.add_from(poly, shift, sign);
    if (!inserted) {
        it->second.trim();
        if (it->second.empty()) table.erase(it);
    }
}

void accumulate_q(Table& table, const std::vector<int>& key, const Poly& poly, int sign) {
    auto [it, inserted] = table.try_emplace(key);
    it->second.add_from(poly, -1, sign);
    it->second.add_from(poly, 0, sign);
    it->second.add_from(poly, 0, sign);
    it->second.add_from(poly, 1, sign);
    if (!inserted) {
        it->second.trim();
        if (it->second.empty()) table.erase(it);
    }
}

LaurentOut export_poly(Poly poly) {
    poly.trim();
    LaurentOut out;
    for (std::size_t i = 0; i < poly.c.size(); ++i)
        if (poly.c[i]) out.emplace_back(poly.lo + static_cast<int>(i), poly.c[i]);
    return out;
}

LaurentOut compute(
    const std::vector<int>& factor_types,
    const std::vector<int>& port_factor,
    const std::vector<int>& wire_partner,
    const std::vector<int>& wire_type,
    const std::vector<int>& plus_partner,
    const std::vector<int>& minus_partner,
    const std::vector<int>& factor_order
) {
    const int factor_count = static_cast<int>(factor_types.size());
    const int port_count = static_cast<int>(port_factor.size());
    if (static_cast<int>(wire_partner.size()) != port_count ||
        static_cast<int>(wire_type.size()) != port_count ||
        static_cast<int>(plus_partner.size()) != port_count ||
        static_cast<int>(minus_partner.size()) != port_count ||
        static_cast<int>(factor_order.size()) != factor_count)
        throw std::invalid_argument("factorized Yamada table size mismatch");

    std::vector<std::vector<int>> factor_ports(static_cast<std::size_t>(factor_count));
    for (int port = 0; port < port_count; ++port) {
        const int factor = port_factor[static_cast<std::size_t>(port)];
        if (factor < 0 || factor >= factor_count) throw std::runtime_error("port factor out of range");
        factor_ports[static_cast<std::size_t>(factor)].push_back(port);
        const int partner = wire_partner[static_cast<std::size_t>(port)];
        if (partner < 0 || partner >= port_count || wire_partner[static_cast<std::size_t>(partner)] != port)
            throw std::runtime_error("malformed factorized wire pairing");
        if (wire_type[static_cast<std::size_t>(port)] != wire_type[static_cast<std::size_t>(partner)])
            throw std::runtime_error("wire type mismatch");
    }
    for (auto& ports : factor_ports) std::sort(ports.begin(), ports.end());

    std::vector<int> step_of(static_cast<std::size_t>(factor_count), -1);
    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        if (factor < 0 || factor >= factor_count || step_of[static_cast<std::size_t>(factor)] >= 0)
            throw std::invalid_argument("factor order must be a permutation");
        step_of[static_cast<std::size_t>(factor)] = step;
    }

    std::vector<std::vector<std::pair<int, int>>> wires_at(static_cast<std::size_t>(factor_count));
    for (int port = 0; port < port_count; ++port) {
        const int partner = wire_partner[static_cast<std::size_t>(port)];
        if (port >= partner) continue;
        const int step = std::max(step_of[static_cast<std::size_t>(port_factor[port])],
                                  step_of[static_cast<std::size_t>(port_factor[partner])]);
        wires_at[static_cast<std::size_t>(step)].emplace_back(port, partner);
    }

    Table states;
    Poly one; one.c.push_back(1);
    states.emplace(std::vector<int>{}, std::move(one));
    std::vector<int> active;
    std::vector<char> processed(static_cast<std::size_t>(factor_count), 0);
    std::vector<int> position(static_cast<std::size_t>(port_count), -1);

    for (int step = 0; step < factor_count; ++step) {
        const int factor = factor_order[static_cast<std::size_t>(step)];
        const auto& ports = factor_ports[static_cast<std::size_t>(factor)];
        active.insert(active.end(), ports.begin(), ports.end());
        std::fill(position.begin(), position.end(), -1);
        for (int i = 0; i < static_cast<int>(active.size()); ++i)
            position[static_cast<std::size_t>(active[static_cast<std::size_t>(i)])] = i;

        const int type = factor_types[static_cast<std::size_t>(factor)];
        if (type != EQ_NEG && type != EQ_POS && type != CROSSING)
            throw std::runtime_error("unknown factor type");
        if (type == CROSSING && ports.size() != 4)
            throw std::runtime_error("crossing factor must have four ports");

        Table introduced;
        introduced.reserve(states.size() * (type == CROSSING ? 3U : 1U));
        for (const auto& [old_labels, poly] : states) {
            int next_label = 0;
            for (int label : old_labels) next_label = std::max(next_label, label + 1);
            std::vector<int> base = old_labels;
            for (std::size_t i = 0; i < ports.size(); ++i) base.push_back(next_label++);
            canonicalize(base);

            const int options = type == CROSSING ? 3 : 1;
            for (int option = 0; option < options; ++option) {
                std::vector<int> labels = base;
                if (type != CROSSING || option == 2) {
                    if (!ports.empty()) {
                        const int anchor = position[static_cast<std::size_t>(ports.front())];
                        if (anchor < 0) throw std::runtime_error("factor port escaped active frontier");
                        for (std::size_t i = 1; i < ports.size(); ++i) {
                            const int other_position = position[static_cast<std::size_t>(ports[i])];
                            if (other_position < 0)
                                throw std::runtime_error("factor port escaped active frontier");
                            if (unite(labels, anchor, other_position))
                                throw std::runtime_error("local equality unexpectedly closed cycle");
                        }
                    }
                } else {
                    const auto& partners = option == 0 ? plus_partner : minus_partner;
                    for (int port : ports) {
                        const int other = partners[static_cast<std::size_t>(port)];
                        if (other < 0 || other >= port_count)
                            throw std::runtime_error("crossing smoothing partner escaped factor");
                        const int left_position = position[static_cast<std::size_t>(port)];
                        const int right_position = position[static_cast<std::size_t>(other)];
                        if (left_position < 0 || right_position < 0)
                            throw std::runtime_error("crossing smoothing partner escaped factor");
                        // Each smoothing pair is symmetric; process it once by
                        // global port id instead of allocating a port_count-sized
                        // seen vector for every state transition.
                        if (port > other) continue;
                        if (unite(labels, left_position, right_position))
                            throw std::runtime_error("crossing smoothing unexpectedly closed cycle");
                    }
                }

                if (type == EQ_NEG) accumulate(introduced, labels, poly, 0, -1);
                else if (type == EQ_POS) accumulate(introduced, labels, poly, 0, 1);
                else if (option == 0) accumulate(introduced, labels, poly, 1, 1);
                else if (option == 1) accumulate(introduced, labels, poly, -1, 1);
                else accumulate(introduced, labels, poly, 0, -1);
            }
        }
        states = std::move(introduced);

        for (const auto& [left_port, right_port] : wires_at[static_cast<std::size_t>(step)]) {
            const int left = position[static_cast<std::size_t>(left_port)];
            const int right = position[static_cast<std::size_t>(right_port)];
            if (left < 0 || right < 0) throw std::runtime_error("wire escaped active frontier");
            const int kind = wire_type[static_cast<std::size_t>(left_port)];
            if (kind == IDENTITY) {
                Table updated;
                updated.reserve(states.size());
                for (const auto& [labels, poly] : states) {
                    std::vector<int> merged = labels;
                    const bool cycle = unite(merged, left, right);
                    if (cycle) accumulate_q(updated, merged, poly, 1);
                    else accumulate(updated, merged, poly, 0, 1);
                }
                states = std::move(updated);
            } else if (kind == PHYSICAL) {
                Table updated;
                updated.reserve(states.size() * 2U);
                for (const auto& [labels, poly] : states) {
                    accumulate(updated, labels, poly, 0, 1);
                    std::vector<int> merged = labels;
                    const bool cycle = unite(merged, left, right);
                    if (cycle) accumulate_q(updated, merged, poly, -1);
                    else accumulate(updated, merged, poly, 0, -1);
                }
                states = std::move(updated);
            } else {
                throw std::runtime_error("unknown wire type");
            }
        }

        processed[static_cast<std::size_t>(factor)] = 1;
        std::vector<int> kept_positions;
        for (int i = 0; i < static_cast<int>(active.size()); ++i) {
            const int partner = wire_partner[static_cast<std::size_t>(active[static_cast<std::size_t>(i)])];
            const int partner_factor = port_factor[static_cast<std::size_t>(partner)];
            if (!processed[static_cast<std::size_t>(partner_factor)]) kept_positions.push_back(i);
        }
        if (kept_positions.size() != active.size()) {
            Table forgotten;
            forgotten.reserve(states.size());
            for (const auto& [labels, poly] : states) {
                std::vector<int> kept;
                kept.reserve(kept_positions.size());
                for (int index : kept_positions) kept.push_back(labels[static_cast<std::size_t>(index)]);
                canonicalize(kept);
                accumulate(forgotten, kept, poly, 0, 1);
            }
            states = std::move(forgotten);
            std::vector<int> new_active;
            new_active.reserve(kept_positions.size());
            for (int index : kept_positions) new_active.push_back(active[static_cast<std::size_t>(index)]);
            active = std::move(new_active);
        }
    }

    if (!active.empty()) throw std::runtime_error("factorized frontier did not close");
    auto found = states.find(std::vector<int>{});
    if (found == states.end()) return {};
    if (states.size() != 1) throw std::runtime_error("factorized frontier retained nonempty states");
    return export_poly(found->second);
}

} // namespace

PYBIND11_MODULE(_yamada_factorized_frontier, module) {
    module.doc() = "Exact dense Yamada connectivity DP with low-arity equality factorization";
    module.def("compute_factorized_frontier", &compute, py::call_guard<py::gil_scoped_release>());
}
