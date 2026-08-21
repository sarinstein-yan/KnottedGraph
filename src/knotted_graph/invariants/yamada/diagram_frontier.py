"""Exact width-adaptive frontier evaluator for a prepared Yamada diagram.

This evaluates the Yamada state sum and the crossing-free graph polynomial in
one connectivity dynamic program. It never enumerates all 3**c resolved states.
The algorithm is generic: its complexity is governed by the frontier width of
the diagram factor graph (fixed vertices + crossings joined by physical arcs),
not by a recognized graph family.

For a resolved graph G,
    H(G; A) = (-1)^|V| sum_{F subseteq E} (-1)^|F| q^beta(F),
    q = A + 2 + A^-1.
We keep degree-two smoothing vertices rather than suppressing them; H is
homeomorphism invariant, so this leaves the exact polynomial unchanged. Thus a
crossing contributes one of three local vertex partitions:
  + smoothing: A times two two-port vertices,
  - smoothing: A^-1 times two two-port vertices,
  vertex state: one four-port vertex, hence an extra -1 from (-1)^|V|.
Each fixed graph vertex similarly contributes -1. Physical arcs are then the
edges of G; excluded/included edge choices have weights +1/-1, and an included
edge raises beta exactly when it closes a connectivity cycle.
"""

from __future__ import annotations

from collections import defaultdict


def _canonical(labels):
    remap = {}
    out = []
    next_label = 0
    for label in labels:
        if label not in remap:
            remap[label] = next_label
            next_label += 1
        out.append(remap[label])
    return tuple(out)


def _union(labels, left, right):
    a = labels[left]
    b = labels[right]
    if a == b:
        return labels, True
    low, high = (a, b) if a < b else (b, a)
    return _canonical(tuple(low if value == high else value for value in labels)), False


def _factor_graph(prepared):
    vertex_count = len(prepared.vertex_ids)
    crossing_count = len(prepared.crossing_ids)
    factor_count = vertex_count + crossing_count
    factor_ports = [[] for _ in range(factor_count)]
    port_factor = [-1] * len(prepared.arc_partner)

    for port in range(len(prepared.arc_partner)):
        fixed = prepared.fixed_terminal_index[port]
        crossing = prepared.crossing_for_port[port]
        if fixed >= 0:
            factor = int(fixed)
        elif crossing >= 0:
            factor = vertex_count + int(crossing)
        else:
            raise RuntimeError("prepared port belongs to no vertex/crossing factor")
        factor_ports[factor].append(port)
        port_factor[port] = factor

    adjacency = [defaultdict(int) for _ in range(factor_count)]
    arcs = []
    for port, partner in enumerate(prepared.arc_partner):
        if port >= partner:
            continue
        left = port_factor[port]
        right = port_factor[partner]
        arcs.append((port, partner, left, right))
        if left != right:
            adjacency[left][right] += 1
            adjacency[right][left] += 1

    return factor_ports, port_factor, adjacency, arcs


def _greedy_factor_order(adjacency, factor_ports):
    """Generic cutwidth-oriented factor order with deterministic tie breaking."""
    count = len(adjacency)
    if count <= 1:
        return list(range(count))
    unprocessed = set(range(count))
    processed = set()
    order = []

    # Begin at a low external-degree factor; isolated factors are harmless first.
    first = min(
        unprocessed,
        key=lambda node: (sum(adjacency[node].values()), len(factor_ports[node]), node),
    )
    order.append(first)
    processed.add(first)
    unprocessed.remove(first)

    while unprocessed:
        def score(node):
            back = sum(mult for other, mult in adjacency[node].items() if other in processed)
            future = sum(mult for other, mult in adjacency[node].items() if other in unprocessed)
            # Prefer extending the processed region, then minimize new cut arcs.
            disconnected = 1 if back == 0 and processed else 0
            return (disconnected, future - back, future, -back, len(factor_ports[node]), node)

        node = min(unprocessed, key=score)
        order.append(node)
        processed.add(node)
        unprocessed.remove(node)
    return order


def _crossing_groups(prepared, crossing_index, spin):
    ports = prepared.ordered_ports[crossing_index]
    if spin == 2:
        return (tuple(ports),)
    partner = prepared.plus_partner if spin == 0 else prepared.minus_partner
    groups = []
    seen = set()
    for port in ports:
        if port in seen:
            continue
        other = partner[port]
        if other not in ports:
            raise RuntimeError("crossing resolution partner escaped crossing")
        groups.append((port, other))
        seen.add(port)
        seen.add(other)
    if len(groups) != 2:
        raise RuntimeError("smoothing did not produce two local degree-two vertices")
    return tuple(groups)


def _apply_groups(labels, positions, groups):
    current = labels
    # New ports have no incident processed arcs yet, so local vertex
    # identifications cannot close a cycle at introduction time.
    for group in groups:
        anchor = positions[group[0]]
        for port in group[1:]:
            current, closes = _union(current, anchor, positions[port])
            if closes:
                # This should only happen if the same port was repeated.
                raise RuntimeError("local vertex partition unexpectedly closed a cycle")
    return current


def _q_to_laurent(weight_by_a_beta):
    total = defaultdict(int)
    q_powers = {0: {0: 1}}

    def q_power(beta):
        while beta not in q_powers:
            k = max(q_powers)
            prev = q_powers[k]
            nxt = defaultdict(int)
            for power, coeff in prev.items():
                nxt[power - 1] += coeff
                nxt[power] += 2 * coeff
                nxt[power + 1] += coeff
            q_powers[k + 1] = dict(nxt)
        return q_powers[beta]

    for (a_power, beta), coefficient in weight_by_a_beta.items():
        if not coefficient:
            continue
        for q_power_exp, q_coeff in q_power(beta).items():
            total[a_power + q_power_exp] += coefficient * q_coeff
    return tuple(sorted((power, coeff) for power, coeff in total.items() if coeff))


def compute_diagram_frontier_laurent(prepared, *, factor_order=None, stats=None):
    """Return the exact Laurent Yamada polynomial by connectivity-frontier DP."""
    if stats is None:
        stats = {}
    factor_ports, port_factor, adjacency, arcs = _factor_graph(prepared)
    vertex_count = len(prepared.vertex_ids)
    crossing_count = len(prepared.crossing_ids)
    order = _greedy_factor_order(adjacency, factor_ports) if factor_order is None else list(factor_order)
    if sorted(order) != list(range(len(factor_ports))):
        raise ValueError("factor_order must contain every fixed vertex/crossing exactly once")
    step_of = {factor: step for step, factor in enumerate(order)}

    backward_arcs = [[] for _ in order]
    future_arc_count = [0] * len(prepared.arc_partner)
    for p, q, left, right in arcs:
        sl = step_of[left]
        sr = step_of[right]
        step = max(sl, sr)
        backward_arcs[step].append((p, q))
        if sl < sr:
            future_arc_count[p] += 1
        elif sr < sl:
            future_arc_count[q] += 1

    active = []
    # key=(canonical boundary partition, A exponent, beta), value=integer coeff
    states = {((), 0, 0): 1}
    max_frontier = 0
    max_states = 1
    transitions = 0

    for step, factor in enumerate(order):
        ports = sorted(factor_ports[factor])
        old_len = len(active)
        active.extend(ports)
        positions = {port: index for index, port in enumerate(active)}

        introduced = defaultdict(int)
        is_crossing = factor >= vertex_count
        if is_crossing:
            crossing = factor - vertex_count
            options = (
                (0, 1, 1),   # spin, A exponent, local (-1)^V sign
                (1, -1, 1),
                (2, 0, -1),
            )
        else:
            options = ((-1, 0, -1),)  # fixed vertex contributes one vertex sign

        for (labels, a_power, beta), coefficient in states.items():
            base_labels = labels + tuple(
                range(max(labels, default=-1) + 1, max(labels, default=-1) + 1 + len(ports))
            )
            base_labels = _canonical(base_labels)
            for spin, a_delta, local_sign in options:
                if is_crossing:
                    groups = _crossing_groups(prepared, crossing, spin)
                else:
                    groups = (tuple(ports),) if ports else ()
                new_labels = _apply_groups(base_labels, positions, groups)
                introduced[(new_labels, a_power + a_delta, beta)] += coefficient * local_sign
                transitions += 1
        states = {key: value for key, value in introduced.items() if value}

        # Every physical arc is processed exactly when its later endpoint factor
        # is introduced (or immediately for a same-factor loop).
        for left_port, right_port in backward_arcs[step]:
            left = positions[left_port]
            right = positions[right_port]
            updated = defaultdict(int)
            for (labels, a_power, beta), coefficient in states.items():
                # edge excluded
                updated[(labels, a_power, beta)] += coefficient
                # edge included: edge weight -1; cycle rank increases iff endpoints
                # were already connected in the current spanning subgraph quotient.
                merged, closes = _union(labels, left, right)
                updated[(merged, a_power, beta + int(closes))] -= coefficient
                transitions += 2
            states = {key: value for key, value in updated.items() if value}

        processed = set(order[: step + 1])
        forget_positions = []
        for position, port in enumerate(active):
            partner_factor = port_factor[prepared.arc_partner[port]]
            if partner_factor in processed:
                forget_positions.append(position)
        if forget_positions:
            remove = set(forget_positions)
            forgotten = defaultdict(int)
            for (labels, a_power, beta), coefficient in states.items():
                kept = tuple(label for index, label in enumerate(labels) if index not in remove)
                forgotten[(_canonical(kept), a_power, beta)] += coefficient
            states = {key: value for key, value in forgotten.items() if value}
            active = [port for index, port in enumerate(active) if index not in remove]

        max_frontier = max(max_frontier, len(active))
        max_states = max(max_states, len(states))

    if active:
        raise RuntimeError("diagram frontier did not close")
    weights = defaultdict(int)
    for (labels, a_power, beta), coefficient in states.items():
        if labels:
            raise RuntimeError("closed frontier retained partition labels")
        weights[(a_power, beta)] += coefficient

    stats.update(
        factor_count=len(order),
        crossing_count=crossing_count,
        max_frontier=max_frontier,
        max_states=max_states,
        transitions=transitions,
        terminal_terms=len(weights),
        factor_order=tuple(order),
    )
    return _q_to_laurent(weights)
