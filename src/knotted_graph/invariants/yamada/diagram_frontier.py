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


class FrontierLimitExceeded(RuntimeError):
    """Raised when an adaptive frontier run exceeds a configured safe limit."""


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
            disconnected = 1 if back == 0 and processed else 0
            return (disconnected, future - back, future, -back, len(factor_ports[node]), node)

        node = min(unprocessed, key=score)
        order.append(node)
        processed.add(node)
        unprocessed.remove(node)
    return order


def plan_diagram_frontier(prepared, *, factor_order=None):
    """Return a cheap deterministic frontier plan without polynomial work.

    ``peak_ports`` is the largest number of live port endpoints immediately
    after introducing a factor and before closing its backward arcs. It is a
    conservative structural proxy for the partition-state cost and is used only
    for performance dispatch, never for correctness.
    """
    factor_ports, port_factor, adjacency, arcs = _factor_graph(prepared)
    order = _greedy_factor_order(adjacency, factor_ports) if factor_order is None else list(factor_order)
    if sorted(order) != list(range(len(factor_ports))):
        raise ValueError("factor_order must contain every fixed vertex/crossing exactly once")

    active = []
    processed = set()
    peak_ports = 0
    max_boundary_ports = 0
    for factor in order:
        active.extend(sorted(factor_ports[factor]))
        peak_ports = max(peak_ports, len(active))
        processed.add(factor)
        active = [
            port
            for port in active
            if port_factor[prepared.arc_partner[port]] not in processed
        ]
        max_boundary_ports = max(max_boundary_ports, len(active))

    if active:
        raise RuntimeError("frontier planner did not close")
    return {
        "factor_order": tuple(order),
        "peak_ports": peak_ports,
        "max_boundary_ports": max_boundary_ports,
        "factor_count": len(order),
        "arc_count": len(arcs),
    }


def _crossing_groups(prepared, crossing_index, spin):
    ports = prepared.ordered_ports[crossing_index]
    if spin == 2:
        return (tuple(ports),)
    partner = prepared.plus_partner if spin == 0 else prepared.minus_partner
    groups = []
    seen = set()
    port_set = set(ports)
    for port in ports:
        if port in seen:
            continue
        other = partner[port]
        if other not in port_set:
            raise RuntimeError("crossing resolution partner escaped crossing")
        groups.append((port, other))
        seen.add(port)
        seen.add(other)
    if len(groups) != 2:
        raise RuntimeError("smoothing did not produce two local degree-two vertices")
    return tuple(groups)


def _apply_groups(labels, positions, groups):
    current = labels
    for group in groups:
        anchor = positions[group[0]]
        for port in group[1:]:
            current, closes = _union(current, anchor, positions[port])
            if closes:
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


def compute_diagram_frontier_laurent(
    prepared,
    *,
    factor_order=None,
    max_states=None,
    max_peak_ports=None,
    stats=None,
):
    """Return exact Laurent Yamada polynomial by connectivity-frontier DP.

    Optional limits are performance guards. Exceeding one raises
    :class:`FrontierLimitExceeded`, allowing a caller to fall back to another
    exact backend without changing the mathematical result.
    """
    if stats is None:
        stats = {}
    factor_ports, port_factor, adjacency, arcs = _factor_graph(prepared)
    vertex_count = len(prepared.vertex_ids)
    crossing_count = len(prepared.crossing_ids)
    plan = plan_diagram_frontier(prepared, factor_order=factor_order)
    order = list(plan["factor_order"])
    if max_peak_ports is not None and plan["peak_ports"] > int(max_peak_ports):
        raise FrontierLimitExceeded(
            f"planned peak frontier {plan['peak_ports']} exceeds {max_peak_ports}"
        )
    step_of = {factor: step for step, factor in enumerate(order)}

    backward_arcs = [[] for _ in order]
    for p, q, left, right in arcs:
        backward_arcs[max(step_of[left], step_of[right])].append((p, q))

    active = []
    states = {((), 0, 0): 1}
    max_frontier = 0
    max_peak_seen = 0
    max_states_seen = 1
    transitions = 0
    processed = set()

    def enforce_state_limit(stage):
        nonlocal max_states_seen
        max_states_seen = max(max_states_seen, len(states))
        if max_states is not None and len(states) > int(max_states):
            stats.update(
                plan,
                crossing_count=crossing_count,
                max_frontier=max_frontier,
                max_peak_seen=max_peak_seen,
                max_states=max_states_seen,
                transitions=transitions,
                aborted_stage=stage,
            )
            raise FrontierLimitExceeded(
                f"frontier states {len(states)} exceed {max_states} at {stage}"
            )

    for step, factor in enumerate(order):
        ports = sorted(factor_ports[factor])
        active.extend(ports)
        max_peak_seen = max(max_peak_seen, len(active))
        positions = {port: index for index, port in enumerate(active)}

        introduced = defaultdict(int)
        is_crossing = factor >= vertex_count
        if is_crossing:
            crossing = factor - vertex_count
            options = ((0, 1, 1), (1, -1, 1), (2, 0, -1))
        else:
            options = ((-1, 0, -1),)

        for (labels, a_power, beta), coefficient in states.items():
            start = max(labels, default=-1) + 1
            base_labels = _canonical(labels + tuple(range(start, start + len(ports))))
            for spin, a_delta, local_sign in options:
                groups = (
                    _crossing_groups(prepared, crossing, spin)
                    if is_crossing
                    else ((tuple(ports),) if ports else ())
                )
                new_labels = _apply_groups(base_labels, positions, groups)
                introduced[(new_labels, a_power + a_delta, beta)] += coefficient * local_sign
                transitions += 1
        states = {key: value for key, value in introduced.items() if value}
        enforce_state_limit(f"factor:{step}:introduced")

        for arc_index, (left_port, right_port) in enumerate(backward_arcs[step]):
            left = positions[left_port]
            right = positions[right_port]
            updated = defaultdict(int)
            for (labels, a_power, beta), coefficient in states.items():
                updated[(labels, a_power, beta)] += coefficient
                merged, closes = _union(labels, left, right)
                updated[(merged, a_power, beta + int(closes))] -= coefficient
                transitions += 2
            states = {key: value for key, value in updated.items() if value}
            enforce_state_limit(f"factor:{step}:arc:{arc_index}")

        processed.add(factor)
        forget_positions = [
            position
            for position, port in enumerate(active)
            if port_factor[prepared.arc_partner[port]] in processed
        ]
        if forget_positions:
            remove = set(forget_positions)
            forgotten = defaultdict(int)
            for (labels, a_power, beta), coefficient in states.items():
                kept = tuple(label for index, label in enumerate(labels) if index not in remove)
                forgotten[(_canonical(kept), a_power, beta)] += coefficient
            states = {key: value for key, value in forgotten.items() if value}
            active = [port for index, port in enumerate(active) if index not in remove]
            enforce_state_limit(f"factor:{step}:forgotten")

        max_frontier = max(max_frontier, len(active))

    if active:
        raise RuntimeError("diagram frontier did not close")
    weights = defaultdict(int)
    for (labels, a_power, beta), coefficient in states.items():
        if labels:
            raise RuntimeError("closed frontier retained partition labels")
        weights[(a_power, beta)] += coefficient

    stats.update(
        plan,
        crossing_count=crossing_count,
        max_frontier=max_frontier,
        max_peak_seen=max_peak_seen,
        max_states=max_states_seen,
        transitions=transitions,
        terminal_terms=len(weights),
    )
    return _q_to_laurent(weights)
