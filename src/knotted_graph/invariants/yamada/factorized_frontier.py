"""Exact low-arity factorization for the Yamada connectivity state sum.

A fixed graph vertex is an equality constraint on all incident ports with one
weight -1.  High arity is therefore representational, not mathematical.  We
factor each such equality into a deterministic chain of arity <=3 equality
factors joined by identity wires.  Exactly one factor carries the original -1
weight; auxiliary equality factors carry +1.  Identity wires only identify
connectivity variables and are *not* graph edges, so they never create the
Yamada edge weight or cycle factor.

This is an exact reparameterization of the same state sum and contains no
runtime-, family-, crossing-count-, or benchmark-dependent dispatch.
"""

from __future__ import annotations

from collections import defaultdict

from .diagram_frontier import _greedy_factor_order

FACTOR_EQUALITY_NEG = 0
FACTOR_EQUALITY_POS = 1
FACTOR_CROSSING = 2
WIRE_PHYSICAL = 0
WIRE_IDENTITY = 1


def build_factorized_frontier(prepared):
    vertex_count = len(prepared.vertex_ids)
    crossing_count = len(prepared.crossing_ids)
    original_port_count = len(prepared.arc_partner)

    vertex_ports = [[] for _ in range(vertex_count)]
    crossing_ports = [[] for _ in range(crossing_count)]
    for port in range(original_port_count):
        fixed = int(prepared.fixed_terminal_index[port])
        crossing = int(prepared.crossing_for_port[port])
        if fixed >= 0:
            vertex_ports[fixed].append(port)
        elif crossing >= 0:
            crossing_ports[crossing].append(port)
        else:
            raise RuntimeError("prepared Yamada port belongs to no factor")

    def neighbor_key(port):
        partner = int(prepared.arc_partner[port])
        fixed = int(prepared.fixed_terminal_index[partner])
        if fixed >= 0:
            return (0, fixed, partner)
        return (1, int(prepared.crossing_for_port[partner]), partner)

    wire_partner = [int(value) for value in prepared.arc_partner]
    wire_type = [WIRE_PHYSICAL] * original_port_count
    port_factor = [-1] * original_port_count
    factor_types = []
    factor_ports = []

    # Degree-zero fixed vertices still contribute their single -1 factor.
    for ports in vertex_ports:
        ordered = sorted(ports, key=neighbor_key)
        if not ordered:
            factor_types.append(FACTOR_EQUALITY_NEG)
            factor_ports.append([])
            continue

        previous_right = None
        for segment_index, original_port in enumerate(ordered):
            factor = len(factor_types)
            local_ports = [original_port]
            port_factor[original_port] = factor

            if previous_right is not None:
                left = len(wire_partner)
                wire_partner.append(previous_right)
                wire_type.append(WIRE_IDENTITY)
                wire_partner[previous_right] = left
                wire_type[previous_right] = WIRE_IDENTITY
                port_factor.append(factor)
                local_ports.append(left)

            if segment_index + 1 < len(ordered):
                right = len(wire_partner)
                wire_partner.append(-1)
                wire_type.append(WIRE_IDENTITY)
                port_factor.append(factor)
                local_ports.append(right)
                previous_right = right
            else:
                previous_right = None

            factor_types.append(
                FACTOR_EQUALITY_NEG if segment_index == 0 else FACTOR_EQUALITY_POS
            )
            factor_ports.append(local_ports)

    crossing_factor_by_index = []
    for ports in crossing_ports:
        if len(ports) != 4:
            raise RuntimeError("prepared crossing must have exactly four ports")
        factor = len(factor_types)
        factor_types.append(FACTOR_CROSSING)
        local_ports = sorted(ports)
        factor_ports.append(local_ports)
        crossing_factor_by_index.append(factor)
        for port in local_ports:
            port_factor[port] = factor

    if any(value < 0 for value in port_factor):
        raise RuntimeError("factorized Yamada port has no owner")
    if any(value < 0 for value in wire_partner):
        raise RuntimeError("unpaired factorization identity wire")

    adjacency = [defaultdict(int) for _ in factor_types]
    for port, partner in enumerate(wire_partner):
        if port >= partner:
            continue
        left = port_factor[port]
        right = port_factor[partner]
        if left != right:
            adjacency[left][right] += 1
            adjacency[right][left] += 1
    factor_order = _greedy_factor_order(adjacency, factor_ports)

    # Crossing pairing tables extend across virtual identity ports with -1.
    plus_partner = [-1] * len(wire_partner)
    minus_partner = [-1] * len(wire_partner)
    for crossing, ports in enumerate(prepared.ordered_ports):
        for port in ports:
            plus_partner[port] = int(prepared.plus_partner[port])
            minus_partner[port] = int(prepared.minus_partner[port])

    return {
        "factor_types": tuple(factor_types),
        "port_factor": tuple(port_factor),
        "wire_partner": tuple(wire_partner),
        "wire_type": tuple(wire_type),
        "plus_partner": tuple(plus_partner),
        "minus_partner": tuple(minus_partner),
        "factor_order": tuple(factor_order),
        "original_port_count": original_port_count,
        "crossing_factor_by_index": tuple(crossing_factor_by_index),
    }
