"""Closed-form Yamada evaluation for the certified Dobrynin--Vesnin theta family.

Dobrynin and Vesnin, *The Yamada polynomial for graphs, embedded knot-wise into
three-dimensional space* (Vychisl. Sistemy 155, 1996, 37--86), Theorem 2,
consider the spatial family Theta(n) obtained from the canonical two-strand
torus-knot diagram T(2,n) (n odd) by adding one exterior theta edge. They prove

    R(Theta(n)) = (A^2+A+1+A^-1+A^-2) A^n
                  - (A+A^-1) A^(-2n)
                  - (A^2+1+A^-2) (-1)^n A^(-n).

This module recognizes that diagram combinatorially from the original-edge
provenance retained by ``Arc.edge_key``.  The recognizer is intentionally
conservative: failure of any certificate returns ``None`` and production falls
back to the general exact evaluator.  The mirror family is handled by the exact
identity R(mirror(G); A) = R(G; A^-1).
"""

from __future__ import annotations

from collections import defaultdict

from .fast import Laurent


def _node_at(arc, side: str):
    if side == "s":
        return str(arc.start_type), int(arc.start_id)
    return str(arc.end_type), int(arc.end_id)


def _port_at_node(arc, node):
    if _node_at(arc, "s") == node:
        return arc.id, "s"
    if _node_at(arc, "e") == node:
        return arc.id, "e"
    raise ValueError("arc is not incident to requested node")


def _trace_original_edge(edge_arcs, start_vertex: int, end_vertex: int):
    """Trace one original edge from a common theta endpoint to the other.

    Returns the crossing order and, at each crossing, the incoming/outgoing
    half-edge ports with respect to that common orientation.  A valid original
    edge is a simple chain in the projection graph: each endpoint has degree one
    in the edge group and every encountered crossing degree two.
    """
    adjacency = defaultdict(list)
    by_id = {}
    for arc in edge_arcs:
        by_id[arc.id] = arc
        adjacency[_node_at(arc, "s")].append(arc.id)
        adjacency[_node_at(arc, "e")].append(arc.id)

    start = ("v", int(start_vertex))
    target = ("v", int(end_vertex))
    if len(adjacency[start]) != 1 or len(adjacency[target]) != 1:
        return None
    for node, incident in adjacency.items():
        if node in {start, target}:
            continue
        if node[0] != "x" or len(incident) != 2:
            return None

    previous_arc = None
    current = start
    order = []
    directions = {}
    seen_arcs = set()

    while current != target:
        candidates = [arc_id for arc_id in adjacency[current] if arc_id != previous_arc]
        if len(candidates) != 1:
            return None
        arc_id = candidates[0]
        if arc_id in seen_arcs:
            return None
        seen_arcs.add(arc_id)
        arc = by_id[arc_id]
        start_node = _node_at(arc, "s")
        end_node = _node_at(arc, "e")
        nxt = end_node if current == start_node else start_node

        if nxt[0] == "x":
            crossing_id = nxt[1]
            incoming_port = _port_at_node(arc, nxt)
            next_candidates = [
                candidate
                for candidate in adjacency[nxt]
                if candidate != arc_id
            ]
            if len(next_candidates) != 1:
                return None
            outgoing_arc = by_id[next_candidates[0]]
            outgoing_port = _port_at_node(outgoing_arc, nxt)
            if crossing_id in directions:
                return None
            order.append(crossing_id)
            directions[crossing_id] = (incoming_port, outgoing_port)

        previous_arc = arc_id
        current = nxt

    if len(seen_arcs) != len(edge_arcs):
        return None
    return tuple(order), directions


def _endpoint_set(edge_arcs):
    endpoints = []
    for arc in edge_arcs:
        if arc.start_type == "v":
            endpoints.append(int(arc.start_id))
        if arc.end_type == "v":
            endpoints.append(int(arc.end_id))
    return tuple(sorted(endpoints))


def _crossing_sign(crossing, arcs_by_id, directions_by_edge, ordered_port_fn):
    """Return the oriented crossing sign from the certified two-edge traversal.

    ``ordered_port_fn`` gives the CCW port cycle with positions 0/2 on the
    over-strand.  Both physical theta edges are oriented from the same graph
    vertex to the other.  The sign is determined by whether the outgoing under
    tangent is the next or previous CCW ray from the outgoing over tangent.
    """
    ordered = list(ordered_port_fn(crossing, arcs_by_id))
    if len(ordered) != 4 or len(set(ordered)) != 4:
        return None
    position = {port: index for index, port in enumerate(ordered)}

    outgoing_positions = []
    for directions in directions_by_edge:
        ports = directions.get(int(crossing.id))
        if ports is None:
            return None
        incoming, outgoing = ports
        if incoming not in position or outgoing not in position:
            return None
        i = position[incoming]
        o = position[outgoing]
        if (o - i) % 4 != 2:
            return None
        outgoing_positions.append(o)

    if {pos % 2 for pos in outgoing_positions} != {0, 1}:
        return None
    over_out = next(pos for pos in outgoing_positions if pos % 2 == 0)
    under_out = next(pos for pos in outgoing_positions if pos % 2 == 1)
    delta = (under_out - over_out) % 4
    if delta == 1:
        return 1
    if delta == 3:
        return -1
    return None


def _theta_formula_laurent(n: int, *, mirror: bool) -> Laurent:
    """Dobrynin--Vesnin Theorem 2 in the package's Laurent representation."""
    coefficients: dict[int, int] = {}

    def put(exponent: int, coefficient: int) -> None:
        if mirror:
            exponent = -exponent
        coefficients[exponent] = coefficients.get(exponent, 0) + coefficient

    for offset in (2, 1, 0, -1, -2):
        put(n + offset, 1)
    put(-2 * n + 1, -1)
    put(-2 * n - 1, -1)
    third = -((-1) ** n)
    for offset in (2, 0, -2):
        put(-n + offset, third)

    return tuple(sorted((power, coeff) for power, coeff in coefficients.items() if coeff))


def certified_theta_twist_laurent(vertices, crossings, arcs, ordered_port_fn):
    """Return a closed-form Laurent value iff the diagram is certified Theta(n).

    Certification requires:

    * exactly two degree-three graph vertices and three original edge keys;
    * one crossing-free exterior edge joining the two vertices;
    * two remaining original edges each traverse every crossing exactly once;
    * the crossing order on those two edges is reversed, as in a 2-braid closure;
    * every crossing is between those two edges and all oriented crossing signs
      are equal.

    These conditions encode the canonical odd-n Dobrynin--Vesnin theta family.
    Any uncertainty returns ``None`` rather than risking a false fast-path hit.
    """
    vertices = list(vertices)
    crossings = list(crossings)
    arcs = list(arcs)
    n = len(crossings)
    if n < 1 or n % 2 == 0 or len(vertices) != 2:
        return None
    if any(len(vertex.incident_arcs) != 3 for vertex in vertices):
        return None

    groups = defaultdict(list)
    for arc in arcs:
        groups[arc.edge_key].append(arc)
    if len(groups) != 3:
        return None

    vertex_ids = tuple(sorted(int(vertex.id) for vertex in vertices))
    if any(_endpoint_set(group) != vertex_ids for group in groups.values()):
        return None

    traced = []
    for edge_key, group in groups.items():
        result = _trace_original_edge(group, vertex_ids[0], vertex_ids[1])
        if result is None:
            return None
        order, directions = result
        traced.append((edge_key, group, order, directions))

    exterior = [entry for entry in traced if not entry[2]]
    braided = [entry for entry in traced if entry[2]]
    if len(exterior) != 1 or len(braided) != 2:
        return None
    if len(exterior[0][1]) != 1:
        return None

    all_crossings = set(int(crossing.id) for crossing in crossings)
    first_order = braided[0][2]
    second_order = braided[1][2]
    if len(first_order) != n or len(second_order) != n:
        return None
    if set(first_order) != all_crossings or set(second_order) != all_crossings:
        return None
    if second_order != tuple(reversed(first_order)):
        return None

    braided_keys = {braided[0][0], braided[1][0]}
    arcs_by_id = {arc.id: arc for arc in arcs}
    for crossing in crossings:
        incident = [arcs_by_id[arc_id].edge_key for arc_id, _ in crossing.incident_arcs]
        if set(incident) != braided_keys:
            return None
        if any(incident.count(edge_key) != 2 for edge_key in braided_keys):
            return None

    directions_by_edge = (braided[0][3], braided[1][3])
    signs = []
    for crossing in crossings:
        sign = _crossing_sign(
            crossing,
            arcs_by_id,
            directions_by_edge,
            ordered_port_fn,
        )
        if sign is None:
            return None
        signs.append(sign)
    if len(set(signs)) != 1:
        return None

    # The positive-sign convention produced by _ordered_crossing_ports matches
    # the published Theta(n) formula; reversing all crossings gives the mirror
    # and therefore A -> A^-1. This mapping is regression-checked against the
    # legacy exhaustive state sum for both chiralities.
    mirror = signs[0] < 0
    return _theta_formula_laurent(n, mirror=mirror)
