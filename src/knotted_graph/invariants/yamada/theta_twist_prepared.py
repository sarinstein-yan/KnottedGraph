"""Prepared-diagram recognizer for the closed-form Theta(n) Yamada family."""

from __future__ import annotations

from .theta_twist import _theta_formula_laurent


def _straight_through_map(prepared):
    opposite = [-1] * len(prepared.arc_partner)
    for ports in prepared.ordered_ports:
        if len(ports) != 4 or len(set(ports)) != 4:
            return None
        opposite[ports[0]] = ports[2]
        opposite[ports[2]] = ports[0]
        opposite[ports[1]] = ports[3]
        opposite[ports[3]] = ports[1]
    return tuple(opposite)


def _trace_physical_edge(prepared, start_port: int, target_vertex: int, opposite):
    """Follow one original spatial-graph edge straight through every crossing."""
    arc_partner = prepared.arc_partner
    fixed = prepared.fixed_terminal_index
    crossing_for_port = prepared.crossing_for_port
    ordered = prepared.ordered_ports

    current = start_port
    seen_ports = set()
    crossing_order = []
    directions = {}
    while True:
        if current in seen_ports:
            return None
        seen_ports.add(current)
        remote = arc_partner[current]
        if remote < 0 or remote >= len(arc_partner):
            return None
        if remote in seen_ports:
            return None
        seen_ports.add(remote)

        terminal = fixed[remote]
        if terminal >= 0:
            if terminal != target_vertex:
                return None
            return tuple(crossing_order), directions, frozenset(seen_ports)

        crossing_index = crossing_for_port[remote]
        if crossing_index < 0 or crossing_index >= len(ordered):
            return None
        if crossing_index in directions:
            # Reject self-crossings of one physical edge. The closed form is for
            # two distinct strands crossing each other, not arbitrary knots on
            # an individual theta edge.
            return None
        through = opposite[remote]
        if through < 0 or crossing_for_port[through] != crossing_index:
            return None
        crossing_order.append(crossing_index)
        directions[crossing_index] = (remote, through)
        current = through


def _crossing_sign(prepared, crossing_index: int, directions_a, directions_b):
    ports = prepared.ordered_ports[crossing_index]
    position = {port: index for index, port in enumerate(ports)}
    outgoing_positions = []
    for directions in (directions_a, directions_b):
        pair = directions.get(crossing_index)
        if pair is None:
            return None
        incoming, outgoing = pair
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


def certified_prepared_theta_twist_laurent(prepared):
    """Return the exact Theta(n) closed form iff the prepared diagram certifies it.

    The certificate is intrinsic to the prepared combinatorial diagram and does
    not rely on benchmark labels or geometry:

    * two graph vertices, each with exactly three incident physical edges;
    * tracing straight through crossings recovers three vertex-to-vertex edges;
    * one edge is crossing-free;
    * each of the other two visits every crossing exactly once;
    * their crossing orders are reversed (the canonical two-braid closure);
    * all crossings have the same oriented sign.

    Any failure returns ``None`` so the general exact evaluator remains the
    authoritative fallback.
    """
    n = len(prepared.crossing_ids)
    if n < 1 or n % 2 == 0 or len(prepared.vertex_ids) != 2:
        return None

    fixed = prepared.fixed_terminal_index
    terminal_ports = {
        vertex: [port for port, terminal in enumerate(fixed) if terminal == vertex]
        for vertex in (0, 1)
    }
    if any(len(terminal_ports[vertex]) != 3 for vertex in (0, 1)):
        return None
    if any(terminal not in {-1, 0, 1} for terminal in fixed):
        return None

    opposite = _straight_through_map(prepared)
    if opposite is None:
        return None

    paths = []
    used_start_ports = set()
    for start_port in terminal_ports[0]:
        traced = _trace_physical_edge(prepared, start_port, 1, opposite)
        if traced is None:
            return None
        order, directions, seen_ports = traced
        paths.append((order, directions, seen_ports))
        used_start_ports.add(start_port)
    if len(used_start_ports) != 3:
        return None

    # Physical paths must partition all arc/crossing ports. They are allowed to
    # meet only at the two graph vertices; individual integer ports are unique.
    all_seen = set()
    for _order, _directions, seen_ports in paths:
        if all_seen.intersection(seen_ports):
            return None
        all_seen.update(seen_ports)
    if all_seen != set(range(len(prepared.arc_partner))):
        return None

    exterior = [path for path in paths if not path[0]]
    braided = [path for path in paths if path[0]]
    if len(exterior) != 1 or len(braided) != 2:
        return None

    order_a, directions_a, _ = braided[0]
    order_b, directions_b, _ = braided[1]
    expected = set(range(n))
    if len(order_a) != n or len(order_b) != n:
        return None
    if set(order_a) != expected or set(order_b) != expected:
        return None
    if order_b != tuple(reversed(order_a)):
        return None

    signs = []
    for crossing_index in range(n):
        sign = _crossing_sign(
            prepared,
            crossing_index,
            directions_a,
            directions_b,
        )
        if sign is None:
            return None
        signs.append(sign)
    if len(set(signs)) != 1:
        return None

    # Dobrynin--Vesnin's published orientation is the positive crossing
    # convention used by PreparedCompactStateBuilder. The opposite common sign
    # is the mirror family and follows by A <-> A^-1.
    return _theta_formula_laurent(n, mirror=signs[0] < 0)
