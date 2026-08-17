"""Prepared compact crossing-state construction.

This module preserves the existing three-state Yamada resolution exactly while
avoiding creation of a NetworkX MultiGraph for every one of the 3^c states.
Static arc connectivity and crossing port order are compiled once per diagram.
The hot state loop uses dense integer port tables and bytearrays rather than
Python tuple/dict/set lookups.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compact import CompactGraph

Port = tuple[int, str]

_PLUS_PAIRS = ((0, 3), (1, 2))
_MINUS_PAIRS = ((0, 1), (2, 3))


@dataclass(slots=True)
class PreparedCompactStateBuilder:
    vertex_ids: tuple[int, ...]
    crossing_ids: tuple[int, ...]
    ordered_ports: tuple[tuple[int, int, int, int], ...]
    arc_partner: tuple[int, ...]
    fixed_terminal_index: tuple[int, ...]
    crossing_for_port: tuple[int, ...]
    plus_partner: tuple[int, ...]
    minus_partner: tuple[int, ...]

    @classmethod
    def prepare(cls, vertices, crossings, arcs, ordered_port_fn):
        vertices = list(vertices)
        crossings = list(crossings)
        arcs = list(arcs)

        vertex_ids = tuple(int(vertex.id) for vertex in vertices)
        crossing_ids = tuple(int(crossing.id) for crossing in crossings)
        vertex_index_by_id = {vertex_id: i for i, vertex_id in enumerate(vertex_ids)}
        crossing_index_by_id = {
            crossing_id: i for i, crossing_id in enumerate(crossing_ids)
        }
        arcs_by_id = {arc.id: arc for arc in arcs}

        # Each physical arc owns exactly two integer ports. Keeping the external
        # (arc_id, side) convention only during preparation removes tuple hashing
        # and dictionary access from every one of the 3^c state builds.
        port_index: dict[Port, int] = {}
        port_count = 2 * len(arcs)
        arc_partner = [-1] * port_count
        fixed_terminal_index = [-1] * port_count
        crossing_for_port = [-1] * port_count

        for arc_index, arc in enumerate(arcs):
            start = 2 * arc_index
            end = start + 1
            port_index[(arc.id, "s")] = start
            port_index[(arc.id, "e")] = end
            arc_partner[start] = end
            arc_partner[end] = start

            if arc.start_type == "v":
                fixed_terminal_index[start] = vertex_index_by_id[int(arc.start_id)]
            else:
                crossing_for_port[start] = crossing_index_by_id[int(arc.start_id)]

            if arc.end_type == "v":
                fixed_terminal_index[end] = vertex_index_by_id[int(arc.end_id)]
            else:
                crossing_for_port[end] = crossing_index_by_id[int(arc.end_id)]

        ordered_ports = tuple(
            tuple(port_index[port] for port in ordered_port_fn(crossing, arcs_by_id))
            for crossing in crossings
        )

        plus_partner = [-1] * port_count
        minus_partner = [-1] * port_count
        for ports in ordered_ports:
            for a, b in _PLUS_PAIRS:
                pa, pb = ports[a], ports[b]
                plus_partner[pa] = pb
                plus_partner[pb] = pa
            for a, b in _MINUS_PAIRS:
                pa, pb = ports[a], ports[b]
                minus_partner[pa] = pb
                minus_partner[pb] = pa

        return cls(
            vertex_ids=vertex_ids,
            crossing_ids=crossing_ids,
            ordered_ports=ordered_ports,  # type: ignore[arg-type]
            arc_partner=tuple(arc_partner),
            fixed_terminal_index=tuple(fixed_terminal_index),
            crossing_for_port=tuple(crossing_for_port),
            plus_partner=tuple(plus_partner),
            minus_partner=tuple(minus_partner),
        )

    def build(self, state: tuple[int, ...]) -> CompactGraph:
        crossing_count = len(self.crossing_ids)
        if len(state) != crossing_count:
            raise ValueError("State length must match the number of crossings.")

        # Original graph vertices are always present. Unresolved crossings are
        # appended in crossing order, exactly matching the reference builder.
        crossing_terminal_index = [-1] * crossing_count
        node_count = len(self.vertex_ids)
        for crossing_index, spin in enumerate(state):
            if spin == 2:
                crossing_terminal_index[crossing_index] = node_count
                node_count += 1
            elif spin not in (0, 1):
                raise ValueError("Invalid spin configuration.")

        fixed_terminal = self.fixed_terminal_index
        crossing_for_port = self.crossing_for_port
        arc_partner = self.arc_partner
        plus_partner = self.plus_partner
        minus_partner = self.minus_partner
        port_count = len(arc_partner)

        # Byte storage is materially cheaper than a set of tuple ports and is
        # sufficient because each physical arc edge is consumed at most once.
        visited = bytearray(port_count)
        graph_edges: list[tuple[int, int]] = []

        def terminal_index(port: int) -> int:
            fixed = fixed_terminal[port]
            if fixed >= 0:
                return fixed
            crossing_index = crossing_for_port[port]
            if crossing_index >= 0 and state[crossing_index] == 2:
                return crossing_terminal_index[crossing_index]
            return -1

        # Trace all terminal-to-terminal resolved graph edges.
        for start_port in range(port_count):
            start_terminal = terminal_index(start_port)
            if start_terminal < 0 or visited[start_port]:
                continue

            current = start_port
            while True:
                other = arc_partner[current]
                visited[current] = 1
                visited[other] = 1

                end_terminal = terminal_index(other)
                if end_terminal >= 0:
                    graph_edges.append((start_terminal, end_terminal))
                    break

                crossing_index = crossing_for_port[other]
                if crossing_index < 0:
                    raise RuntimeError(
                        "Resolved crossing port has neither terminal nor resolution partner."
                    )
                spin = state[crossing_index]
                if spin == 0:
                    current = plus_partner[other]
                elif spin == 1:
                    current = minus_partner[other]
                else:
                    raise RuntimeError(
                        "Unresolved crossing should have been detected as a terminal."
                    )
                if current < 0:
                    raise RuntimeError("Malformed crossing resolution table.")

        # Remaining unvisited arc ports form terminal-free closed components.
        # The reference representation uses one dummy vertex with one loop for
        # each such component.
        closed_loop_count = 0
        for start_port in range(port_count):
            if visited[start_port]:
                continue

            closed_loop_count += 1
            current = start_port
            while True:
                other = arc_partner[current]
                visited[current] = 1
                visited[other] = 1

                crossing_index = crossing_for_port[other]
                if crossing_index < 0:
                    raise RuntimeError("Malformed terminal-free resolved component.")
                spin = state[crossing_index]
                if spin == 0:
                    current = plus_partner[other]
                elif spin == 1:
                    current = minus_partner[other]
                else:
                    raise RuntimeError("Malformed terminal-free resolved component.")

                if current < 0:
                    raise RuntimeError("Malformed crossing resolution table.")
                if visited[current]:
                    break

        total_nodes = node_count + closed_loop_count
        matrix = [[0] * total_nodes for _ in range(total_nodes)]

        for i, j in graph_edges:
            matrix[i][j] += 1
            if i != j:
                matrix[j][i] += 1

        for loop_index in range(closed_loop_count):
            i = node_count + loop_index
            matrix[i][i] = 1

        return CompactGraph(tuple(tuple(row) for row in matrix))
