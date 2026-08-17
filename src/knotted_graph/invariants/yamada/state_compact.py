"""Prepared compact crossing-state construction.

This module preserves the existing three-state Yamada resolution exactly while
avoiding creation of a NetworkX MultiGraph for every one of the 3^c states.
Static arc connectivity and crossing port order are compiled once per diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

from knotted_graph.projection.geom import Arc, Crossing, Vertex

from .compact import CompactGraph

Port = tuple[int, str]
Terminal = tuple[str, int]


@dataclass(slots=True)
class PreparedCompactStateBuilder:
    vertices: list[Vertex]
    crossings: list[Crossing]
    arcs: list[Arc]
    ordered_ports: tuple[tuple[Port, Port, Port, Port], ...]
    crossing_index_by_id: dict[int, int]
    arc_partner: dict[Port, Port]
    fixed_terminal: dict[Port, Terminal]
    crossing_terminal: dict[Port, int]
    all_ports: tuple[Port, ...]

    @classmethod
    def prepare(cls, vertices, crossings, arcs, ordered_port_fn):
        vertices = list(vertices)
        crossings = list(crossings)
        arcs = list(arcs)
        arcs_by_id = {arc.id: arc for arc in arcs}
        crossing_index_by_id = {
            crossing.id: index for index, crossing in enumerate(crossings)
        }

        ordered_ports = tuple(
            tuple(ordered_port_fn(crossing, arcs_by_id))
            for crossing in crossings
        )

        arc_partner: dict[Port, Port] = {}
        fixed_terminal: dict[Port, Terminal] = {}
        crossing_terminal: dict[Port, int] = {}
        ports: list[Port] = []

        for arc in arcs:
            start = (arc.id, "s")
            end = (arc.id, "e")
            ports.extend((start, end))
            arc_partner[start] = end
            arc_partner[end] = start

            if arc.start_type == "v":
                fixed_terminal[start] = ("v", int(arc.start_id))
            else:
                crossing_terminal[start] = crossing_index_by_id[arc.start_id]

            if arc.end_type == "v":
                fixed_terminal[end] = ("v", int(arc.end_id))
            else:
                crossing_terminal[end] = crossing_index_by_id[arc.end_id]

        return cls(
            vertices=vertices,
            crossings=crossings,
            arcs=arcs,
            ordered_ports=ordered_ports,  # type: ignore[arg-type]
            crossing_index_by_id=crossing_index_by_id,
            arc_partner=arc_partner,
            fixed_terminal=fixed_terminal,
            crossing_terminal=crossing_terminal,
            all_ports=tuple(ports),
        )

    def build(
        self,
        state: tuple[int, ...],
        *,
        plus_pairs=((0, 3), (1, 2)),
        minus_pairs=((0, 1), (2, 3)),
    ) -> CompactGraph:
        if len(state) != len(self.crossings):
            raise ValueError("State length must match the number of crossings.")
        if any(spin not in (0, 1, 2) for spin in state):
            raise ValueError("Invalid spin configuration.")

        # The old builder always creates every original graph vertex and every
        # unresolved crossing as a terminal node, including isolated vertices.
        terminals: list[Terminal] = [
            ("v", int(vertex.id)) for vertex in self.vertices
        ]
        terminals.extend(
            ("x", int(crossing.id))
            for crossing, spin in zip(self.crossings, state)
            if spin == 2
        )
        terminal_index = {terminal: i for i, terminal in enumerate(terminals)}

        # Resolution partner at a crossing. Arc partners are static and were
        # compiled once in ``prepare``.
        resolution_partner: dict[Port, Port] = {}
        for crossing_index, spin in enumerate(state):
            if spin == 2:
                continue
            ports = self.ordered_ports[crossing_index]
            pairs = plus_pairs if spin == 0 else minus_pairs
            for a, b in pairs:
                pa, pb = ports[a], ports[b]
                resolution_partner[pa] = pb
                resolution_partner[pb] = pa

        def terminal_for(port: Port) -> Terminal | None:
            fixed = self.fixed_terminal.get(port)
            if fixed is not None:
                return fixed
            crossing_index = self.crossing_terminal.get(port)
            if crossing_index is not None and state[crossing_index] == 2:
                return ("x", int(self.crossings[crossing_index].id))
            return None

        # Each path alternates arc edge / resolution edge. Mark ports whose arc
        # edge has been consumed; every physical resolved graph edge contains at
        # least one arc edge, so this counts each path once.
        visited_arc_ports: set[Port] = set()
        graph_edges: list[tuple[int, int]] = []

        for start_port in self.all_ports:
            start_terminal = terminal_for(start_port)
            if start_terminal is None or start_port in visited_arc_ports:
                continue

            current = start_port
            while True:
                other = self.arc_partner[current]
                visited_arc_ports.add(current)
                visited_arc_ports.add(other)

                end_terminal = terminal_for(other)
                if end_terminal is not None:
                    graph_edges.append(
                        (terminal_index[start_terminal], terminal_index[end_terminal])
                    )
                    break

                partner = resolution_partner.get(other)
                if partner is None:
                    raise RuntimeError(
                        "Resolved crossing port has neither terminal nor resolution partner."
                    )
                current = partner

        # Remaining ports belong to closed components with no graph/crossing
        # terminal. The reference implementation represents each such component
        # by one dummy vertex carrying one loop.
        closed_loop_count = 0
        for start_port in self.all_ports:
            if start_port in visited_arc_ports:
                continue

            closed_loop_count += 1
            current = start_port
            while True:
                other = self.arc_partner[current]
                visited_arc_ports.add(current)
                visited_arc_ports.add(other)
                partner = resolution_partner.get(other)
                if partner is None:
                    # A fixed/unresolved terminal would have been reached from
                    # the terminal traversal above, so this indicates malformed
                    # diagram incidence rather than a legitimate closed loop.
                    raise RuntimeError("Malformed terminal-free resolved component.")
                current = partner
                if current in visited_arc_ports:
                    break

        node_count = len(terminals) + closed_loop_count
        matrix = [[0] * node_count for _ in range(node_count)]

        for i, j in graph_edges:
            matrix[i][j] += 1
            if i != j:
                matrix[j][i] += 1

        for loop_index in range(closed_loop_count):
            i = len(terminals) + loop_index
            matrix[i][i] += 1

        return CompactGraph(tuple(tuple(row) for row in matrix))
