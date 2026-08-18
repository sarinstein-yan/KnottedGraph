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

    def _find_reidemeister_ii_pair(self):
        """Return one conservatively recognized Reidemeister-II crossing pair.

        A removable bigon consists of two crossings joined by exactly two
        physical arcs. Those arcs occupy adjacent cyclic positions at both
        crossings and preserve over/under parity between their endpoints.
        """
        arc_partner = self.arc_partner

        for first in range(len(self.ordered_ports)):
            first_ports = self.ordered_ports[first]
            for second in range(first):
                second_ports = self.ordered_ports[second]
                second_position = {port: i for i, port in enumerate(second_ports)}
                shared = []
                for first_position, first_port in enumerate(first_ports):
                    partner = arc_partner[first_port]
                    if partner not in second_position:
                        continue
                    shared.append(
                        (
                            first_position,
                            second_position[partner],
                            first_port,
                            partner,
                        )
                    )

                if len(shared) != 2:
                    continue

                first_positions = [entry[0] for entry in shared]
                second_positions = [entry[1] for entry in shared]
                if (first_positions[0] - first_positions[1]) % 4 not in (1, 3):
                    continue
                if (second_positions[0] - second_positions[1]) % 4 not in (1, 3):
                    continue
                if any((a % 2) != (b % 2) for a, b, _, _ in shared):
                    continue

                removed = set(first_ports) | set(second_ports)
                splices = []
                valid = True
                for first_position, second_position_index, _, _ in shared:
                    first_external = first_ports[(first_position + 2) % 4]
                    second_external = second_ports[(second_position_index + 2) % 4]
                    remote_first = arc_partner[first_external]
                    remote_second = arc_partner[second_external]
                    if (
                        remote_first in removed
                        or remote_second in removed
                        or remote_first == remote_second
                    ):
                        valid = False
                        break
                    splices.append((remote_first, remote_second))
                if valid and len({port for pair in splices for port in pair}) == 4:
                    return first, second, tuple(splices)
        return None

    def _remove_reidemeister_ii_pair(self, first, second, splices):
        removed_crossings = {first, second}
        removed_ports = set(self.ordered_ports[first]) | set(self.ordered_ports[second])
        active_ports = [
            port for port in range(len(self.arc_partner)) if port not in removed_ports
        ]
        old_to_new = {old: new for new, old in enumerate(active_ports)}

        partner = list(self.arc_partner)
        for left, right in splices:
            partner[left] = right
            partner[right] = left

        new_arc_partner = []
        for old in active_ports:
            old_partner = partner[old]
            if old_partner not in old_to_new:
                raise RuntimeError("RII reduction left an arc attached to a removed port.")
            new_arc_partner.append(old_to_new[old_partner])

        surviving_crossings = [
            index
            for index in range(len(self.crossing_ids))
            if index not in removed_crossings
        ]
        crossing_remap = {
            old: new for new, old in enumerate(surviving_crossings)
        }
        new_crossing_for_port = []
        for old_port in active_ports:
            old_crossing = self.crossing_for_port[old_port]
            if old_crossing < 0:
                new_crossing_for_port.append(-1)
            else:
                if old_crossing in removed_crossings:
                    raise RuntimeError("RII reduction retained a removed crossing port.")
                new_crossing_for_port.append(crossing_remap[old_crossing])

        new_ordered_ports = tuple(
            tuple(old_to_new[port] for port in self.ordered_ports[index])
            for index in surviving_crossings
        )
        port_count = len(active_ports)
        plus_partner = [-1] * port_count
        minus_partner = [-1] * port_count
        for ports in new_ordered_ports:
            for a, b in _PLUS_PAIRS:
                pa, pb = ports[a], ports[b]
                plus_partner[pa] = pb
                plus_partner[pb] = pa
            for a, b in _MINUS_PAIRS:
                pa, pb = ports[a], ports[b]
                minus_partner[pa] = pb
                minus_partner[pb] = pa

        return PreparedCompactStateBuilder(
            vertex_ids=self.vertex_ids,
            crossing_ids=tuple(self.crossing_ids[index] for index in surviving_crossings),
            ordered_ports=new_ordered_ports,
            arc_partner=tuple(new_arc_partner),
            fixed_terminal_index=tuple(
                self.fixed_terminal_index[port] for port in active_ports
            ),
            crossing_for_port=tuple(new_crossing_for_port),
            plus_partner=tuple(plus_partner),
            minus_partner=tuple(minus_partner),
        )

    def reduce_reidemeister_ii(self):
        """Cancel all conservatively detectable Reidemeister-II bigons.

        The returned builder represents the same regular-isotopy class while
        containing two fewer crossings per accepted move. The original builder
        is not mutated.
        """
        current = self
        count = 0
        while True:
            candidate = current._find_reidemeister_ii_pair()
            if candidate is None:
                return current, count
            current = current._remove_reidemeister_ii_pair(*candidate)
            count += 1

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
