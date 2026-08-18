"""Apply the experimental internal RII Yamada reducer on the analysis branch."""

from pathlib import Path


def patch_state_builder() -> None:
    path = Path("src/knotted_graph/invariants/yamada/state_compact.py")
    text = path.read_text()
    marker = "    def build(self, state: tuple[int, ...]) -> CompactGraph:\n"
    if "def reduce_reidemeister_ii(self):" in text:
        return
    assert text.count(marker) == 1
    methods = r'''    def _find_reidemeister_ii_pair(self):
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

'''
    path.write_text(text.replace(marker, methods + marker))


def patch_polynomial() -> None:
    path = Path("src/knotted_graph/invariants/yamada/polynomial.py")
    text = path.read_text()
    if "prepared, _ = prepared.reduce_reidemeister_ii()" in text:
        return
    old = '''        prepared = PreparedCompactStateBuilder.prepare(\n            self.vertices,\n            self.crossings,\n            self.arcs,\n            _ordered_crossing_ports,\n        )\n        for config in itertools.product([0, 1, 2], repeat=len(self.crossings)):\n            yield prepared.build(config), config.count(0) - config.count(1)\n'''
    new = '''        prepared = PreparedCompactStateBuilder.prepare(\n            self.vertices,\n            self.crossings,\n            self.arcs,\n            _ordered_crossing_ports,\n        )\n        prepared, _ = prepared.reduce_reidemeister_ii()\n        crossing_count = len(prepared.crossing_ids)\n        for config in itertools.product([0, 1, 2], repeat=crossing_count):\n            yield prepared.build(config), config.count(0) - config.count(1)\n'''
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))


if __name__ == "__main__":
    patch_state_builder()
    patch_polynomial()
