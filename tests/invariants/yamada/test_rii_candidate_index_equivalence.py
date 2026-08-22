from __future__ import annotations

import random

from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def _legacy_find(builder: PreparedCompactStateBuilder):
    arc_partner = builder.arc_partner
    for first in range(len(builder.ordered_ports)):
        first_ports = builder.ordered_ports[first]
        for second in range(first):
            second_ports = builder.ordered_ports[second]
            second_position = {port: i for i, port in enumerate(second_ports)}
            shared = []
            for first_position, first_port in enumerate(first_ports):
                partner = arc_partner[first_port]
                if partner not in second_position:
                    continue
                shared.append(
                    (first_position, second_position[partner], first_port, partner)
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


def _random_builder(rng: random.Random, crossings: int):
    port_count = 4 * crossings
    ordered = tuple(
        tuple(range(4 * crossing, 4 * crossing + 4))
        for crossing in range(crossings)
    )
    ports = list(range(port_count))
    rng.shuffle(ports)
    arc_partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        arc_partner[left] = right
        arc_partner[right] = left
    crossing_for_port = tuple(port // 4 for port in range(port_count))
    plus_partner = [-1] * port_count
    minus_partner = [-1] * port_count
    for group in ordered:
        for a, b in ((0, 3), (1, 2)):
            plus_partner[group[a]] = group[b]
            plus_partner[group[b]] = group[a]
        for a, b in ((0, 1), (2, 3)):
            minus_partner[group[a]] = group[b]
            minus_partner[group[b]] = group[a]
    return PreparedCompactStateBuilder(
        vertex_ids=(),
        crossing_ids=tuple(range(crossings)),
        ordered_ports=ordered,
        arc_partner=tuple(arc_partner),
        fixed_terminal_index=(-1,) * port_count,
        crossing_for_port=crossing_for_port,
        plus_partner=tuple(plus_partner),
        minus_partner=tuple(minus_partner),
    )


def test_indexed_rii_candidate_search_matches_legacy_randomized():
    rng = random.Random(20260822)
    for crossings in range(2, 13):
        for _ in range(100):
            builder = _random_builder(rng, crossings)
            assert builder._find_reidemeister_ii_pair() == _legacy_find(builder)
