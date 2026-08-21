from __future__ import annotations

import random

from knotted_graph.invariants.yamada.diagram_frontier import plan_diagram_frontier
from knotted_graph.invariants.yamada.frontier_ordering import (
    MULTISTART_MIN_CROSSINGS,
    plan_frontier_to_target,
)
from knotted_graph.invariants.yamada.skein_hybrid import _resolution_tables
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def _random_crossing_diagram(seed: int, crossings: int):
    rng = random.Random(seed)
    port_count = 4 * crossings
    ordered = tuple(tuple(range(4 * i, 4 * i + 4)) for i in range(crossings))
    ports = list(range(port_count))
    rng.shuffle(ports)
    arc_partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        arc_partner[left] = right
        arc_partner[right] = left
    crossing_for_port = [-1] * port_count
    for crossing, group in enumerate(ordered):
        for port in group:
            crossing_for_port[port] = crossing
    plus, minus = _resolution_tables(ordered, port_count)
    return PreparedCompactStateBuilder(
        vertex_ids=(),
        crossing_ids=tuple(range(crossings)),
        ordered_ports=ordered,
        arc_partner=tuple(arc_partner),
        fixed_terminal_index=tuple([-1] * port_count),
        crossing_for_port=tuple(crossing_for_port),
        plus_partner=plus,
        minus_partner=minus,
    )


def test_multistart_crossover_guard_keeps_small_rejected_plan():
    assert MULTISTART_MIN_CROSSINGS == 12
    prepared = _random_crossing_diagram(230032, 10)
    cheap = plan_diagram_frontier(prepared)
    planned = plan_frontier_to_target(prepared, 10)
    assert cheap["peak_ports"] == 12
    assert planned == cheap


def test_multistart_rescues_known_twelve_crossing_generic_case():
    prepared = _random_crossing_diagram(232068, 12)
    cheap = plan_diagram_frontier(prepared)
    planned = plan_frontier_to_target(prepared, 10)
    assert cheap["peak_ports"] == 12
    assert planned["initial_peak_ports"] == 12
    assert planned["peak_ports"] == 10
    assert planned["max_boundary_ports"] == 6
    assert planned["ordering_multistart"] is True
    assert planned["ordering_candidates"] == 12
