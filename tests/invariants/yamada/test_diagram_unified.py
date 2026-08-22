from __future__ import annotations

import itertools
import random

import pytest

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_unified import (
    compute_unified_laurent,
    contract_frontier_laurent,
    native_frontier_available,
)
from knotted_graph.invariants.yamada.factorized_frontier import (
    compute_factorized_frontier_laurent,
    native_factorized_available,
)
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.state_compact import (
    PreparedCompactStateBuilder,
    _MINUS_PAIRS,
    _PLUS_PAIRS,
)


def _resolution_tables(ordered_ports, port_count):
    """Independent test helper for the two exact crossing smoothings."""
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
    return tuple(plus_partner), tuple(minus_partner)


def _random_prepared(seed: int, crossing_count: int):
    rng = random.Random(seed)
    vertex_count = rng.randint(0, 4)
    fixed_port_count = 2 * rng.randint(0, 4)
    if vertex_count == 0:
        fixed_port_count = 0

    crossing_ports = 4 * crossing_count
    port_count = crossing_ports + fixed_port_count
    ordered = tuple(
        tuple(range(4 * crossing, 4 * crossing + 4))
        for crossing in range(crossing_count)
    )
    fixed_terminal_index = [-1] * port_count
    crossing_for_port = [-1] * port_count
    for crossing, ports in enumerate(ordered):
        for port in ports:
            crossing_for_port[port] = crossing
    for port in range(crossing_ports, port_count):
        fixed_terminal_index[port] = rng.randrange(vertex_count)

    ports = list(range(port_count))
    rng.shuffle(ports)
    arc_partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        arc_partner[left] = right
        arc_partner[right] = left

    plus, minus = _resolution_tables(ordered, port_count)
    return PreparedCompactStateBuilder(
        vertex_ids=tuple(f"v{index}" for index in range(vertex_count)),
        crossing_ids=tuple(f"c{index}" for index in range(crossing_count)),
        ordered_ports=ordered,
        arc_partner=tuple(arc_partner),
        fixed_terminal_index=tuple(fixed_terminal_index),
        crossing_for_port=tuple(crossing_for_port),
        plus_partner=plus,
        minus_partner=minus,
    )


def _exhaustive(prepared):
    evaluator = PythonCompactYamadaEvaluator()
    total = ()
    for config in itertools.product((0, 1, 2), repeat=len(prepared.crossing_ids)):
        total = add(
            total,
            shift(
                evaluator.compute_laurent(prepared.build(config)),
                config.count(0) - config.count(1),
            ),
        )
    return total


@pytest.mark.parametrize("crossing_count", [0, 1, 2, 3, 4])
def test_unified_matches_independent_exhaustive_random_diagrams(crossing_count):
    for offset in range(16):
        prepared = _random_prepared(117000 + 131 * crossing_count + offset, crossing_count)
        assert compute_unified_laurent(prepared) == _exhaustive(prepared)


@pytest.mark.skipif(not native_factorized_available(), reason="native factorized frontier not built")
@pytest.mark.parametrize("crossing_count", [0, 1, 2, 3, 4])
def test_factorized_frontier_matches_independent_exhaustive(crossing_count):
    for offset in range(32):
        prepared = _random_prepared(119000 + 139 * crossing_count + offset, crossing_count)
        assert compute_factorized_frontier_laurent(prepared) == _exhaustive(prepared)


@pytest.mark.skipif(not native_frontier_available(), reason="native frontier not built")
@pytest.mark.parametrize("crossing_count", [0, 1, 2, 3, 4])
def test_native_frontier_matches_independent_exhaustive(crossing_count):
    for offset in range(16):
        prepared = _random_prepared(121000 + 137 * crossing_count + offset, crossing_count)
        stats = {}
        assert contract_frontier_laurent(prepared, stats=stats) == _exhaustive(prepared)
        assert stats["native_frontier_calls"] == 1
        assert stats.get("python_frontier_calls", 0) == 0
