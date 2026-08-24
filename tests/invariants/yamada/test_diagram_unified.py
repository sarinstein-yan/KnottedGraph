from __future__ import annotations

import itertools
import random

import pytest

from knotted_graph.invariants.yamada import factorized_frontier as factorized_module
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


def _as_laurent(value):
    return tuple((int(power), int(coefficient)) for power, coefficient in value)


def _factorized_has_bigint_backend() -> bool:
    return native_factorized_available() and hasattr(
        factorized_module._yamada_factorized_frontier,
        "compute_factorized_frontier_bigint",
    )


def _single_vertex_loop_args(loop_count: int):
    port_count = 2 * loop_count
    wire_partner = [-1] * port_count
    for index in range(loop_count):
        left = 2 * index
        right = left + 1
        wire_partner[left] = right
        wire_partner[right] = left
    return (
        [0],
        [0] * port_count,
        wire_partner,
        [0] * port_count,
        [-1] * port_count,
        [-1] * port_count,
        [0],
    )


def _mul_poly(left, right):
    out = {}
    for left_power, left_coeff in left.items():
        for right_power, right_coeff in right.items():
            power = left_power + right_power
            out[power] = out.get(power, 0) + left_coeff * right_coeff
    return {power: coeff for power, coeff in out.items() if coeff}


def _single_vertex_loop_expected(loop_count: int):
    poly = {0: -1}
    one_minus_q = {-1: -1, 0: -1, 1: -1}
    for _ in range(loop_count):
        poly = _mul_poly(poly, one_minus_q)
    return tuple(sorted(poly.items()))


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


@pytest.mark.skipif(
    not _factorized_has_bigint_backend(),
    reason="native factorized bigint frontier not built",
)
@pytest.mark.parametrize("crossing_count", [0, 1, 2, 3, 4])
def test_factorized_native_int64_and_bigint_backends_agree(crossing_count):
    for offset in range(8):
        prepared = _random_prepared(123000 + 149 * crossing_count + offset, crossing_count)
        data = factorized_module.build_factorized_frontier(prepared)
        args = factorized_module._frontier_args(data)
        int64_value = factorized_module._yamada_factorized_frontier.compute_factorized_frontier_int64(*args)
        bigint_value = factorized_module._yamada_factorized_frontier.compute_factorized_frontier_bigint(*args)
        assert _as_laurent(bigint_value) == _as_laurent(int64_value)


@pytest.mark.skipif(
    not _factorized_has_bigint_backend(),
    reason="native factorized bigint frontier not built",
)
def test_factorized_native_bigint_handles_true_int64_overflow():
    args = _single_vertex_loop_args(45)
    with pytest.raises(OverflowError):
        factorized_module._yamada_factorized_frontier.compute_factorized_frontier_int64(*args)

    actual = _as_laurent(
        factorized_module._yamada_factorized_frontier.compute_factorized_frontier_bigint(*args)
    )
    assert actual == _single_vertex_loop_expected(45)
    assert max(abs(coefficient) for _, coefficient in actual) > 2**63 - 1


def test_factorized_overflow_dispatch_uses_native_bigint(monkeypatch):
    calls = []

    class FakeFactorizedExtension:
        def compute_factorized_frontier_int64(self, *args):
            calls.append("int64")
            raise OverflowError("forced test overflow")

        def compute_factorized_frontier_bigint(self, *args):
            calls.append("bigint")
            return ((0, 2**80),)

    monkeypatch.setattr(
        factorized_module,
        "_yamada_factorized_frontier",
        FakeFactorizedExtension(),
    )
    stats = {}
    prepared = _random_prepared(125000, 0)

    assert factorized_module.compute_factorized_frontier_laurent(prepared, stats=stats) == (
        (0, 2**80),
    )
    assert calls == ["int64", "bigint"]
    assert stats == {"coefficient_backend": "native-bigint", "int64_overflow": True}


@pytest.mark.skipif(not native_frontier_available(), reason="native frontier not built")
@pytest.mark.parametrize("crossing_count", [0, 1, 2, 3, 4])
def test_native_frontier_matches_independent_exhaustive(crossing_count):
    for offset in range(16):
        prepared = _random_prepared(121000 + 137 * crossing_count + offset, crossing_count)
        stats = {}
        assert contract_frontier_laurent(prepared, stats=stats) == _exhaustive(prepared)
        assert stats["native_frontier_calls"] == 1
        assert stats.get("python_frontier_calls", 0) == 0
