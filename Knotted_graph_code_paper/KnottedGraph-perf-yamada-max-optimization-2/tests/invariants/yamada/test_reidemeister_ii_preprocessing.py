from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.fast import (
    FastNegamiSpecializedEvaluator,
    FastYamadaEvaluator,
    add,
    shift,
    to_sympy,
)
from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    _ordered_crossing_ports,
)
from knotted_graph.invariants.yamada.state_compact import (
    PreparedCompactStateBuilder,
)
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def _weave(crossing_count: int, phase: float = 0.0) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    span = float(crossing_count + 2)
    left, right = -span, span
    graph.add_node("u", pos=np.array([left, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([right, 0.0, 0.0]))

    x = np.linspace(left, right, crossing_count + 3)
    y1 = np.zeros(crossing_count + 3)
    y2 = np.zeros(crossing_count + 3)
    for index in range(1, crossing_count + 2):
        sign = 1.0 if index % 2 else -1.0
        amplitude = 1.0 + 0.07 * np.sin(index + phase)
        y1[index] = sign * amplitude
        y2[index] = -sign * amplitude
    strand1 = np.column_stack(
        [x, y1, np.full(crossing_count + 3, 0.5)]
    )
    strand2 = np.column_stack(
        [x, y2, np.full(crossing_count + 3, -0.5)]
    )
    strand1[[0, -1], 2] = 0.0
    strand2[[0, -1], 2] = 0.0
    third = np.array(
        [[left, 0, 0], [left + 1, 3, 0], [right - 1, 3, 0], [right, 0, 0]],
        dtype=float,
    )
    graph.add_edge("u", "v", pts=strand1)
    graph.add_edge("u", "v", pts=strand2)
    graph.add_edge("u", "v", pts=third)
    return graph


def _prepared(crossing_count: int, phase: float = 0.0):
    processor = PDCode(_weave(crossing_count, phase))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    assert len(processor.crossings) == crossing_count
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )
    return yamada, prepared


def _state_sum(builder, evaluator):
    total = ()
    for config in itertools.product(
        [0, 1, 2], repeat=len(builder.crossing_ids)
    ):
        value = evaluator.compute_laurent(builder.build(config))
        total = add(
            total,
            shift(value, config.count(0) - config.count(1)),
        )
    return sp.expand(to_sympy(total, A))


def test_rii_preprocessing_preserves_exact_unnormalized_state_sum():
    for evaluator_cls in (
        FastNegamiSpecializedEvaluator,
        FastYamadaEvaluator,
    ):
        for crossing_count in range(1, 8):
            for phase in (0.0, 0.37):
                _, original = _prepared(crossing_count, phase)
                reduced, moves = original.reduce_reidemeister_ii()
                assert moves == crossing_count // 2
                assert len(reduced.crossing_ids) == crossing_count % 2
                assert _state_sum(reduced, evaluator_cls()) == _state_sum(
                    original, evaluator_cls()
                )


def test_public_yamada_result_is_unchanged_by_internal_rii_reduction():
    for crossing_count in range(1, 7):
        yamada, original = _prepared(crossing_count, phase=0.19)
        expected = _state_sum(original, FastNegamiSpecializedEvaluator())
        actual = yamada.compute(
            A,
            normalize=False,
            n_jobs=1,
            method="negami",
        )
        assert sp.expand(actual) == expected


def test_rii_reduction_does_not_mutate_prepared_input():
    _, original = _prepared(6)
    snapshot = (
        original.crossing_ids,
        original.ordered_ports,
        original.arc_partner,
        original.fixed_terminal_index,
        original.crossing_for_port,
    )
    reduced, moves = original.reduce_reidemeister_ii()
    assert moves == 3
    assert len(reduced.crossing_ids) == 0
    assert snapshot == (
        original.crossing_ids,
        original.ordered_ports,
        original.arc_partner,
        original.fixed_terminal_index,
        original.crossing_for_port,
    )
