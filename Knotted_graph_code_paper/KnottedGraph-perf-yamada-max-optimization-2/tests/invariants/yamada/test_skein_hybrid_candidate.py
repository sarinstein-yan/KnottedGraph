from __future__ import annotations

import numpy as np
import networkx as nx

from knotted_graph.invariants.yamada.compact import CompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, scale, shift
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import (
    bulk_laurent,
    compute_hybrid_laurent,
    invert_crossing,
    resolve_crossing,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode


def _spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def _weave(crossing_count: int) -> nx.MultiGraph:
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
        y1[index] = sign
        y2[index] = -sign
    strand1 = np.column_stack([x, y1, np.full(crossing_count + 3, 0.5)])
    strand2 = np.column_stack([x, y2, np.full(crossing_count + 3, -0.5)])
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


def _prepared(graph, rotation=(0.0, 0.0, 0.0)):
    processor = PDCode(graph)
    processor.compute(rotation_angles=rotation)
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )
    return prepared


def _bulk(prepared):
    return bulk_laurent(prepared, CompactYamadaEvaluator())


def _delta(positive, negative):
    difference = add(positive, scale(negative, -1))
    return add(shift(difference, 1), scale(shift(difference, -1), -1))


def test_single_crossing_resolution_partitions_the_exact_state_sum():
    prepared = _prepared(
        _spring_embedding(nx.complete_graph(4), 4),
        rotation=(-35.0, 21.0, 7.0),
    )
    if not prepared.crossing_ids:
        prepared = _prepared(_weave(1))

    for crossing_index in range(len(prepared.crossing_ids)):
        try:
            positive = resolve_crossing(prepared, crossing_index, 0)
            negative = resolve_crossing(prepared, crossing_index, 1)
            vertex = resolve_crossing(prepared, crossing_index, 2)
        except ValueError:
            continue
        expected = _bulk(prepared)
        actual = add(
            add(shift(_bulk(positive), 1), shift(_bulk(negative), -1)),
            _bulk(vertex),
        )
        assert actual == expected


def test_crossing_inversion_skein_identity_is_exact():
    prepared = _prepared(_weave(2))
    assert len(prepared.crossing_ids) == 2
    crossing_index = 0
    positive = resolve_crossing(prepared, crossing_index, 0)
    negative = resolve_crossing(prepared, crossing_index, 1)
    inverted = invert_crossing(prepared, crossing_index)
    assert add(_bulk(inverted), _delta(_bulk(positive), _bulk(negative))) == _bulk(
        prepared
    )


def test_hybrid_matches_bulk_on_connected_nontrivial_diagrams():
    cases = [
        (_spring_embedding(nx.complete_graph(4), 11), (17.0, -29.0, 13.0)),
        (_spring_embedding(nx.complete_bipartite_graph(3, 3), 7), (31.0, 19.0, -11.0)),
        (
            _spring_embedding(nx.petersen_graph(), 9),
            (-134.58074129795634, 55.40942502382338, 0.0),
        ),
    ]
    checked = 0
    for graph, rotation in cases:
        prepared = _prepared(graph, rotation)
        if len(prepared.crossing_ids) > 6:
            continue
        expected = _bulk(prepared)
        stats = {}
        actual = compute_hybrid_laurent(
            prepared,
            CompactYamadaEvaluator(),
            stats=stats,
        )
        assert actual == expected
        checked += 1
    assert checked >= 2


def test_inversion_lookahead_path_preserves_exact_output():
    rii_diagram = _prepared(_weave(2))
    twisted = invert_crossing(rii_diagram, 0)
    reduced, moves = twisted.reduce_reidemeister_ii()
    assert moves == 0
    assert len(reduced.crossing_ids) == 2

    stats = {
        "calls": 0,
        "memo_hits": 0,
        "rii_moves": 0,
        "inversion_steps": 0,
        "resolution_steps": 0,
        "bulk_fallbacks": 0,
    }
    expected = _bulk(twisted)
    actual = compute_hybrid_laurent(
        twisted,
        CompactYamadaEvaluator(),
        stats=stats,
    )
    assert actual == expected
    assert stats["inversion_steps"] >= 1
