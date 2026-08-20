from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from knotted_graph.invariants.yamada.compact import CompactYamadaEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import bulk_laurent
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.invariants.yamada.streaming_candidate import ChunkedEvaluatorProxy
from knotted_graph.projection import PDCode


def _spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def _prepared(graph, rotation):
    processor = PDCode(graph)
    processor.compute(rotation_angles=rotation)
    calculator = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )
    prepared, _ = prepared.reduce_reidemeister_ii()
    return prepared


def test_chunked_native_batches_equal_one_full_batch_for_multiple_chunk_sizes():
    prepared = _prepared(
        _spring_embedding(nx.petersen_graph(), 9),
        (-134.58074129795634, 55.40942502382338, 0.0),
    )
    assert len(prepared.crossing_ids) == 6
    expected = bulk_laurent(prepared, CompactYamadaEvaluator())

    for chunk_size in (1, 7, 64, 256, 1000):
        actual = bulk_laurent(
            prepared,
            ChunkedEvaluatorProxy(
                CompactYamadaEvaluator(),
                chunk_size=chunk_size,
            ),
        )
        assert actual == expected


def test_chunking_preserves_ordered_laurent_sum_on_synthetic_state_stream():
    prepared = _prepared(
        _spring_embedding(nx.complete_graph(4), 3),
        (19.0, -23.0, 11.0),
    )
    states = list(
        itertools.islice(
            (
                (prepared.build(config), config.count(0) - config.count(1))
                for config in itertools.product(
                    (0, 1, 2), repeat=len(prepared.crossing_ids)
                )
            ),
            200,
        )
    )
    expected = CompactYamadaEvaluator().compute_many_laurent(states)
    for chunk_size in (1, 3, 16, 31):
        actual = ChunkedEvaluatorProxy(
            CompactYamadaEvaluator(), chunk_size=chunk_size
        ).compute_many_laurent(iter(states))
        assert actual == expected
