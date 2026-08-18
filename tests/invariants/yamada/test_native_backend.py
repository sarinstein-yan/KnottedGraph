from __future__ import annotations

import random

import networkx as nx
import pytest

from knotted_graph.invariants.yamada.compact import (
    CompactGraph,
    CompactYamadaEvaluator,
    PythonCompactYamadaEvaluator,
)
from knotted_graph.invariants.yamada.native import (
    NativeCompactEvaluator,
    native_available,
)


def _random_multigraph(seed: int) -> nx.MultiGraph:
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n))
    for _ in range(rng.randint(0, 15)):
        graph.add_edge(rng.randrange(n), rng.randrange(n))
    return graph


@pytest.mark.parametrize("seed", range(40))
def test_selected_backend_matches_explicit_python_reference(seed: int):
    graph = _random_multigraph(20260818 + seed)
    compact = CompactGraph.from_networkx(graph)
    expected = PythonCompactYamadaEvaluator().compute_laurent(compact)
    actual = CompactYamadaEvaluator().compute_laurent(compact)
    assert actual == expected


def test_native_constructor_is_selected_when_extension_is_present():
    evaluator = CompactYamadaEvaluator()
    if native_available():
        assert isinstance(evaluator, NativeCompactEvaluator)
        assert evaluator.backend == "native"
    else:
        assert isinstance(evaluator, PythonCompactYamadaEvaluator)


def test_native_batch_matches_python_state_sum_when_available():
    if not native_available():
        pytest.skip("native extension is not built in this environment")

    states = []
    for seed in range(25):
        graph = CompactGraph.from_networkx(_random_multigraph(9000 + seed))
        exponent = seed % 7 - 3
        states.append((graph, exponent))

    from knotted_graph.invariants.yamada.fast import add, shift

    reference = PythonCompactYamadaEvaluator()
    expected = ()
    for graph, exponent in states:
        expected = add(expected, shift(reference.compute_laurent(graph), exponent))

    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    actual = evaluator.compute_many_laurent(states)
    assert actual == expected
    assert evaluator.native_calls == 1


def test_python_fallback_remains_available_independently_of_native_extension():
    graph = nx.MultiGraph(nx.complete_graph(4))
    compact = CompactGraph.from_networkx(graph)
    reference = PythonCompactYamadaEvaluator()
    assert reference.compute_laurent(compact)
    assert isinstance(reference.memo, dict)
