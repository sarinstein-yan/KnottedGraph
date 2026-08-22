from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator, native_available
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode


def _crossed_cycle():
    """A valid spatial 4-cycle with one nonadjacent-edge crossing in xy."""
    graph = nx.MultiGraph()
    positions = {
        0: np.array([-1.0, -1.0, 0.0]),
        1: np.array([1.0, 1.0, 1.0]),
        2: np.array([-1.0, 1.0, 0.0]),
        3: np.array([1.0, -1.0, -1.0]),
    }
    for node, pos in positions.items():
        graph.add_node(node, pos=pos)
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 0)):
        graph.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return graph


def _prepared_from_pd(processor):
    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices, yamada.crossings, yamada.arcs, _ordered_crossing_ports
    )
    return prepared.reduce_reidemeister_ii()[0]


def _python_state_sum(prepared):
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


def test_native_exhaustive_oracle_matches_exact_python_state_sum():
    assert native_available()
    # This is deliberately a validation oracle, not a production diagram route.
    processor = PDCode(_crossed_cycle())
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    prepared = _prepared_from_pd(processor)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    assert evaluator.compute_prepared_bulk_laurent(prepared) == _python_state_sum(prepared)


def test_public_method_aliases_use_identical_production_algorithm():
    A = sp.Symbol("A")
    processor = PDCode(_crossed_cycle())
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    yamada = Yamada.from_PDCode(processor)
    expected = yamada.compute(A, normalize=False, n_jobs=1, method="recursive")
    actual = yamada.compute(A, normalize=False, n_jobs=-1, method="negami")
    assert sp.expand(actual - expected) == 0
