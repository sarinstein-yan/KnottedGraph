from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_structural import (
    _reduce_r1_queue,
    compute_structural_laurent,
)
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator, native_available
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import diagram_key, find_reidemeister_i
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode

ROOT = Path(__file__).resolve().parents[3]
TORUS_SCRIPT = ROOT / "dev" / "benchmark_topoly_essential_torus_scaling.py"


def _load_torus_module():
    dev = ROOT / "dev"
    if str(dev) not in sys.path:
        sys.path.insert(0, str(dev))
    spec = importlib.util.spec_from_file_location("kg_essential_torus_test", TORUS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepared(n: int, *, mirror: bool = False):
    module = _load_torus_module()
    if mirror:
        graph = module.essential_torus_graph(n)
        for _node, data in graph.nodes(data=True):
            data["pos"] = data["pos"].copy()
            data["pos"][2] *= -1.0
        for _u, _v, _key, data in graph.edges(keys=True, data=True):
            data["pts"] = data["pts"].copy()
            data["pts"][:, 2] *= -1.0
        processor = PDCode(graph)
        processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        assert len(processor.crossings) == n
    else:
        _graph, processor, _pdcode = module.prepare_essential_torus(n)

    yamada = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )
    reduced, _moves = prepared.reduce_reidemeister_ii()
    assert len(reduced.crossing_ids) == n
    return reduced


def _sequential_r1_chain(prepared):
    current = prepared
    exponent = 0
    moves = 0
    while True:
        found = find_reidemeister_i(current)
        if found is None:
            return current, exponent, moves
        current, delta = found
        exponent += delta
        moves += 1


@pytest.mark.parametrize(
    ("n", "mirror"),
    [(9, False), (11, False), (9, True), (11, True)],
)
def test_queue_r1_closure_is_identical_to_sequential_r1(n, mirror):
    """Queue R1 is an implementation optimization, not a changed reduction rule."""
    prepared = _prepared(n, mirror=mirror)
    queue_state, queue_exponent, queue_moves = _reduce_r1_queue(prepared)
    sequential_state, sequential_exponent, sequential_moves = _sequential_r1_chain(
        prepared
    )

    assert queue_exponent == sequential_exponent
    assert queue_moves == sequential_moves
    assert diagram_key(queue_state) == diagram_key(sequential_state)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_structural_recursion_matches_exhaustive_exact_state_sum(n):
    """Generic structural recursion equals the independent exhaustive 3**c sum."""
    prepared = _prepared(n)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    exhaustive = evaluator.compute_prepared_bulk_laurent(prepared)
    stats = {}
    structural = compute_structural_laurent(prepared, evaluator, stats=stats)
    assert structural == exhaustive
    assert stats["calls"] >= 1


@pytest.mark.parametrize("n", [9, 11])
def test_production_dispatch_is_generic_and_matches_exhaustive_oracle(n):
    """High-crossing production dispatch uses generic structural recursion exactly."""
    assert native_available()
    prepared = _prepared(n)
    oracle = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = oracle.compute_prepared_bulk_laurent(prepared)

    optimized = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    actual = optimized.compute_prepared_laurent(prepared)
    assert actual == expected
    assert optimized.structural_calls == 1
    assert optimized.last_structural_stats["r1_moves"] > 0
    assert optimized.last_structural_stats["max_bulk_crossings"] <= 4


@pytest.mark.parametrize("mirror", [False, True])
def test_r1_structural_path_preserves_both_crossing_orientations(mirror):
    """Both A^-2 and A^+2 curl orientations agree with exhaustive evaluation."""
    assert native_available()
    prepared = _prepared(9, mirror=mirror)
    oracle = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = oracle.compute_prepared_bulk_laurent(prepared)

    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    stats = {}
    actual = compute_structural_laurent(prepared, evaluator, stats=stats)
    assert actual == expected
    assert stats["r1_moves"] > 0
    assert stats["max_bulk_crossings"] <= 4
