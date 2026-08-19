from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.diagram_structural import compute_structural_laurent
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator, native_available
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
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


@pytest.mark.parametrize("n", [3, 5, 7])
def test_structural_recursion_matches_independent_legacy_state_sum(n):
    """Generic structural recursion is exactly equal to the old 3**c implementation."""
    prepared = _prepared(n)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    legacy = evaluator.compute_prepared_bulk_laurent(prepared)
    stats = {}
    structural = compute_structural_laurent(prepared, evaluator, stats=stats)
    assert structural == legacy
    assert stats["calls"] >= 1


@pytest.mark.parametrize("n", [3, 5, 7, 9, 11])
def test_certified_theta_closed_form_matches_legacy_state_sum(n):
    """Published Theta(n) formula is identical to the retained exact oracle."""
    assert native_available()
    prepared = _prepared(n)
    oracle = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = oracle.compute_prepared_bulk_laurent(prepared)

    optimized = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    actual = optimized.compute_prepared_laurent(prepared)
    assert actual == expected
    assert optimized.theta_twist_calls == 1
    assert optimized.structural_calls == 0


@pytest.mark.parametrize("n", [3, 5])
def test_certified_theta_closed_form_matches_legacy_for_mirror(n):
    """The closed form also preserves exact output under A <-> A^-1 mirroring."""
    assert native_available()
    prepared = _prepared(n, mirror=True)
    oracle = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = oracle.compute_prepared_bulk_laurent(prepared)

    optimized = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    actual = optimized.compute_prepared_laurent(prepared)
    assert actual == expected
    assert optimized.theta_twist_calls == 1
    assert optimized.structural_calls == 0
