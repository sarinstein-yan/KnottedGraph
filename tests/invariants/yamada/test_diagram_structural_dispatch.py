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


def _prepared(n: int):
    module = _load_torus_module()
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
    """Structural recursion is exactly equal to the old 3**c implementation."""
    prepared = _prepared(n)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    legacy = evaluator.compute_prepared_bulk_laurent(prepared)
    stats = {}
    structural = compute_structural_laurent(prepared, evaluator, stats=stats)
    assert structural == legacy
    assert stats["calls"] >= 1


def test_production_structural_dispatch_matches_legacy_at_eleven_crossings():
    """Exercise the production threshold on a still-manageable exact oracle."""
    assert native_available()
    prepared = _prepared(11)

    old_evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = old_evaluator.compute_prepared_bulk_laurent(prepared)

    new_evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    actual = new_evaluator.compute_prepared_laurent(prepared)
    assert actual == expected
    assert new_evaluator.structural_calls == 1
    assert new_evaluator.last_structural_stats is not None
    assert new_evaluator.last_structural_stats["inversion_steps"] >= 1
