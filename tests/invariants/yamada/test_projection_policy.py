import networkx as nx
import numpy as np
import pytest
import sympy as sp

from knotted_graph.projection import pd_code
from knotted_graph.projection.pd_code import ProjectionResult, compute_yamada_polynomial


class _FakeProcessor:
    def __init__(self, value):
        self.value = value
        self.vertices = {}
        self.crossings = {}
        self.arcs = {}

    def compute_yamada(self, variable, normalize=True, n_jobs=-1, method="negami"):
        return sp.Integer(self.value)


def _embedded_edge() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([1.0, 0.0, 0.0]))
    graph.add_edge("u", "v", pts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    return graph


def test_yamada_graph_entry_samples_default_ten_and_selects_fewest_crossings(monkeypatch):
    captured = {}
    crossing_counts = {
        (0.0, 0.0, 0.0): 4,
        (1.0, 0.0, 0.0): 2,
        (2.0, 0.0, 0.0): 2,
    }

    def fake_angles(count, order):
        captured["count"] = count
        captured["order"] = order
        return np.asarray([[float(i), 0.0, 0.0] for i in range(count)])

    def fake_compute_projection(graph, rotation_angles, rotation_order):
        crossings = crossing_counts.get(rotation_angles, 8)
        return ProjectionResult(
            processor=_FakeProcessor(value=int(rotation_angles[0]) + 100),
            rotation_angles=rotation_angles,
            rotation_order=rotation_order,
            pd_code=f"pd-{rotation_angles[0]}",
            num_crossings=crossings,
        )

    monkeypatch.setattr(pd_code, "generate_isotopy_angles", fake_angles)
    monkeypatch.setattr(pd_code, "_compute_projection", fake_compute_projection)

    A = sp.Symbol("A")
    result = compute_yamada_polynomial(_embedded_edge(), A, return_result=True)

    assert captured == {"count": 10, "order": "ZYX"}
    assert result.projection.rotation_angles == (1.0, 0.0, 0.0)
    assert result.projection.num_crossings == 2
    assert result.polynomial == 101


def test_yamada_graph_entry_explicit_rotation_bypasses_sampling(monkeypatch):
    def fail_if_sampled(count, order):
        raise AssertionError("explicit rotation should not sample projections")

    def fake_compute_projection(graph, rotation_angles, rotation_order):
        return ProjectionResult(
            processor=_FakeProcessor(value=7),
            rotation_angles=rotation_angles,
            rotation_order=rotation_order,
            pd_code="explicit",
            num_crossings=1,
        )

    monkeypatch.setattr(pd_code, "generate_isotopy_angles", fail_if_sampled)
    monkeypatch.setattr(pd_code, "_compute_projection", fake_compute_projection)

    A = sp.Symbol("A")
    result = compute_yamada_polynomial(
        _embedded_edge(),
        A,
        rotation_angles=(12, 34, 56),
        return_result=True,
    )

    assert result.projection.rotation_angles == (12.0, 34.0, 56.0)
    assert result.polynomial == 7


def test_yamada_graph_entry_warns_on_large_selected_diagram(monkeypatch):
    def fake_compute_projection(graph, rotation_angles, rotation_order):
        return ProjectionResult(
            processor=_FakeProcessor(value=1),
            rotation_angles=rotation_angles,
            rotation_order=rotation_order,
            pd_code="large",
            num_crossings=10,
        )

    monkeypatch.setattr(pd_code, "_compute_projection", fake_compute_projection)

    A = sp.Symbol("A")
    with pytest.warns(RuntimeWarning, match="10 crossings"):
        compute_yamada_polynomial(
            _embedded_edge(),
            A,
            rotation_angles=(0, 0, 0),
            crossing_warning_threshold=10,
        )


def test_yamada_graph_entry_rejects_non_embedded_graph_before_projection():
    A = sp.Symbol("A")

    with pytest.raises(ValueError, match="graph has no nodes"):
        compute_yamada_polynomial(nx.MultiGraph(), A, rotation_angles=(0, 0, 0))
