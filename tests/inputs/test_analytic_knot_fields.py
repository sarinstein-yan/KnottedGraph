import sys
import types

import networkx as nx
import numpy as np
import pytest

from knotted_graph.inputs import (
    KnotFunction,
    KnotFunctionPath,
    SemiholomorphicPolynomial,
    braid_component_count,
    braid_to_semiholomorphic,
    inverse_stereographic_s3,
    sample_s3,
)


def test_inverse_stereographic_map_lands_on_s3():
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(100, 3))
    u, v = inverse_stereographic_s3(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    np.testing.assert_allclose(np.abs(u) ** 2 + np.abs(v) ** 2, 1.0, atol=1e-14)


def test_braid_component_count_matches_standard_closures():
    assert braid_component_count([1, 1, 1], 2) == 1
    assert braid_component_count([1, 1], 2) == 2
    assert braid_component_count([1, -2] * 3, 3) == 3


def test_generic_braid_compiler_resolves_non_torus_figure_eight_braid():
    polynomial, report = braid_to_semiholomorphic(
        [1, -2, 1, -2], strands=3,
        fourier_modes=(4, 8, 12, 16), validation_samples=256,
    )
    assert report.passed
    assert report.error_fraction < 0.2
    assert report.components == 1
    assert polynomial.degree_u == 3
    assert "not a formal proof" in report.interpretation


def test_torus_constructor_has_exact_zero_parametrization():
    p, q = 2, 3
    low, high = 0.0, 1.0
    for _ in range(100):
        b = (low + high) / 2
        a = b ** (q / p)
        if a * a + b * b < 1:
            low = b
        else:
            high = b
    b = (low + high) / 2
    a = b ** (q / p)
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    knot = KnotFunction.torus(p, q)
    np.testing.assert_allclose(
        knot.evaluate_s3(a * np.exp(1j * q * t), b * np.exp(1j * p * t)),
        0.0, atol=1e-12,
    )


def test_named_figure_eight_has_reference_and_generic_braid_routes():
    preferred = KnotFunction.from_name("4_1")
    assert preferred.metadata["construction"] == "published_reference_semiholomorphic"
    assert preferred.expected_components == 1
    assert preferred.projection_pole_value == pytest.approx(-4 + 0j)
    compiled = KnotFunction.from_name(
        "4_1", construction="braid",
        fourier_modes=(4, 8, 12, 16), validation_samples=256,
    )
    assert compiled.construction_report is not None
    assert compiled.construction_report.passed


def test_path_removes_scale_and_global_phase_gauge():
    start = KnotFunction.torus(2, 3)
    base = KnotFunction.torus(2, 5)
    end = KnotFunction(
        evaluator=lambda x, y, z: 7j * base(x, y, z),
        s3_evaluator=lambda u, v: 7j * base.evaluate_s3(u, v),
        name="scaled",
    )
    path = KnotFunctionPath(start, end, sample_count=512, seed=2)
    assert path.gauge.overlap_after_alignment.real >= 0
    assert abs(path.gauge.overlap_after_alignment.imag) < 1e-10
    u, v = sample_s3(100, seed=7)
    ratio = path.at(1).evaluate_s3(u, v) / end.evaluate_s3(u, v)
    np.testing.assert_allclose(ratio, ratio[0])


def test_projection_pole_sublevel_is_rejected():
    field = KnotFunction.from_semiholomorphic(
        SemiholomorphicPolynomial({(1, 0, 0): 1}), chart_angle=0
    )
    assert field.projection_pole_value == 0j
    with pytest.raises(ValueError, match="projection pole"):
        field.sublevel_mask(0.1, span=((-2, 2),) * 3, dimension=16)


def test_figure_eight_tubular_topology_converges():
    pytest.importorskip("skimage")
    report = KnotFunction.from_name("4_1").tubular_convergence(
        0.55, span=((-4.0, 4.0),) * 3, dimensions=(128, 160)
    )
    assert report.converged
    for diagnostic in report.diagnostics:
        assert diagnostic.matches_expected_tubular_neighborhood
        assert diagnostic.volume_components == 1
        assert diagnostic.surface_components == 1
        assert diagnostic.total_boundary_genus == 1
        assert diagnostic.surface_is_closed
        assert not diagnostic.touches_box_boundary


def test_spatial_graph_voxel_coordinates_are_rescaled(monkeypatch):
    field = KnotFunction.from_function(
        lambda x, y, z: (x * x + y * y + z * z - 0.5) + 1j * z
    )

    def fake_skeletonize(mask, *, padding=1):
        return mask

    def fake_extract(skeleton, **kwargs):
        graph = nx.MultiGraph()
        graph.add_node(0, pos=np.array([1.0, 2.0, 3.0]))
        graph.add_node(1, pos=np.array([2.0, 2.0, 3.0]))
        graph.add_edge(0, 1, pts=np.array([[1., 2., 3.], [2., 2., 3.]]), weight=1.)
        return graph

    monkeypatch.setitem(
        sys.modules, "knotted_graph.extraction",
        types.SimpleNamespace(
            skeletonize_volume=fake_skeletonize,
            skeleton_image_to_graph=fake_extract,
        ),
    )
    sample = field.sample(span=((-2, 2),) * 3, dimension=9)
    graph = field.to_spatial_graph(0.3, sample=sample)
    np.testing.assert_allclose(graph.nodes[0]["pos"], [-1.5, -1.0, -0.5])
    assert graph.edges[0, 1, 0]["weight"] == pytest.approx(0.5)
