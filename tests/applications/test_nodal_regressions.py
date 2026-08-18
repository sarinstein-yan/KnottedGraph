from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import skimage.morphology as morph
import sympy as sp

import knotted_graph.applications.nodal.skeleton as skeleton_module
from knotted_graph.applications.nodal import NodalSkeleton


def _bare_nodal_skeleton() -> NodalSkeleton:
    """Construct only the state required by skeleton_graph()."""
    model = NodalSkeleton.__new__(NodalSkeleton)
    model.skeleton_graph_cache = None
    model.skeleton_graph_cache_args = None
    model._pv_data_args = None
    return model


def test_skeleton_graph_converts_external_image_to_graph(monkeypatch):
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1, 1, 1:3] = 1

    converted = nx.MultiGraph()
    converted.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    captured = {}

    def fake_skeleton_image_to_graph(value):
        captured["image"] = np.asarray(value).copy()
        return converted

    monkeypatch.setattr(
        skeleton_module,
        "skeleton_image_to_graph",
        fake_skeleton_image_to_graph,
    )
    monkeypatch.setattr(
        skeleton_module,
        "smooth_edges",
        lambda graph, epsilon, copy=False: graph,
    )

    model = _bare_nodal_skeleton()
    result = model.skeleton_graph(
        simplify=False,
        smooth_epsilon=0,
        skeleton_image=image,
    )

    assert result is converted
    assert captured["image"].dtype == bool
    assert np.array_equal(captured["image"], image.astype(bool))


def test_skeleton_graph_rejects_non_3d_external_image():
    model = _bare_nodal_skeleton()

    with pytest.raises(ValueError, match="three-dimensional"):
        model.skeleton_graph(
            simplify=False,
            smooth_epsilon=0,
            skeleton_image=np.zeros((4, 4), dtype=bool),
        )


def test_fields_pv_does_not_require_supported_berry_curvature():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (kx, ky, kz),
        k_symbols=(kx, ky, kz),
        dimension=5,
        axis_scale=(1.0, 1.0, 1.0),
    )

    assert not model._berry_prerequisites["valid"]
    fields = model.fields_pv
    assert "real" in fields.point_data
    assert "imag" in fields.point_data
    assert "gap" in fields.point_data
    assert "ES_helper" in fields.point_data
    assert "berry" not in fields.point_data
    assert "|berry|" not in fields.point_data


def test_broadcast_bloch_grid_is_byte_identical_to_dense_meshgrid():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (
            sp.sin(kx) + sp.cos(ky),
            sp.I * sp.Rational(1, 5),
            sp.cos(kz) + sp.sin(kx + ky),
        ),
        k_symbols=(kx, ky, kz),
        dimension=11,
    )

    dense_grids = (model.kx_grid, model.ky_grid, model.kz_grid)
    reference = np.asarray(
        [
            func(*dense_grids).astype(np.complex128)
            if expr.free_symbols
            else np.full_like(
                model.kx_grid,
                complex(expr),
                dtype=np.complex128,
            )
            for expr, func in zip(model.bloch_vec, model.bloch_vec_funcs)
        ]
    )

    assert np.array_equal(model._bloch_vec_grid, reference)


def test_cropped_lee_skeleton_is_byte_identical_to_full_volume():
    mask = np.zeros((32, 28, 30), dtype=bool)
    mask[7:25, 6:22, 8:24] = True
    mask[13:19, 3:25, 13:17] = True

    reference = morph.skeletonize(mask, method="lee")
    stub = SimpleNamespace(_interior_mask=mask)
    optimized = NodalSkeleton._skeleton_image.func(stub)

    assert np.array_equal(optimized, reference)


def test_coordinate_grids_are_lazy_but_preserve_legacy_arrays():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (sp.sin(kx), sp.cos(ky), sp.I / 5 + sp.cos(kz)),
        k_symbols=(kx, ky, kz),
        dimension=9,
    )

    grid_names = ("kx_grid", "ky_grid", "kz_grid")
    assert all(name not in model.__dict__ for name in grid_names)

    expected = np.meshgrid(
        model.kx_vals,
        model.ky_vals,
        model.kz_vals,
        indexing="ij",
    )
    for name, want in zip(grid_names, expected):
        got = getattr(model, name)
        assert name in model.__dict__
        assert np.array_equal(got, want)
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.flags.writeable
        assert got.flags.c_contiguous


def test_lazy_grid_preserves_assignment_and_deletion_semantics():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (kx, ky, sp.I / 5 + kz),
        k_symbols=(kx, ky, kz),
        dimension=5,
    )

    model.kx_grid = None
    assert model.kx_grid is None
    del model.kx_grid
    assert "kx_grid" not in model.__dict__
    regenerated = model.kx_grid
    expected = np.meshgrid(
        model.kx_vals, model.ky_vals, model.kz_vals, indexing="ij"
    )[0]
    assert np.array_equal(regenerated, expected)


def test_streamed_spectrum_is_byte_identical_to_stacked_bloch_grid():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (
            sp.sin(kx) + sp.cos(ky),
            sp.I * sp.Rational(1, 5),
            sp.cos(kz) + sp.sin(kx + ky),
        ),
        k_symbols=(kx, ky, kz),
        dimension=13,
    )

    streamed = model.spectrum.copy()
    reference = np.sqrt(np.sum(model._bloch_vec_grid**2, axis=0))
    assert np.array_equal(streamed, reference)


def test_skeleton_workflow_does_not_materialize_dense_coordinate_grids():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (
            sp.cos(kx) + sp.cos(ky) + sp.cos(kz),
            sp.I * sp.Rational(1, 2),
            sp.sin(kx),
        ),
        k_symbols=(kx, ky, kz),
        dimension=17,
    )

    _ = model._skeleton_image
    assert all(
        name not in model.__dict__
        for name in ("kx_grid", "ky_grid", "kz_grid")
    )


def test_skeleton_coords_match_legacy_dense_grid_formula_exactly():
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    model = NodalSkeleton(
        (
            sp.cos(kx) + sp.cos(ky) + sp.cos(kz),
            sp.I * sp.Rational(1, 2),
            sp.sin(kx),
        ),
        k_symbols=(kx, ky, kz),
        dimension=17,
    )

    indices = np.where(model._skeleton_image)
    expected_grids = np.meshgrid(
        model.kx_vals,
        model.ky_vals,
        model.kz_vals,
        indexing="ij",
    )
    expected = np.asarray(
        [grid[indices] for grid in expected_grids]
    ).T

    assert np.array_equal(model.skeleton_coords, expected)
    # The optimized property itself must not force the public dense grids into
    # memory merely to obtain skeleton coordinates.
    assert all(
        name not in model.__dict__
        for name in ("kx_grid", "ky_grid", "kz_grid")
    )


def test_memory_optimizations_are_native_class_implementation():
    """Optimized behavior must not be installed by import-time monkeypatching."""
    assert NodalSkeleton.__init__.__module__ == skeleton_module.__name__
    assert NodalSkeleton.spectrum.func.__module__ == skeleton_module.__name__
    assert NodalSkeleton.skeleton_coords.func.__module__ == skeleton_module.__name__
    assert NodalSkeleton.kx_grid.__class__.__module__ == skeleton_module.__name__
