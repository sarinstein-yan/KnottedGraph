from __future__ import annotations

import gc
import json
import statistics
import time

import numpy as np
import skimage.morphology as morph
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import (
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)
from knotted_graph.core import remove_leaf_nodes, simplify_edges, smooth_edges
from knotted_graph.extraction import skeleton_image_to_graph

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


def _median(fn, repeats=3):
    values = []
    answer = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def _specs():
    return [
        ("hopf_0.20", lambda: hopf_link_bloch_vector(0.20, k_symbols=(kx, ky, kz))),
        ("trefoil_0.19", lambda: trefoil_bloch_vector(0.19, k_symbols=(kx, ky, kz))),
        (
            "torus12_0.50",
            lambda: pq_torus_knot_bloch_vector(
                1, 2, 0.50, k_symbols=(kx, ky, kz)
            ),
        ),
        ("solomon_1.00", lambda: solomon_bloch_vector(1.00, k_symbols=(kx, ky, kz))),
    ]


def _model(builder, dimension=160):
    return NodalSkeleton(
        builder(),
        k_symbols=(kx, ky, kz),
        dimension=dimension,
        axis_scale=(1.0, 1.0, 1.5),
    )


def _dense_bloch(model: NodalSkeleton):
    grids = (model.kx_grid, model.ky_grid, model.kz_grid)
    return np.asarray(
        [
            func(*grids).astype(np.complex128)
            if expr.free_symbols
            else np.full_like(
                model.kx_grid, complex(expr), dtype=np.complex128
            )
            for expr, func in zip(model.bloch_vec, model.bloch_vec_funcs)
        ]
    )


def _sparse_bloch(model: NodalSkeleton):
    shape = (model.dimension,) * 3
    grids = (
        model.kx_vals[:, None, None],
        model.ky_vals[None, :, None],
        model.kz_vals[None, None, :],
    )
    arrays = []
    for expr, func in zip(model.bloch_vec, model.bloch_vec_funcs):
        if expr.free_symbols:
            value = np.asarray(func(*grids), dtype=np.complex128)
            arrays.append(np.broadcast_to(value, shape))
        else:
            arrays.append(np.full(shape, complex(expr), dtype=np.complex128))
    return np.asarray(arrays)


def _crop_slices(mask: np.ndarray):
    occupied = [
        np.flatnonzero(mask.any(axis=axes))
        for axes in ((1, 2), (0, 2), (0, 1))
    ]
    if any(len(indices) == 0 for indices in occupied):
        return None
    bounds = []
    for axis, indices in enumerate(occupied):
        lo = max(0, int(indices[0]) - 1)
        hi = min(mask.shape[axis], int(indices[-1]) + 2)
        bounds.append(slice(lo, hi))
    return tuple(bounds)


def _cropped_lee(mask: np.ndarray):
    slices = _crop_slices(mask)
    if slices is None:
        return np.zeros_like(mask, dtype=bool), 0
    crop = mask[slices]
    skeleton_crop = morph.skeletonize(crop, method="lee")
    out = np.zeros_like(mask, dtype=bool)
    out[slices] = skeleton_crop
    return out, int(crop.size)


def _reference_skeleton_graph(model: NodalSkeleton):
    """Pre-optimization pipeline retained here as an exact benchmark oracle."""
    dense_bloch = _dense_bloch(model)
    spectrum = np.sqrt(np.sum(dense_bloch**2, axis=0))
    mask = spectrum.real == 0
    image = morph.skeletonize(mask, method="lee")
    if np.sum(image) == 0:
        raise ValueError("reference skeleton image is empty")
    graph = skeleton_image_to_graph(image)
    graph = remove_leaf_nodes(graph)
    graph = simplify_edges(graph)
    return smooth_edges(graph, epsilon=2, copy=False)


def _graph_summary(graph):
    edge_lengths = sorted(
        int(np.asarray(data.get("pts", [])).shape[0])
        for _, _, _, data in graph.edges(keys=True, data=True)
    )
    positions = sorted(
        tuple(float(x) for x in np.asarray(data.get("pos"), dtype=float))
        for _, data in graph.nodes(data=True)
    )
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "degrees": sorted(int(degree) for _, degree in graph.degree()),
        "edge_point_counts": edge_lengths,
        "positions": positions,
    }


def main():
    rows = []
    dimension = 160

    for case_name, builder in _specs():
        print(f"CASE={case_name}", flush=True)
        model = _model(builder, dimension=dimension)

        dense_t, dense = _median(lambda: _dense_bloch(model), repeats=2)
        sparse_t, sparse = _median(lambda: _sparse_bloch(model), repeats=2)
        grid_equal = np.array_equal(dense, sparse)
        production_grid_equal = np.array_equal(dense, model._bloch_vec_grid)
        grid_max_abs = float(np.max(np.abs(dense - sparse)))
        if not grid_equal or not production_grid_equal:
            raise AssertionError(
                f"{case_name}: optimized Bloch-grid evaluation is not byte-identical; "
                f"max_abs={grid_max_abs}"
            )

        mask_t, mask = _median(lambda: model._interior_mask, repeats=2)
        full_t, full_skeleton = _median(
            lambda: morph.skeletonize(mask, method="lee"), repeats=3
        )
        crop_t, crop_result = _median(lambda: _cropped_lee(mask), repeats=3)
        cropped_skeleton, crop_voxels = crop_result
        skeleton_equal = np.array_equal(full_skeleton, cropped_skeleton)
        production_skeleton_equal = np.array_equal(
            full_skeleton, model._skeleton_image
        )
        if not skeleton_equal or not production_skeleton_equal:
            mismatch = int(
                np.count_nonzero(full_skeleton != model._skeleton_image)
            )
            raise AssertionError(
                f"{case_name}: production Lee skeleton changed {mismatch} voxels"
            )

        raw_t, raw_graph = _median(
            lambda: skeleton_image_to_graph(full_skeleton), repeats=2
        )
        leaf_t, leaf_graph = _median(
            lambda: remove_leaf_nodes(raw_graph), repeats=3
        )
        simplify_t, simplified = _median(
            lambda: simplify_edges(leaf_graph), repeats=3
        )
        smooth_t, smoothed = _median(
            lambda: smooth_edges(simplified, epsilon=2, copy=True), repeats=3
        )

        reference_model = _model(builder, dimension=dimension)
        reference_t, reference_graph = _median(
            lambda: _reference_skeleton_graph(reference_model), repeats=1
        )
        production_model = _model(builder, dimension=dimension)
        production_t, production_graph = _median(
            lambda: production_model.skeleton_graph(
                simplify=True, smooth_epsilon=2
            ),
            repeats=1,
        )

        graph_equal = (
            _graph_summary(reference_graph)
            == _graph_summary(production_graph)
            == _graph_summary(smoothed)
        )
        if not graph_equal:
            raise AssertionError(
                f"{case_name}: production skeleton graph differs from exact reference"
            )

        row = {
            "case": case_name,
            "dimension": dimension,
            "voxels": int(mask.size),
            "interior_voxels": int(np.count_nonzero(mask)),
            "crop_voxels": crop_voxels,
            "crop_fraction": crop_voxels / mask.size,
            "skeleton_voxels": int(np.count_nonzero(full_skeleton)),
            "dense_bloch_s": dense_t,
            "sparse_bloch_s": sparse_t,
            "sparse_bloch_speedup": dense_t / sparse_t,
            "bloch_grid_exact": grid_equal and production_grid_equal,
            "mask_access_s": mask_t,
            "full_lee_s": full_t,
            "cropped_lee_s": crop_t,
            "cropped_lee_speedup": full_t / crop_t,
            "skeleton_exact": skeleton_equal and production_skeleton_equal,
            "raw_graph_s": raw_t,
            "remove_leaf_s": leaf_t,
            "simplify_edges_s": simplify_t,
            "smooth_edges_s": smooth_t,
            "reference_skeleton_graph_s": reference_t,
            "production_skeleton_graph_s": production_t,
            "end_to_end_speedup": reference_t / production_t,
            "graph_exact": graph_equal,
            "raw_nodes": raw_graph.number_of_nodes(),
            "raw_edges": raw_graph.number_of_edges(),
            "final_nodes": production_graph.number_of_nodes(),
            "final_edges": production_graph.number_of_edges(),
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)

        del dense, sparse, mask, full_skeleton, cropped_skeleton
        gc.collect()

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
