from __future__ import annotations

import gc
import json
import statistics
import time

import numpy as np
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal._memory import _streamed_spectrum
from knotted_graph.applications.nodal.models import (
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


def _specs():
    return [
        ("hopf_0.20", lambda: hopf_link_bloch_vector(0.20, k_symbols=(kx, ky, kz))),
        ("trefoil_0.19", lambda: trefoil_bloch_vector(0.19, k_symbols=(kx, ky, kz))),
        (
            "torus12_0.50",
            lambda: pq_torus_knot_bloch_vector(1, 2, 0.50, k_symbols=(kx, ky, kz)),
        ),
        ("solomon_1.00", lambda: solomon_bloch_vector(1.00, k_symbols=(kx, ky, kz))),
    ]


def _model(builder, dimension):
    return NodalSkeleton(
        builder(),
        k_symbols=(kx, ky, kz),
        dimension=dimension,
        axis_scale=(1.0, 1.0, 1.5),
    )


def _median(fn, repeats=3):
    values = []
    answer = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def _stacked_spectrum(model: NodalSkeleton):
    shape = (model.dimension,) * 3
    grids = (
        model.kx_vals[:, None, None],
        model.ky_vals[None, :, None],
        model.kz_vals[None, None, :],
    )
    components = []
    for expr, func in zip(model.bloch_vec, model.bloch_vec_funcs):
        if expr.free_symbols:
            value = np.asarray(func(*grids), dtype=np.complex128)
            components.append(np.broadcast_to(value, shape))
        else:
            components.append(np.full(shape, complex(expr), dtype=np.complex128))
    bloch = np.asarray(components)
    return np.sqrt(np.sum(bloch**2, axis=0))


def _legacy_grid_bytes(dimension: int) -> int:
    return 3 * dimension**3 * np.dtype(np.float64).itemsize


def _bloch_stack_bytes(dimension: int) -> int:
    return 3 * dimension**3 * np.dtype(np.complex128).itemsize


def main():
    dimension = 160
    rows = []

    for case_name, builder in _specs():
        print(f"CASE={case_name}", flush=True)
        model = _model(builder, dimension)

        stacked_t, stacked = _median(lambda: _stacked_spectrum(model), repeats=3)
        streamed_t, streamed = _median(lambda: _streamed_spectrum(model), repeats=3)

        exact = np.array_equal(stacked, streamed)
        max_abs = float(np.max(np.abs(stacked - streamed)))
        if not exact:
            raise AssertionError(
                f"{case_name}: streamed spectrum changed values; max_abs={max_abs}"
            )

        # No RAM optimization is accepted if it makes the actual spectrum hot
        # path slower.  A 2% tolerance is only to avoid rejecting equal-speed
        # implementations due CI timer jitter; reported ratios remain raw.
        if streamed_t > stacked_t * 1.02:
            raise AssertionError(
                f"{case_name}: streamed spectrum regressed runtime: "
                f"stacked={stacked_t:.6g}s streamed={streamed_t:.6g}s"
            )

        row = {
            "case": case_name,
            "dimension": dimension,
            "stacked_spectrum_s": stacked_t,
            "streamed_spectrum_s": streamed_t,
            "streamed_speedup": stacked_t / streamed_t,
            "spectrum_exact": exact,
            "legacy_dense_coordinate_grid_bytes": _legacy_grid_bytes(dimension),
            "legacy_bloch_stack_bytes": _bloch_stack_bytes(dimension),
            "lazy_grid_bytes_avoided_until_access": _legacy_grid_bytes(dimension),
            "streamed_bloch_stack_bytes_avoided": _bloch_stack_bytes(dimension),
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    # Public compatibility of coordinate arrays: values, shape, dtype,
    # writability and C-contiguity must remain those of np.meshgrid.
    model = _model(_specs()[0][1], 32)
    expected = np.meshgrid(
        model.kx_vals, model.ky_vals, model.kz_vals, indexing="ij"
    )
    for name, want in zip(("kx_grid", "ky_grid", "kz_grid"), expected):
        got = getattr(model, name)
        assert np.array_equal(got, want)
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.flags.writeable
        assert got.flags.c_contiguous

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
