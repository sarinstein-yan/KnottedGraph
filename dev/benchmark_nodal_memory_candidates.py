from __future__ import annotations

import gc
import json
import statistics
import time

import numpy as np
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal._memory import install_memory_optimizations
from knotted_graph.applications.nodal.models import (
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


class CandidateNodalSkeleton(NodalSkeleton):
    """Isolated candidate so production NodalSkeleton remains untouched."""


install_memory_optimizations(CandidateNodalSkeleton)


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


def _model(cls, builder, dimension):
    return cls(
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
        reference = _model(NodalSkeleton, builder, dimension)
        candidate = _model(CandidateNodalSkeleton, builder, dimension)

        # Candidate constructor must not materialize any of the legacy dense
        # coordinate grids.
        for name in ("kx_grid", "ky_grid", "kz_grid"):
            if name in candidate.__dict__:
                raise AssertionError(f"{case_name}: {name} was eagerly allocated")

        stacked_t, stacked = _median(lambda: _stacked_spectrum(reference), repeats=3)
        streamed_t, streamed = _median(lambda: candidate.spectrum, repeats=1)
        # cached_property timing above would become zero after first access, so
        # use fresh candidates for the repeated timing samples.
        streamed_samples = []
        streamed = None
        for _ in range(3):
            fresh = _model(CandidateNodalSkeleton, builder, dimension)
            gc.collect()
            start = time.perf_counter()
            streamed = fresh.spectrum
            streamed_samples.append(time.perf_counter() - start)
        streamed_t = statistics.median(streamed_samples)

        exact = np.array_equal(stacked, streamed)
        max_abs = float(np.max(np.abs(stacked - streamed)))
        if not exact:
            raise AssertionError(
                f"{case_name}: streamed spectrum changed values; max_abs={max_abs}"
            )

        # No RAM optimization is accepted if it makes the actual spectrum hot
        # path slower. A 2% tolerance is solely CI timer jitter.
        if streamed_t > stacked_t * 1.02:
            raise AssertionError(
                f"{case_name}: streamed spectrum regressed runtime: "
                f"stacked={stacked_t:.6g}s streamed={streamed_t:.6g}s"
            )

        # Skeleton output must also be exactly unchanged before production use.
        reference_image = reference._skeleton_image
        candidate_image = candidate._skeleton_image
        skeleton_exact = np.array_equal(reference_image, candidate_image)
        if not skeleton_exact:
            mismatch = int(np.count_nonzero(reference_image != candidate_image))
            raise AssertionError(
                f"{case_name}: memory candidate changed {mismatch} skeleton voxels"
            )

        row = {
            "case": case_name,
            "dimension": dimension,
            "stacked_spectrum_s": stacked_t,
            "streamed_spectrum_s": streamed_t,
            "streamed_speedup": stacked_t / streamed_t,
            "spectrum_exact": exact,
            "skeleton_exact": skeleton_exact,
            "legacy_dense_coordinate_grid_bytes": _legacy_grid_bytes(dimension),
            "legacy_bloch_stack_bytes": _bloch_stack_bytes(dimension),
            "lazy_grid_bytes_avoided_until_access": _legacy_grid_bytes(dimension),
            "streamed_bloch_stack_bytes_avoided": _bloch_stack_bytes(dimension),
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    # Public coordinate-grid compatibility of the actual candidate descriptor:
    # same values, dtype, shape, writability and C-contiguity as np.meshgrid.
    candidate = _model(CandidateNodalSkeleton, _specs()[0][1], 32)
    assert all(name not in candidate.__dict__ for name in ("kx_grid", "ky_grid", "kz_grid"))
    expected = np.meshgrid(
        candidate.kx_vals,
        candidate.ky_vals,
        candidate.kz_vals,
        indexing="ij",
    )
    for name, want in zip(("kx_grid", "ky_grid", "kz_grid"), expected):
        got = getattr(candidate, name)
        assert name in candidate.__dict__
        assert np.array_equal(got, want)
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.flags.writeable
        assert got.flags.c_contiguous

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
