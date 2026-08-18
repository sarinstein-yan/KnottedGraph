from __future__ import annotations

import gc
import json
import multiprocessing as mp
import statistics
import threading
import time
from functools import cached_property

import numpy as np
import psutil
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal._memory import _optimized_init
from knotted_graph.applications.nodal.models import (
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


class ReferenceNodalSkeleton(NodalSkeleton):
    """Frozen implementation immediately before the memory optimizations."""

    def __init__(
        self,
        char,
        k_symbols=None,
        span=((-np.pi, np.pi), (-np.pi, np.pi), (0, np.pi)),
        dimension=200,
        axis_scale=(1.0, 1.0, 2.0),
    ):
        _optimized_init(
            self,
            char,
            k_symbols=k_symbols,
            span=span,
            dimension=dimension,
            axis_scale=axis_scale,
        )
        self.kx_grid, self.ky_grid, self.kz_grid = np.meshgrid(
            self.kx_vals,
            self.ky_vals,
            self.kz_vals,
            indexing="ij",
        )

    @cached_property
    def spectrum(self):
        return np.sqrt(np.sum(self._bloch_vec_grid**2, axis=0))



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


def _builder(case_name: str):
    return dict(_specs())[case_name]


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


def _process_worker(kind: str, case_name: str, dimension: int, queue):
    try:
        builder = _builder(case_name)
        char = builder()
        cls = ReferenceNodalSkeleton if kind == "reference" else NodalSkeleton
        gc.collect()

        process = psutil.Process()
        baseline_rss = process.memory_info().rss
        peak_rss = [baseline_rss]
        stop = threading.Event()

        def sample_rss():
            while not stop.is_set():
                rss = process.memory_info().rss
                if rss > peak_rss[0]:
                    peak_rss[0] = rss
                time.sleep(0.001)

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        start = time.perf_counter()
        try:
            model = cls(
                char,
                k_symbols=(kx, ky, kz),
                dimension=dimension,
                axis_scale=(1.0, 1.0, 1.5),
            )
            image = model._skeleton_image
            elapsed = time.perf_counter() - start
            current_rss = process.memory_info().rss
            peak_rss[0] = max(peak_rss[0], current_rss)
        finally:
            stop.set()
            sampler.join(timeout=1.0)

        queue.put(
            {
                "status": "ok",
                "time_s": elapsed,
                "baseline_rss_bytes": baseline_rss,
                "peak_rss_bytes": peak_rss[0],
                "incremental_peak_rss_bytes": max(0, peak_rss[0] - baseline_rss),
                "skeleton_voxels": int(np.count_nonzero(image)),
                "grids_materialized": [
                    name in model.__dict__
                    for name in ("kx_grid", "ky_grid", "kz_grid")
                ],
            }
        )
    except BaseException as exc:  # pragma: no cover - benchmark diagnostics
        queue.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _fresh_process_measure(kind: str, case_name: str, dimension: int):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_process_worker,
        args=(kind, case_name, dimension, queue),
    )
    process.start()
    process.join(120.0)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        raise AssertionError(f"{case_name}/{kind}: memory worker timed out")
    if queue.empty():
        raise AssertionError(
            f"{case_name}/{kind}: worker exited {process.exitcode} without data"
        )
    result = queue.get()
    if result["status"] != "ok":
        raise AssertionError(f"{case_name}/{kind}: {result['error']}")
    return result


def main():
    dimension = 160
    rows = []

    for case_name, builder in _specs():
        print(f"CASE={case_name}", flush=True)
        reference = _model(ReferenceNodalSkeleton, builder, dimension)
        candidate = _model(NodalSkeleton, builder, dimension)

        for name in ("kx_grid", "ky_grid", "kz_grid"):
            if name in candidate.__dict__:
                raise AssertionError(f"{case_name}: {name} was eagerly allocated")

        stacked_t, stacked = _median(lambda: _stacked_spectrum(reference), repeats=3)
        streamed_samples = []
        streamed = None
        for _ in range(3):
            fresh = _model(NodalSkeleton, builder, dimension)
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
        if streamed_t > stacked_t * 1.02:
            raise AssertionError(
                f"{case_name}: streamed spectrum regressed runtime: "
                f"stacked={stacked_t:.6g}s streamed={streamed_t:.6g}s"
            )

        reference_image = reference._skeleton_image
        candidate_image = candidate._skeleton_image
        skeleton_exact = np.array_equal(reference_image, candidate_image)
        if not skeleton_exact:
            mismatch = int(np.count_nonzero(reference_image != candidate_image))
            raise AssertionError(
                f"{case_name}: memory candidate changed {mismatch} skeleton voxels"
            )

        reference_runs = [
            _fresh_process_measure("reference", case_name, dimension)
            for _ in range(2)
        ]
        candidate_runs = [
            _fresh_process_measure("candidate", case_name, dimension)
            for _ in range(2)
        ]
        reference_peak = min(
            run["incremental_peak_rss_bytes"] for run in reference_runs
        )
        candidate_peak = min(
            run["incremental_peak_rss_bytes"] for run in candidate_runs
        )
        reference_e2e = statistics.median(run["time_s"] for run in reference_runs)
        candidate_e2e = statistics.median(run["time_s"] for run in candidate_runs)

        if candidate_peak >= reference_peak:
            raise AssertionError(
                f"{case_name}: candidate did not reduce incremental peak RSS: "
                f"reference={reference_peak} candidate={candidate_peak}"
            )
        if candidate_e2e > reference_e2e * 1.03:
            raise AssertionError(
                f"{case_name}: candidate regressed construction->skeleton time: "
                f"reference={reference_e2e:.6g}s candidate={candidate_e2e:.6g}s"
            )
        if any(candidate_runs[0]["grids_materialized"]):
            raise AssertionError(
                f"{case_name}: skeleton workflow unexpectedly materialized dense grids"
            )
        if {
            run["skeleton_voxels"] for run in reference_runs + candidate_runs
        } != {int(np.count_nonzero(reference_image))}:
            raise AssertionError(f"{case_name}: fresh-process skeleton outputs changed")

        row = {
            "case": case_name,
            "dimension": dimension,
            "stacked_spectrum_s": stacked_t,
            "streamed_spectrum_s": streamed_t,
            "streamed_speedup": stacked_t / streamed_t,
            "spectrum_exact": exact,
            "skeleton_exact": skeleton_exact,
            "reference_pipeline_s": reference_e2e,
            "candidate_pipeline_s": candidate_e2e,
            "pipeline_speedup": reference_e2e / candidate_e2e,
            "reference_incremental_peak_rss_bytes": reference_peak,
            "candidate_incremental_peak_rss_bytes": candidate_peak,
            "incremental_peak_rss_reduction_bytes": reference_peak - candidate_peak,
            "incremental_peak_rss_reduction_fraction": 1.0 - candidate_peak / reference_peak,
            "legacy_dense_coordinate_grid_bytes": _legacy_grid_bytes(dimension),
            "legacy_bloch_stack_bytes": _bloch_stack_bytes(dimension),
            "lazy_grid_bytes_avoided_until_access": _legacy_grid_bytes(dimension),
            "streamed_bloch_stack_bytes_avoided": _bloch_stack_bytes(dimension),
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    candidate = _model(NodalSkeleton, _specs()[0][1], 32)
    assert all(
        name not in candidate.__dict__
        for name in ("kx_grid", "ky_grid", "kz_grid")
    )
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
