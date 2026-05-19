from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .curve_io import (
    network_from_vertices,
    read_obj_vertices,
    write_curve_obj,
    write_layout_json,
    write_repulsor_curve,
)
from .driver import DEFAULT_DRIVER_BINARY, DriverConfig, SolverOptions, build_driver, effective_repulsor_root, run_driver
from .metrics import clearance_report, node_distance, read_certificate, read_history_summary
from .models import RepulsiveLayoutResult
from .protein_examples import build_protein_example, set_special_node_distance
from .render import render_tube_html


def run_protein_example(
    sample: str,
    workspace: Path,
    *,
    pdb_cache: Path | None = None,
    pdb_path: Path | None = None,
    total_arc_points: int | None = None,
    seed: int = 7,
    target_node_distance: float | None = None,
    node_distance_scale: float = 1.0,
    solver_options: SolverOptions | None = None,
    driver_config: DriverConfig | None = None,
    force_build: bool = False,
    keep_workspace: bool = False,
    save_steps: bool = True,
    render_html: bool = True,
) -> RepulsiveLayoutResult:
    workspace = workspace.resolve()
    if workspace.exists() and not keep_workspace:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    driver_config = driver_config or DriverConfig()
    solver_options = solver_options or SolverOptions()
    pdb_cache = (pdb_cache or (workspace.parent / "pdb_cache")).resolve()

    build_driver(driver_config, force=force_build)

    network = build_protein_example(
        sample,
        pdb_cache=pdb_cache,
        total_arc_points=total_arc_points,
        seed=seed,
        pdb_path=pdb_path,
    )
    original_node_distance = node_distance(network.node_positions, network.node_order)
    target_distance = target_node_distance
    if target_distance is None and abs(node_distance_scale - 1.0) > 1e-12:
        target_distance = original_node_distance * node_distance_scale
    if target_distance is not None:
        _, applied_node_distance = set_special_node_distance(network, target_distance)
    else:
        applied_node_distance = original_node_distance

    initial_obj = workspace / "initial.obj"
    curve_txt = workspace / "repulsor_curve.txt"
    final_obj = workspace / "final.obj"
    initial_json = workspace / "initial_layout.json"
    final_json = workspace / "final_layout.json"
    initial_html = workspace / "initial_tubes.html"
    final_html = workspace / "final_tubes.html"
    metadata_json = workspace / "metadata.json"
    history_csv = workspace / "repulsor_history.csv"
    steps_dir = workspace / "certified_steps" if save_steps else None
    clearance_json = workspace / "clearance_report.json"

    arc_indices = write_curve_obj(network, initial_obj)
    initial_vertices = read_obj_vertices(initial_obj)
    write_repulsor_curve(initial_vertices, arc_indices, network.arc_order, curve_txt)

    params: dict[str, Any] = {
        "sample": sample.lower(),
        "seed": seed,
        "steps": solver_options.steps,
        "q": solver_options.q,
        "p": solver_options.p,
        "threads": solver_options.threads,
        "max_time": solver_options.max_time,
        "safe_fraction": solver_options.safe_fraction,
        "max_backtracks": solver_options.max_backtracks,
        "max_iter": solver_options.max_iter,
        "tolerance": solver_options.tolerance,
        "pin_special_vertices": not solver_options.free_special_vertices,
        "total_arc_points": total_arc_points,
        "node_distance_original": original_node_distance,
        "node_distance_target": target_distance,
        "node_distance_applied": applied_node_distance,
        "node_distance_scale": node_distance_scale,
    }
    write_layout_json(network, params, initial_json)
    if render_html:
        render_tube_html(network, initial_html, f"{network.name} initial")

    t0 = time.perf_counter()
    process = run_driver(
        input_curve=curve_txt,
        output_obj=final_obj,
        history_csv=history_csv,
        options=solver_options,
        config=driver_config,
        save_steps_dir=steps_dir,
    )
    elapsed = time.perf_counter() - t0

    final_vertices = read_obj_vertices(final_obj)
    final_network = network_from_vertices(network, final_vertices, arc_indices)
    params["node_distance_final"] = node_distance(final_network.node_positions, final_network.node_order)
    write_layout_json(final_network, params, final_json)
    if render_html:
        render_tube_html(final_network, final_html, f"{network.name} Repulsor safe-step")

    certificate = read_certificate(history_csv)
    history_summary = read_history_summary(history_csv)
    clearance = {
        "initial": clearance_report(initial_vertices, arc_indices, network.arc_order),
        "final": clearance_report(final_vertices, arc_indices, network.arc_order),
        "note": (
            "This is a static clearance sanity report. The topology-preserving "
            "certificate is the per-step Repulsor MaximumSafeStepSize history."
        ),
    }
    clearance_json.write_text(json.dumps(clearance, indent=2), encoding="utf-8")

    metadata: dict[str, Any] = {
        "example": network.name,
        "solver": "Repulsor",
        "workspace": str(workspace),
        "repulsor_root": str(effective_repulsor_root(driver_config)),
        "driver_source": str(Path(driver_config.driver_source).resolve()),
        "driver_binary": str(Path(driver_config.driver_binary or DEFAULT_DRIVER_BINARY).resolve()),
        "initial_obj": str(initial_obj),
        "curve_txt": str(curve_txt),
        "final_obj": str(final_obj),
        "initial_layout_json": str(initial_json),
        "final_layout_json": str(final_json),
        "initial_html": str(initial_html) if render_html else None,
        "final_html": str(final_html) if render_html else None,
        "history_csv": str(history_csv),
        "steps_dir": str(steps_dir) if steps_dir is not None else None,
        "clearance_report_json": str(clearance_json),
        "elapsed_seconds": elapsed,
        "certificate": certificate,
        "history_summary": history_summary,
        "clearance_summary": {
            "initial_min_distance": clearance["initial"]["min_non_adjacent_segment_distance"],
            "final_min_distance": clearance["final"]["min_non_adjacent_segment_distance"],
        },
        "arc_specs": network.arc_specs,
        "arc_vertex_indices": arc_indices,
        "parameters": params,
        "protein_metadata": network.metadata,
        "driver_stdout_tail": process.stdout.splitlines()[-20:],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return RepulsiveLayoutResult(workspace=workspace, metadata=metadata)
