from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import networkx as nx

from .curve_io import (
    network_from_vertices,
    read_obj_vertices,
    write_curve_obj,
    write_layout_json,
    write_repulsor_curve,
)
from .decimation import DecimationOptions, decimate_curve_network
from .driver import DEFAULT_DRIVER_BINARY, DriverConfig, SolverOptions, build_driver, effective_repulsor_root, run_driver
from .graph_io import (
    graph_from_curve_vertices,
    graph_to_curve_arrays,
    mapping_metadata,
    reindex_mapping,
    write_graph_curve,
    write_graph_obj,
    write_pinned_vertices,
)
from .metrics import clearance_report, node_distance, read_certificate, read_history_summary
from .models import GraphLayoutResult, RepulsiveLayoutResult
from .protein_examples import build_protein_example, set_special_node_distance
from .render import render_tube_html
from .resampling import ResamplingOptions, resample_curve_network
from .topology import verify_obj_step_sequence


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _segment_edges_from_indices(
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for edge_id in edge_order:
        indices = edge_indices[edge_id]
        edges.extend((int(a), int(b)) for a, b in zip(indices, indices[1:]))
    return tuple(edges)


def _collar_vertex_indices(
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
    depth: int,
) -> set[int]:
    if depth < 0:
        raise ValueError("pin_node_collar_points must be non-negative")
    if depth == 0:
        return set()

    indices_to_pin: set[int] = set()
    for edge_id in edge_order:
        indices = edge_indices[edge_id]
        if len(indices) <= 2:
            continue
        indices_to_pin.update(int(i) for i in indices[1 : 1 + depth])
        indices_to_pin.update(int(i) for i in indices[max(1, len(indices) - 1 - depth) : -1])
    return indices_to_pin


def _curve_total_length(
    vertices: Any,
    edge_indices: dict[str, list[int]],
    edge_order: tuple[str, ...],
) -> float:
    points = read_obj_vertices(vertices) if isinstance(vertices, Path) else vertices
    total = 0.0
    for edge_id in edge_order:
        indices = edge_indices[edge_id]
        for a, b in zip(indices, indices[1:]):
            total += float(((points[int(a)] - points[int(b)]) ** 2).sum() ** 0.5)
    return total


def _bbox_diag(vertices: Any) -> float:
    points = read_obj_vertices(vertices) if isinstance(vertices, Path) else vertices
    if len(points) == 0:
        return 0.0
    return float(((points.max(axis=0) - points.min(axis=0)) ** 2).sum() ** 0.5)


def relax_spatial_graph(
    graph: nx.MultiGraph,
    workspace: Path | str,
    *,
    solver_options: SolverOptions | None = None,
    decimation_options: DecimationOptions | None = None,
    driver_config: DriverConfig | None = None,
    force_build: bool = False,
    keep_workspace: bool = False,
    save_steps: bool = False,
    pin_graph_nodes: bool = True,
    pin_node_collar_points: int = 0,
    simplify_after_layout: bool = True,
    resampling_options: ResamplingOptions | None = None,
    verify_topology: bool = False,
) -> GraphLayoutResult:
    """Relax a KnottedGraph spatial MultiGraph with Repulsor safe steps.

    The input graph must use the standard KnottedGraph geometry convention:
    each node has a ``pos`` 3-vector and each edge optionally has a ``pts``
    polyline. The returned graph preserves node labels, edge keys, and existing
    attributes, but replaces ``pos`` and ``pts`` with the relaxed coordinates.
    """

    workspace = Path(workspace).resolve()
    if workspace.exists() and not keep_workspace:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    driver_config = driver_config or DriverConfig()
    solver_options = solver_options or SolverOptions()

    build_driver(driver_config, force=force_build)

    initial_vertices, mapping = graph_to_curve_arrays(graph, resampling_options=resampling_options)
    if resampling_options is not None and resampling_options.allow_downsample:
        target_counts = {
            edge_id: int(report["target_point_count"])
            for edge_id, report in (mapping.resampling_report or {}).get("edges", {}).items()
            if int(report["target_point_count"]) < len(mapping.edge_vertex_indices[edge_id])
        }
        if target_counts:
            initial_pinned_indices = set(mapping.node_indices.values()) if pin_graph_nodes else set()
            if pin_graph_nodes:
                initial_pinned_indices.update(
                    _collar_vertex_indices(mapping.edge_vertex_indices, mapping.edge_order, pin_node_collar_points)
                )
            initial_decimation = decimate_curve_network(
                initial_vertices,
                mapping.edge_vertex_indices,
                mapping.edge_order,
                pinned_indices=initial_pinned_indices,
                options=DecimationOptions(
                    min_points_per_edge=resampling_options.min_points_per_edge,
                    max_points_per_edge=target_counts,
                    min_clearance=resampling_options.downsample_min_clearance or 1e-5,
                ),
            )
            resampling_report = dict(mapping.resampling_report or {})
            resampling_report["safe_downsampling"] = initial_decimation.report
            initial_vertices = initial_decimation.vertices
            mapping = reindex_mapping(mapping, initial_decimation.old_to_new, initial_decimation.edge_indices)
            mapping = replace(mapping, resampling_report=resampling_report)

    initial_obj = workspace / "initial.obj"
    curve_txt = workspace / "repulsor_curve.txt"
    final_obj = workspace / "final.obj"
    final_simplified_obj = workspace / "final_simplified.obj"
    metadata_json = workspace / "metadata.json"
    history_csv = workspace / "repulsor_history.csv"
    steps_dir = workspace / "certified_steps" if save_steps else None
    clearance_json = workspace / "clearance_report.json"
    topology_verification_json = workspace / "topology_verification.json"
    pinned_vertices_txt = workspace / "pinned_vertices.txt"

    write_graph_obj(initial_vertices, mapping.segment_edges, initial_obj)
    write_graph_curve(initial_vertices, mapping.segment_edges, curve_txt)
    pinned_indices = set(mapping.node_indices.values()) if pin_graph_nodes else set()
    if pin_graph_nodes:
        pinned_indices.update(_collar_vertex_indices(mapping.edge_vertex_indices, mapping.edge_order, pin_node_collar_points))
    pinned_vertices_path = pinned_vertices_txt if pinned_indices else None
    if pinned_vertices_path is not None:
        write_pinned_vertices(pinned_indices, pinned_vertices_path)

    t0 = time.perf_counter()
    process = run_driver(
        input_curve=curve_txt,
        output_obj=final_obj,
        history_csv=history_csv,
        options=solver_options,
        config=driver_config,
        save_steps_dir=steps_dir,
        pinned_vertices=pinned_vertices_path,
    )
    elapsed = time.perf_counter() - t0

    final_vertices = read_obj_vertices(final_obj)
    output_vertices = final_vertices
    output_mapping = mapping
    output_obj = final_obj
    decimation_report = None
    if simplify_after_layout:
        decimation = decimate_curve_network(
            final_vertices,
            mapping.edge_vertex_indices,
            mapping.edge_order,
            pinned_indices=pinned_indices,
            options=decimation_options,
        )
        output_vertices = decimation.vertices
        output_mapping = reindex_mapping(mapping, decimation.old_to_new, decimation.edge_indices)
        output_obj = final_simplified_obj
        decimation_report = decimation.report
        write_graph_obj(output_vertices, output_mapping.segment_edges, output_obj)

    relaxed_graph = graph_from_curve_vertices(graph, output_vertices, output_mapping)

    certificate = read_certificate(history_csv)
    history_summary = read_history_summary(history_csv)
    topology_verification = None
    if verify_topology and steps_dir is not None:
        topology_verification = verify_obj_step_sequence(
            steps_dir,
            epsilon=solver_options.topology_tolerance,
        )
        topology_verification_json.write_text(
            json.dumps(topology_verification, indent=2),
            encoding="utf-8",
        )
        if not topology_verification["verified"]:
            raise RuntimeError(
                "Independent topology verifier found swept segment crossing; "
                f"see {topology_verification_json}"
            )
    clearance = {
        "initial": clearance_report(initial_vertices, mapping.edge_vertex_indices, mapping.edge_order),
        "relaxed": clearance_report(final_vertices, mapping.edge_vertex_indices, mapping.edge_order),
        "final": clearance_report(output_vertices, output_mapping.edge_vertex_indices, output_mapping.edge_order),
        "note": (
            "This is a static clearance sanity report. The topology-preserving "
            "certificate is the per-step swept topology check in the solver history; "
            "Repulsor MaximumSafeStepSize is still used as the first step-size bound. "
            "The decimator accepts only shortcuts whose swept triangle stays "
            "clear of non-adjacent segments by a conservative distance threshold."
        ),
    }
    clearance_json.write_text(json.dumps(clearance, indent=2), encoding="utf-8")
    compactness = {
        "initial_total_length": _curve_total_length(initial_vertices, mapping.edge_vertex_indices, mapping.edge_order),
        "relaxed_total_length": _curve_total_length(final_vertices, mapping.edge_vertex_indices, mapping.edge_order),
        "final_total_length": _curve_total_length(output_vertices, output_mapping.edge_vertex_indices, output_mapping.edge_order),
        "initial_bbox_diag": _bbox_diag(initial_vertices),
        "relaxed_bbox_diag": _bbox_diag(final_vertices),
        "final_bbox_diag": _bbox_diag(output_vertices),
    }

    metadata: dict[str, Any] = {
        "solver": "Repulsor",
        "workspace": str(workspace),
        "repulsor_root": str(effective_repulsor_root(driver_config)),
        "driver_source": str(Path(driver_config.driver_source).resolve()),
        "driver_binary": str(Path(driver_config.driver_binary or DEFAULT_DRIVER_BINARY).resolve()),
        "initial_obj": str(initial_obj),
        "curve_txt": str(curve_txt),
        "repulsor_final_obj": str(final_obj),
        "final_obj": str(output_obj),
        "final_simplified_obj": str(final_simplified_obj) if simplify_after_layout else None,
        "pinned_vertices": str(pinned_vertices_path) if pinned_vertices_path is not None else None,
        "history_csv": str(history_csv),
        "steps_dir": str(steps_dir) if steps_dir is not None else None,
        "clearance_report_json": str(clearance_json),
        "topology_verification_json": (
            str(topology_verification_json) if topology_verification is not None else None
        ),
        "elapsed_seconds": elapsed,
        "certificate": certificate,
        "history_summary": history_summary,
        "topology_verification": topology_verification,
        "clearance_summary": {
            "initial_min_distance": clearance["initial"]["min_non_adjacent_segment_distance"],
            "relaxed_min_distance": clearance["relaxed"]["min_non_adjacent_segment_distance"],
            "final_min_distance": clearance["final"]["min_non_adjacent_segment_distance"],
        },
        "parameters": {
            "steps": solver_options.steps,
            "q": solver_options.q,
            "p": solver_options.p,
            "threads": solver_options.threads,
            "max_time": solver_options.max_time,
            "safe_fraction": solver_options.safe_fraction,
            "max_backtracks": solver_options.max_backtracks,
            "max_iter": solver_options.max_iter,
            "tolerance": solver_options.tolerance,
            "repulsion_weight": solver_options.repulsion_weight,
            "length_weight": solver_options.length_weight,
            "curve_length_floor_weight": solver_options.curve_length_floor_weight,
            "bend_weight": solver_options.bend_weight,
            "tube_radius": solver_options.tube_radius,
            "tube_gap": solver_options.tube_gap,
            "tube_weight": solver_options.tube_weight,
            "pin_special_vertices": not solver_options.free_special_vertices,
            "pin_graph_nodes": pin_graph_nodes,
            "pin_node_collar_points": pin_node_collar_points,
            "simplify_after_layout": simplify_after_layout,
            "verify_topology": verify_topology,
            "resampling_options": _json_safe(resampling_options),
            "decimation_options": _json_safe(decimation_options or DecimationOptions()),
        },
        "resampling": _json_safe(mapping.resampling_report),
        "compactness": compactness,
        "curve_mapping": mapping_metadata(output_mapping),
        "repulsor_curve_mapping": mapping_metadata(mapping),
        "decimation": decimation_report,
        "graph_metadata": _json_safe(dict(graph.graph)),
        "driver_stdout_tail": process.stdout.splitlines()[-20:],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return GraphLayoutResult(graph=relaxed_graph, workspace=workspace, metadata=metadata)


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
    decimation_options: DecimationOptions | None = None,
    driver_config: DriverConfig | None = None,
    force_build: bool = False,
    keep_workspace: bool = False,
    save_steps: bool = False,
    render_html: bool = True,
    simplify_after_layout: bool = True,
    pin_node_collar_points: int = 0,
    resampling_options: ResamplingOptions | None = None,
    verify_topology: bool = False,
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
    network, resampling_report = resample_curve_network(network, resampling_options)

    initial_obj = workspace / "initial.obj"
    curve_txt = workspace / "repulsor_curve.txt"
    final_obj = workspace / "final.obj"
    initial_json = workspace / "initial_layout.json"
    final_json = workspace / "final_layout.json"
    initial_html = workspace / "initial_tubes.html"
    final_html = workspace / "final_tubes.html"
    final_simplified_obj = workspace / "final_simplified.obj"
    final_simplified_json = workspace / "final_simplified_layout.json"
    final_simplified_html = workspace / "final_simplified_tubes.html"
    metadata_json = workspace / "metadata.json"
    history_csv = workspace / "repulsor_history.csv"
    steps_dir = workspace / "certified_steps" if save_steps else None
    clearance_json = workspace / "clearance_report.json"
    topology_verification_json = workspace / "topology_verification.json"
    pinned_vertices_txt = workspace / "pinned_vertices.txt"

    arc_indices = write_curve_obj(network, initial_obj)
    initial_vertices = read_obj_vertices(initial_obj)
    write_repulsor_curve(initial_vertices, arc_indices, network.arc_order, curve_txt)
    pinned_indices = set(range(len(network.node_order))) if not solver_options.free_special_vertices else set()
    if not solver_options.free_special_vertices:
        pinned_indices.update(_collar_vertex_indices(arc_indices, network.arc_order, pin_node_collar_points))
    pinned_vertices_path = pinned_vertices_txt if pinned_indices else None
    if pinned_vertices_path is not None:
        write_pinned_vertices(pinned_indices, pinned_vertices_path)

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
        "repulsion_weight": solver_options.repulsion_weight,
        "length_weight": solver_options.length_weight,
        "curve_length_floor_weight": solver_options.curve_length_floor_weight,
        "bend_weight": solver_options.bend_weight,
        "tube_radius": solver_options.tube_radius,
        "tube_gap": solver_options.tube_gap,
        "tube_weight": solver_options.tube_weight,
        "pin_special_vertices": not solver_options.free_special_vertices,
        "total_arc_points": total_arc_points,
        "resampling_options": _json_safe(resampling_options),
        "node_distance_original": original_node_distance,
        "node_distance_target": target_distance,
        "node_distance_applied": applied_node_distance,
        "node_distance_scale": node_distance_scale,
        "simplify_after_layout": simplify_after_layout,
        "verify_topology": verify_topology,
        "pin_node_collar_points": pin_node_collar_points,
        "decimation_options": _json_safe(decimation_options or DecimationOptions()),
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
        pinned_vertices=pinned_vertices_path,
    )
    elapsed = time.perf_counter() - t0

    final_vertices = read_obj_vertices(final_obj)
    final_network = network_from_vertices(network, final_vertices, arc_indices)
    params["node_distance_final"] = node_distance(final_network.node_positions, final_network.node_order)
    write_layout_json(final_network, params, final_json)
    if render_html:
        render_tube_html(final_network, final_html, f"{network.name} Repulsor safe-step")

    decimation_report = None
    simplified_arc_indices = None
    if simplify_after_layout:
        decimation = decimate_curve_network(
            final_vertices,
            arc_indices,
            network.arc_order,
            pinned_indices=pinned_indices,
            options=decimation_options,
        )
        simplified_arc_indices = decimation.edge_indices
        simplified_network = network_from_vertices(network, decimation.vertices, simplified_arc_indices)
        params["node_distance_simplified"] = node_distance(
            simplified_network.node_positions,
            simplified_network.node_order,
        )
        write_graph_obj(
            decimation.vertices,
            _segment_edges_from_indices(simplified_arc_indices, network.arc_order),
            final_simplified_obj,
        )
        write_layout_json(simplified_network, params, final_simplified_json)
        if render_html:
            render_tube_html(
                simplified_network,
                final_simplified_html,
                f"{network.name} simplified safe-step",
            )
        decimation_report = decimation.report

    certificate = read_certificate(history_csv)
    history_summary = read_history_summary(history_csv)
    topology_verification = None
    if verify_topology and steps_dir is not None:
        topology_verification = verify_obj_step_sequence(
            steps_dir,
            epsilon=solver_options.topology_tolerance,
        )
        topology_verification_json.write_text(
            json.dumps(topology_verification, indent=2),
            encoding="utf-8",
        )
        if not topology_verification["verified"]:
            raise RuntimeError(
                "Independent topology verifier found swept segment crossing; "
                f"see {topology_verification_json}"
            )
    clearance = {
        "initial": clearance_report(initial_vertices, arc_indices, network.arc_order),
        "final": clearance_report(final_vertices, arc_indices, network.arc_order),
        "final_simplified": (
            clearance_report(
                decimation.vertices,
                simplified_arc_indices,
                network.arc_order,
            )
            if simplify_after_layout and simplified_arc_indices is not None
            else None
        ),
        "note": (
            "This is a static clearance sanity report. The topology-preserving "
            "certificate is the per-step swept topology check in the solver history; "
            "Repulsor MaximumSafeStepSize is still used as the first step-size bound. "
            "The decimator accepts only shortcuts whose swept triangle stays "
            "clear of non-adjacent segments by a conservative distance threshold."
        ),
    }
    clearance_json.write_text(json.dumps(clearance, indent=2), encoding="utf-8")
    compactness = {
        "initial_total_length": _curve_total_length(initial_vertices, arc_indices, network.arc_order),
        "final_total_length": _curve_total_length(final_vertices, arc_indices, network.arc_order),
        "initial_bbox_diag": _bbox_diag(initial_vertices),
        "final_bbox_diag": _bbox_diag(final_vertices),
        "final_simplified_total_length": (
            _curve_total_length(decimation.vertices, simplified_arc_indices, network.arc_order)
            if simplify_after_layout and simplified_arc_indices is not None
            else None
        ),
        "final_simplified_bbox_diag": (
            _bbox_diag(decimation.vertices)
            if simplify_after_layout and simplified_arc_indices is not None
            else None
        ),
    }

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
        "final_simplified_obj": str(final_simplified_obj) if simplify_after_layout else None,
        "initial_layout_json": str(initial_json),
        "final_layout_json": str(final_json),
        "final_simplified_layout_json": str(final_simplified_json) if simplify_after_layout else None,
        "initial_html": str(initial_html) if render_html else None,
        "final_html": str(final_html) if render_html else None,
        "final_simplified_html": (
            str(final_simplified_html) if render_html and simplify_after_layout else None
        ),
        "output_html": (
            str(final_simplified_html)
            if render_html and simplify_after_layout
            else str(final_html) if render_html else None
        ),
        "history_csv": str(history_csv),
        "steps_dir": str(steps_dir) if steps_dir is not None else None,
        "pinned_vertices": str(pinned_vertices_path) if pinned_vertices_path is not None else None,
        "clearance_report_json": str(clearance_json),
        "topology_verification_json": (
            str(topology_verification_json) if topology_verification is not None else None
        ),
        "elapsed_seconds": elapsed,
        "certificate": certificate,
        "history_summary": history_summary,
        "topology_verification": topology_verification,
        "clearance_summary": {
            "initial_min_distance": clearance["initial"]["min_non_adjacent_segment_distance"],
            "final_min_distance": clearance["final"]["min_non_adjacent_segment_distance"],
            "final_simplified_min_distance": (
                clearance["final_simplified"]["min_non_adjacent_segment_distance"]
                if clearance["final_simplified"] is not None
                else None
            ),
        },
        "arc_specs": network.arc_specs,
        "arc_vertex_indices": arc_indices,
        "simplified_arc_vertex_indices": simplified_arc_indices,
        "decimation": decimation_report,
        "resampling": _json_safe(resampling_report),
        "compactness": compactness,
        "parameters": params,
        "protein_metadata": network.metadata,
        "driver_stdout_tail": process.stdout.splitlines()[-20:],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return RepulsiveLayoutResult(workspace=workspace, metadata=metadata)
