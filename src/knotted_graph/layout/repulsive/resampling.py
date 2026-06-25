from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .models import CurveNetwork


@dataclass(frozen=True)
class ResamplingOptions:
    """Polyline resampling before Repulsor optimization."""

    target_segment_length: float | None = None
    points_per_edge: int | None = None
    min_points_per_edge: int = 2
    max_points_per_edge: int | None = None
    allow_downsample: bool = False
    downsample_min_clearance: float | None = None


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def max_segment_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())


def resample_polyline_to_count(points: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(points) < 2:
        raise ValueError("points must contain at least two points")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total < 1e-12:
        return np.repeat(points[:1], n_points, axis=0)

    targets = np.linspace(0.0, total, n_points)
    out: list[np.ndarray] = []
    j = 0
    for target in targets:
        while j + 1 < len(cumulative) and cumulative[j + 1] < target:
            j += 1
        if j + 1 >= len(cumulative):
            out.append(points[-1])
            continue
        a = float(cumulative[j])
        b = float(cumulative[j + 1])
        alpha = 0.0 if b - a < 1e-12 else (target - a) / (b - a)
        out.append((1.0 - alpha) * points[j] + alpha * points[j + 1])
    return np.asarray(out, dtype=float)


def requested_point_count_for_options(points: np.ndarray, options: ResamplingOptions | None) -> int:
    if options is None:
        return len(points)
    if options.min_points_per_edge < 2:
        raise ValueError("min_points_per_edge must be at least 2")
    if options.max_points_per_edge is not None and options.max_points_per_edge < options.min_points_per_edge:
        raise ValueError("max_points_per_edge must be greater than or equal to min_points_per_edge")
    if options.points_per_edge is not None and options.points_per_edge < 2:
        raise ValueError("points_per_edge must be at least 2")
    if options.target_segment_length is not None and options.target_segment_length <= 0:
        raise ValueError("target_segment_length must be positive")

    if options.points_per_edge is not None:
        count = int(options.points_per_edge)
    elif options.target_segment_length is not None:
        count = int(math.ceil(polyline_length(points) / options.target_segment_length)) + 1
    else:
        count = len(points)

    count = max(count, int(options.min_points_per_edge))
    if options.max_points_per_edge is not None:
        count = min(count, int(options.max_points_per_edge))
    return count


def point_count_for_options(points: np.ndarray, options: ResamplingOptions | None) -> int:
    requested_count = requested_point_count_for_options(points, options)
    return max(len(points), requested_count)


def resample_polyline_for_options(
    points: np.ndarray,
    options: ResamplingOptions | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=float)
    requested_count = requested_point_count_for_options(points, options)
    n_points = point_count_for_options(points, options)
    resampled = resample_polyline_to_count(points, n_points) if n_points != len(points) else points.copy()
    report = {
        "point_count_before": int(len(points)),
        "target_point_count": int(requested_count),
        "point_count_after": int(len(resampled)),
        "unsafe_downsampled": False,
        "safe_downsample_requested": bool(requested_count < len(points)),
        "safe_downsample_allowed": bool(options.allow_downsample) if options is not None else False,
        "length": polyline_length(points),
        "max_segment_length_before": max_segment_length(points),
        "max_segment_length_after": max_segment_length(resampled),
    }
    return resampled, report


def resample_curve_network(
    network: CurveNetwork,
    options: ResamplingOptions | None,
) -> tuple[CurveNetwork, dict[str, Any]]:
    if options is None:
        return network, {"enabled": False}

    arc_polylines: dict[str, np.ndarray] = {}
    edge_reports: dict[str, Any] = {}
    for arc_name in network.arc_order:
        resampled, report = resample_polyline_for_options(network.arc_polylines[arc_name], options)
        arc_polylines[arc_name] = resampled
        edge_reports[arc_name] = report

    node_positions = dict(network.node_positions)
    if len(network.node_order) == 2:
        start_node, end_node = network.node_order
        start = np.asarray(node_positions[start_node], dtype=float)
        end = np.asarray(node_positions[end_node], dtype=float)
        for arc_name in network.arc_order:
            arc_polylines[arc_name][0] = start
            arc_polylines[arc_name][-1] = end

    resampled_network = CurveNetwork(
        name=network.name,
        node_order=network.node_order,
        node_positions=node_positions,
        arc_order=network.arc_order,
        arc_polylines=arc_polylines,
        arc_specs=network.arc_specs,
        node_colors=network.node_colors,
        arc_colors=network.arc_colors,
        metadata=dict(network.metadata),
    )
    safe_downsampling_report = None
    if options.allow_downsample:
        target_counts = {
            arc_name: int(report["target_point_count"])
            for arc_name, report in edge_reports.items()
            if int(report["target_point_count"]) < int(report["point_count_after"])
        }
        if target_counts:
            if len(network.node_order) != 2:
                raise ValueError("safe downsampling for CurveNetwork currently expects exactly two graph nodes")
            from .curve_io import network_from_vertices, network_to_vertices
            from .decimation import DecimationOptions, decimate_curve_network

            vertices, edge_indices = network_to_vertices(resampled_network)
            decimation = decimate_curve_network(
                vertices,
                edge_indices,
                resampled_network.arc_order,
                pinned_indices=set(range(len(resampled_network.node_order))),
                options=DecimationOptions(
                    min_points_per_edge=options.min_points_per_edge,
                    max_points_per_edge=target_counts,
                    min_clearance=options.downsample_min_clearance or 1e-5,
                ),
            )
            resampled_network = network_from_vertices(resampled_network, decimation.vertices, decimation.edge_indices)
            safe_downsampling_report = decimation.report
            for arc_name in resampled_network.arc_order:
                edge_reports[arc_name]["point_count_after_safe_downsample"] = int(
                    len(resampled_network.arc_polylines[arc_name])
                )
                edge_reports[arc_name]["safe_downsampled"] = bool(
                    edge_reports[arc_name]["point_count_after_safe_downsample"]
                    < edge_reports[arc_name]["point_count_after"]
                )

    report = {
        "enabled": True,
        "options": asdict(options),
        "edges": edge_reports,
        "safe_downsampling": safe_downsampling_report,
    }
    return resampled_network, report
