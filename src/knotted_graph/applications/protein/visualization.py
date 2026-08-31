"""Static figures for protein crosslink topology analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from knotted_graph.core.embedding import oriented_edge_polyline

from .models import ProteinTopologyAnalysis
from .null_models import NullRobustnessComparison


CROSSLINK_COLORS = {
    "disulfide": "#d23b31",
    "covalent": "#8e44ad",
    "metal_coordination": "#2e9f55",
    "other": "#555555",
}


def _save(fig, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def plot_protein_graph_3d(
    graph: nx.MultiGraph,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
):
    """Plot backbone and physical crosslinks in their input coordinates."""

    fig = plt.figure(figsize=(7, 5))
    axis = fig.add_subplot(111, projection="3d")
    seen_labels: set[str] = set()
    all_points = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        points = oriented_edge_polyline(graph, u, v, key, data)
        all_points.append(points)
        edge_kind = str(data.get("edge_kind", "backbone"))
        if edge_kind == "crosslink":
            crosslink_type = str(data.get("crosslink_type", "other"))
            color = CROSSLINK_COLORS.get(crosslink_type, CROSSLINK_COLORS["other"])
            label = crosslink_type.replace("_", " ")
            linewidth = 2.2
        else:
            color = "#2864b7"
            label = "backbone"
            linewidth = 1.2
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=0.9,
            label=label if label not in seen_labels else None,
        )
        seen_labels.add(label)
    if all_points:
        stacked = np.vstack(all_points)
        spans = np.ptp(stacked, axis=0)
        center = stacked.mean(axis=0)
        radius = max(float(spans.max()) / 2.0, 1e-6)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_title(title or str(graph.graph.get("input_id", "protein crosslinks")))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    if seen_labels:
        axis.legend(loc="best", fontsize=8)
    _save(fig, output_path)
    return fig, axis


def plot_edge_importance(
    analysis: ProteinTopologyAnalysis,
    *,
    output_path: str | Path | None = None,
):
    """Plot single-edge topological-information indicators Xi."""

    labels = [record.crosslink_id for record in analysis.singles]
    values = [
        np.nan if record.changed is None else int(record.changed)
        for record in analysis.singles
    ]
    colors = [
        "#c0392b" if value == 1 else "#7f8c8d" if value == 0 else "#f1c40f"
        for value in values
    ]
    width = max(7.0, 0.5 * len(labels))
    fig, axis = plt.subplots(figsize=(width, 4.2))
    axis.bar(np.arange(len(labels)), np.nan_to_num(values, nan=0.5), color=colors)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks([0, 1], ["unchanged", "changed"])
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("single-edge indicator $X_i$")
    axis.set_title(f"{analysis.input_id}: topological-information-carrying crosslinks")
    fig.tight_layout()
    _save(fig, output_path)
    return fig, axis


def plot_pair_synergy_heatmap(
    analysis: ProteinTopologyAnalysis,
    *,
    output_path: str | Path | None = None,
):
    """Plot pair synergy scores, highlighting cooperative deletions."""

    labels = list(analysis.crosslink_ids)
    index = {crosslink_id: position for position, crosslink_id in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan)
    np.fill_diagonal(matrix, 0.0)
    for record in analysis.pairs:
        i = index[record.crosslink_i]
        j = index[record.crosslink_j]
        if record.synergy_score is not None:
            matrix[i, j] = matrix[j, i] = record.synergy_score
    size = max(5.0, 0.55 * len(labels))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_title(f"{analysis.input_id}: crosslink-pair synergy")
    fig.colorbar(image, ax=axis, label="pair change minus strongest single change")
    fig.tight_layout()
    _save(fig, output_path)
    return fig, axis


def plot_robustness_distribution(
    analyses: Iterable[ProteinTopologyAnalysis],
    *,
    output_path: str | Path | None = None,
):
    """Plot the dataset distribution of single-edge robustness R1."""

    records = [
        (analysis.input_id, analysis.robustness_r1)
        for analysis in analyses
        if analysis.robustness_r1 is not None
    ]
    labels = [label for label, _ in records]
    values = [float(value) for _, value in records]
    fig, axis = plt.subplots(figsize=(max(7.0, 0.5 * len(labels)), 4.2))
    axis.bar(np.arange(len(labels)), values, color="#2c7fb8")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("robustness $R_1 = 1-f_{top}$")
    axis.set_title("Protein crosslink-topology robustness")
    fig.tight_layout()
    _save(fig, output_path)
    return fig, axis


def plot_natural_vs_null(
    comparison: NullRobustnessComparison,
    *,
    output_path: str | Path | None = None,
):
    """Plot natural robustness against a rewired null distribution."""

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    bins = min(max(5, int(np.sqrt(len(comparison.null_values)))), 30)
    axis.hist(comparison.null_values, bins=bins, color="#9ecae1", edgecolor="white")
    axis.axvline(
        comparison.natural_robustness_r1,
        color="#c0392b",
        linewidth=2,
        label=f"natural R1={comparison.natural_robustness_r1:.3f}",
    )
    axis.axvline(
        comparison.null_mean,
        color="#2c3e50",
        linestyle="--",
        label=f"null mean={comparison.null_mean:.3f}",
    )
    axis.set_xlabel("robustness R1")
    axis.set_ylabel("null replicate count")
    axis.set_title("Natural protein versus rewired crosslink null model")
    axis.legend()
    fig.tight_layout()
    _save(fig, output_path)
    return fig, axis


def close_figure(figure) -> None:
    plt.close(figure)
