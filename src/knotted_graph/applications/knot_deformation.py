"""Topological scans of analytic-knot-field deformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.inputs.knot_field import DEFAULT_SPAN, Span3D
from knotted_graph.inputs.knot_path import KnotFunctionPath


@dataclass(frozen=True)
class KnotDeformationRecord:
    lam: float
    radius: float
    nodes: int
    edges: int
    components: int
    cycle_rank: int
    degree_sequence: tuple[int, ...]
    yamada: sp.Expr | None
    phase_signature: str
    error: str | None = None


@dataclass
class KnotDeformationScanResult:
    lambdas: np.ndarray
    radii: np.ndarray
    records: list[KnotDeformationRecord]

    def record_grid(self) -> np.ndarray:
        lookup = {(record.lam, record.radius): record for record in self.records}
        grid = np.empty((len(self.radii), len(self.lambdas)), dtype=object)
        for row, radius in enumerate(self.radii):
            for column, lam in enumerate(self.lambdas):
                grid[row, column] = lookup[(float(lam), float(radius))]
        return grid

    def phase_grid(self) -> tuple[np.ndarray, dict[int, str]]:
        signatures = sorted({record.phase_signature for record in self.records})
        ids = {signature: index for index, signature in enumerate(signatures)}
        grid = np.empty((len(self.radii), len(self.lambdas)), dtype=int)
        for row, records in enumerate(self.record_grid()):
            for column, record in enumerate(records):
                grid[row, column] = ids[record.phase_signature]
        return grid, {index: signature for signature, index in ids.items()}

    def transition_points(self) -> list[dict]:
        grid = self.record_grid()
        transitions: list[dict] = []
        for row, radius in enumerate(self.radii):
            for column in range(1, len(self.lambdas)):
                left = grid[row, column - 1]
                right = grid[row, column]
                if left.phase_signature != right.phase_signature:
                    transitions.append({
                        "radius": float(radius),
                        "lambda_left": float(self.lambdas[column - 1]),
                        "lambda_right": float(self.lambdas[column]),
                        "phase_left": left.phase_signature,
                        "phase_right": right.phase_signature,
                    })
        return transitions

    def plot_phase_diagram(self, ax=None):
        import matplotlib.pyplot as plt
        labels, legend = self.phase_grid()
        if ax is None:
            _, ax = plt.subplots()
        image = ax.imshow(
            labels,
            origin="lower",
            aspect="auto",
            extent=(
                float(self.lambdas[0]), float(self.lambdas[-1]),
                float(self.radii[0]), float(self.radii[-1]),
            ),
            interpolation="nearest",
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"level radius $\epsilon$")
        ax.set_title("Analytic-knot-field topology scan")
        image._knotted_graph_phase_legend = legend
        return ax


def _graph_signature(graph: nx.MultiGraph) -> tuple:
    components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
    degree_sequence = tuple(sorted((degree for _, degree in graph.degree()), reverse=True))
    return (
        graph.number_of_nodes(), graph.number_of_edges(), components,
        cycle_rank, degree_sequence,
    )


def _phase_signature(graph: nx.MultiGraph, yamada: sp.Expr | None, error: str | None) -> str:
    if yamada is not None:
        return "yamada:" + sp.srepr(sp.expand(yamada))
    if error is not None:
        return "error:" + error
    return "graph:" + repr(_graph_signature(graph))


class KnotDeformationScan:
    """Sample a two-parameter ``(lambda, radius)`` knot-field deformation."""

    def __init__(
        self,
        path: KnotFunctionPath,
        *,
        lambdas: Sequence[float],
        radii: Sequence[float],
        span: Span3D = DEFAULT_SPAN,
        dimension: int | Sequence[int] = 96,
        invariant: str | None = None,
        yamada_variable: sp.Symbol | None = None,
        yamada_options: dict | None = None,
        graph_options: dict | None = None,
        continue_on_error: bool = True,
    ) -> None:
        self.path = path
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        if self.lambdas.ndim != 1 or len(self.lambdas) == 0:
            raise ValueError("lambdas must be a non-empty one-dimensional sequence")
        if self.radii.ndim != 1 or len(self.radii) == 0:
            raise ValueError("radii must be a non-empty one-dimensional sequence")
        if np.any((self.lambdas < 0) | (self.lambdas > 1)):
            raise ValueError("all lambda samples must lie in [0, 1]")
        if np.any(self.radii <= 0):
            raise ValueError("all radii must be positive")
        if invariant not in (None, "yamada"):
            raise ValueError("invariant must be None or 'yamada'")
        self.span = span
        self.dimension = dimension
        self.invariant = invariant
        self.yamada_variable = yamada_variable or sp.Symbol("A")
        self.yamada_options = dict(yamada_options or {})
        self.graph_options = dict(graph_options or {})
        self.continue_on_error = bool(continue_on_error)

    def run(self) -> KnotDeformationScanResult:
        records: list[KnotDeformationRecord] = []
        for lam in self.lambdas:
            field = self.path.at(float(lam))
            sample = field.sample(span=self.span, dimension=self.dimension)
            for radius in self.radii:
                graph = nx.MultiGraph()
                yamada = None
                error = None
                try:
                    graph = field.to_spatial_graph(
                        float(radius), sample=sample, **self.graph_options
                    )
                    if self.invariant == "yamada":
                        from knotted_graph.projection import compute_yamada_polynomial
                        yamada = compute_yamada_polynomial(
                            graph, self.yamada_variable, **self.yamada_options
                        )
                except Exception as exc:
                    if not self.continue_on_error:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                nodes, edges, components, cycle_rank, degree_sequence = _graph_signature(graph)
                records.append(KnotDeformationRecord(
                    lam=float(lam), radius=float(radius), nodes=nodes, edges=edges,
                    components=components, cycle_rank=cycle_rank,
                    degree_sequence=degree_sequence, yamada=yamada,
                    phase_signature=_phase_signature(graph, yamada, error), error=error,
                ))
        return KnotDeformationScanResult(
            lambdas=self.lambdas.copy(), radii=self.radii.copy(), records=records
        )


__all__ = [
    "KnotDeformationRecord", "KnotDeformationScan", "KnotDeformationScanResult",
]
