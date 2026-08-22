"""Reusable Bloch-vector deformation scans for ``NodalSkeleton``.

This is intentionally separate from generic S3/R3 knot fields: an arbitrary
analytic knot field is not automatically a periodic Brillouin-zone model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp

BlochVector = tuple[sp.Expr, sp.Expr, sp.Expr]
BlochVectorFactory = Callable[[float], Sequence[sp.Expr]]


def _vector(value: Sequence[sp.Expr]) -> BlochVector:
    if len(value) != 3:
        raise ValueError("a Bloch vector must contain exactly three components")
    return tuple(sp.sympify(component) for component in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class NodalBlochPath:
    start: BlochVectorFactory
    end: BlochVectorFactory
    start_name: str = "start"
    end_name: str = "end"

    def endpoints(self, gamma: float) -> tuple[BlochVector, BlochVector]:
        return _vector(self.start(float(gamma))), _vector(self.end(float(gamma)))

    def at(self, gamma: float, lam: float) -> BlochVector:
        lam = float(lam)
        if not 0 <= lam <= 1:
            raise ValueError("lam must lie in [0, 1]")
        start, end = self.endpoints(gamma)
        return tuple(
            sp.expand((1 - lam) * left + lam * right)
            for left, right in zip(start, end)
        )  # type: ignore[return-value]

    def at_components(self, gamma: float, weights: Sequence[float]) -> BlochVector:
        if len(weights) != 3:
            raise ValueError("weights must contain (lambda_x, lambda_y, lambda_z)")
        weights = tuple(float(value) for value in weights)
        if any(value < 0 or value > 1 for value in weights):
            raise ValueError("all component weights must lie in [0, 1]")
        start, end = self.endpoints(gamma)
        return tuple(
            sp.expand((1 - weight) * left + weight * right)
            for left, right, weight in zip(start, end, weights)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class NodalPhaseRecord:
    lam: float
    gamma: float
    yamada: sp.Expr | None
    phase_signature: str
    error: str | None = None


@dataclass
class NodalPhaseScanResult:
    lambdas: np.ndarray
    gammas: np.ndarray
    records: list[NodalPhaseRecord]

    def record_grid(self) -> np.ndarray:
        lookup = {(record.gamma, record.lam): record for record in self.records}
        grid = np.empty((len(self.gammas), len(self.lambdas)), dtype=object)
        for row, gamma in enumerate(self.gammas):
            for column, lam in enumerate(self.lambdas):
                grid[row, column] = lookup[(float(gamma), float(lam))]
        return grid

    def phase_grid(self) -> tuple[np.ndarray, dict[int, str]]:
        signatures = sorted({record.phase_signature for record in self.records})
        ids = {signature: index for index, signature in enumerate(signatures)}
        labels = np.empty((len(self.gammas), len(self.lambdas)), dtype=int)
        for row, records in enumerate(self.record_grid()):
            for column, record in enumerate(records):
                labels[row, column] = ids[record.phase_signature]
        return labels, {index: signature for signature, index in ids.items()}

    def transition_intervals(self) -> list[dict]:
        grid = self.record_grid()
        changes: list[dict] = []
        for row, gamma in enumerate(self.gammas):
            for column in range(1, len(self.lambdas)):
                left, right = grid[row, column - 1], grid[row, column]
                if left.phase_signature != right.phase_signature:
                    changes.append({
                        "gamma": float(gamma),
                        "lambda_left": float(self.lambdas[column - 1]),
                        "lambda_right": float(self.lambdas[column]),
                        "phase_left": left.phase_signature,
                        "phase_right": right.phase_signature,
                    })
        return changes


def _signature(polynomial: sp.Expr) -> str:
    try:
        canonical = sp.factor(sp.together(sp.expand(polynomial)))
    except Exception:
        canonical = sp.expand(polynomial)
    return "yamada:" + sp.srepr(canonical)


class NodalPhaseScan:
    def __init__(
        self,
        path: NodalBlochPath,
        *,
        lambdas: Sequence[float],
        gammas: Sequence[float],
        dimension: int = 96,
        span=None,
        normalize_yamada: bool = True,
        yamada_variable: sp.Symbol | None = None,
        yamada_options: dict | None = None,
        continue_on_error: bool = True,
    ) -> None:
        self.path = path
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.gammas = np.asarray(gammas, dtype=float)
        if self.lambdas.ndim != 1 or len(self.lambdas) == 0:
            raise ValueError("lambdas must be non-empty and one-dimensional")
        if self.gammas.ndim != 1 or len(self.gammas) == 0:
            raise ValueError("gammas must be non-empty and one-dimensional")
        if np.any((self.lambdas < 0) | (self.lambdas > 1)):
            raise ValueError("all lambda samples must lie in [0, 1]")
        self.dimension, self.span = int(dimension), span
        self.normalize_yamada = bool(normalize_yamada)
        self.yamada_variable = yamada_variable or sp.Symbol("A")
        self.yamada_options = dict(yamada_options or {})
        self.continue_on_error = bool(continue_on_error)

    def run(self) -> NodalPhaseScanResult:
        from knotted_graph.applications.nodal.skeleton import NodalSkeleton

        records: list[NodalPhaseRecord] = []
        for gamma in self.gammas:
            start, end = self.path.endpoints(float(gamma))
            for lam in self.lambdas:
                vector = tuple(
                    sp.expand((1 - float(lam)) * left + float(lam) * right)
                    for left, right in zip(start, end)
                )
                polynomial = None
                error = None
                try:
                    kwargs = {"char": vector, "dimension": self.dimension}
                    if self.span is not None:
                        kwargs["span"] = self.span
                    skeleton = NodalSkeleton(**kwargs)
                    polynomial = skeleton.yamada_polynomial(
                        variable=self.yamada_variable,
                        normalize=self.normalize_yamada,
                        **self.yamada_options,
                    )
                    signature = _signature(polynomial)
                except Exception as exc:
                    if not self.continue_on_error:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    signature = "error:" + error
                records.append(NodalPhaseRecord(
                    float(lam), float(gamma), polynomial, signature, error
                ))
        return NodalPhaseScanResult(self.lambdas.copy(), self.gammas.copy(), records)


__all__ = [
    "NodalBlochPath", "NodalPhaseRecord", "NodalPhaseScan", "NodalPhaseScanResult",
]
