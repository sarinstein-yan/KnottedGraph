"""Gauge-fixed function-space homotopies between analytic knot fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .knot_field import KnotFunction, Span3D, sample_s3


@dataclass(frozen=True)
class PathGauge:
    start_scale: complex
    end_scale: complex
    end_phase: complex
    sample_domain: str
    sample_count: int
    overlap_after_alignment: complex


class KnotFunctionPath:
    r"""RMS-normalized, global-phase-aligned linear field homotopy.

    Gauge fixing removes two elementary freedoms of a zero-set representative,
    but does not make the path canonical. Intermediate topology remains a
    property of the selected representatives and homotopy.
    """

    def __init__(
        self,
        start: KnotFunction,
        end: KnotFunction,
        *,
        normalize: bool = True,
        phase_align: bool = True,
        sample_count: int = 4096,
        seed: int = 0,
        r3_alignment_span: Span3D = ((-2.0, 2.0),) * 3,
    ) -> None:
        if sample_count < 32:
            raise ValueError("sample_count must be at least 32")
        self.start, self.end = start, end
        if start.s3_evaluator is not None and end.s3_evaluator is not None:
            u, v = sample_s3(sample_count, seed=seed)
            f0, f1 = start.evaluate_s3(u, v), end.evaluate_s3(u, v)
            domain = "S3"
        else:
            rng = np.random.default_rng(seed)
            points = np.column_stack([
                rng.uniform(bounds[0], bounds[1], sample_count)
                for bounds in r3_alignment_span
            ])
            f0 = start(points[:, 0], points[:, 1], points[:, 2])
            f1 = end(points[:, 0], points[:, 1], points[:, 2])
            domain = "R3"
        norm0 = float(np.sqrt(np.mean(np.abs(f0) ** 2)))
        norm1 = float(np.sqrt(np.mean(np.abs(f1) ** 2)))
        if min(norm0, norm1) <= np.finfo(float).eps:
            raise ValueError("cannot normalize a field vanishing on the alignment sample")
        scale0 = 1.0 / norm0 if normalize else 1.0
        scale1 = 1.0 / norm1 if normalize else 1.0
        overlap = complex(np.vdot(scale0 * f0, scale1 * f1))
        phase = (
            np.exp(-1j * np.angle(overlap))
            if phase_align and abs(overlap) > 0 else 1.0 + 0j
        )
        overlap_after = complex(np.vdot(scale0 * f0, phase * scale1 * f1))
        self.gauge = PathGauge(
            complex(scale0), complex(scale1), complex(phase), domain,
            sample_count, overlap_after,
        )

    def at(self, lam: float) -> KnotFunction:
        lam = float(lam)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must lie in [0, 1]")
        a = (1.0 - lam) * self.gauge.start_scale
        b = lam * self.gauge.end_scale * self.gauge.end_phase
        same_chart = abs(self.start.s3_chart_angle - self.end.s3_chart_angle) < 1e-14
        if (
            self.start.semiholomorphic is not None
            and self.end.semiholomorphic is not None
            and same_chart
        ):
            polynomial = self.start.semiholomorphic.add_scaled(
                self.end.semiholomorphic, self_scale=a, other_scale=b
            )
            return KnotFunction.from_semiholomorphic(
                polynomial,
                name=f"{self.start.name}->{self.end.name}@{lam:.6g}",
                chart_angle=self.start.s3_chart_angle,
                metadata=self._metadata(lam),
            )

        def evaluator(x, y, z):
            return a * self.start(x, y, z) + b * self.end(x, y, z)

        has_s3 = self.start.s3_evaluator is not None and self.end.s3_evaluator is not None

        def s3_evaluator(u, v):
            return a * self.start.evaluate_s3(u, v) + b * self.end.evaluate_s3(u, v)

        return KnotFunction(
            evaluator=evaluator,
            name=f"{self.start.name}->{self.end.name}@{lam:.6g}",
            s3_evaluator=s3_evaluator if has_s3 else None,
            metadata=self._metadata(lam),
        )

    def _metadata(self, lam: float) -> dict:
        return {
            "construction": "gauge_fixed_linear_homotopy",
            "lambda": lam,
            "start": self.start.name,
            "end": self.end.name,
            "gauge": self.gauge,
        }


__all__ = ["KnotFunctionPath", "PathGauge"]
