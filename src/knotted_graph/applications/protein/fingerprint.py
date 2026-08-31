"""Canonical, cached Yamada fingerprints for protein spatial graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time
from typing import Any

import networkx as nx
import sympy as sp

from knotted_graph.projection import select_projection

from .graph import embedding_hash
from .models import FingerprintRecord


def canonical_laurent_terms(
    expression: sp.Expr,
    variable: sp.Symbol,
) -> tuple[tuple[int, str], ...]:
    """Convert an exact univariate Laurent polynomial to stable terms."""

    expanded = sp.expand(expression)
    if expanded == 0:
        return ()
    coefficients: dict[int, sp.Expr] = {}
    for term in sp.Add.make_args(expanded):
        coefficient, exponent = term.as_coeff_exponent(variable)
        if variable in coefficient.free_symbols or not exponent.is_Integer:
            raise ValueError(
                f"Expression is not a univariate Laurent polynomial in {variable}: "
                f"{expression}"
            )
        exponent_int = int(exponent)
        coefficients[exponent_int] = sp.expand(
            coefficients.get(exponent_int, sp.Integer(0)) + coefficient
        )
    return tuple(
        (exponent, sp.sstr(coefficient))
        for exponent, coefficient in sorted(coefficients.items())
        if coefficient != 0
    )


def fingerprint_id(terms: tuple[tuple[int, str], ...]) -> str:
    payload = json.dumps(terms, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FingerprintSettings:
    variable_name: str = "A"
    rotation_angles: tuple[float, float, float] | None = None
    rotation_order: str = "ZYX"
    num_rotation_samples: int = 10
    max_crossings: int | None = 16
    normalize: bool = True
    n_jobs: int = -1
    method: str = "negami"

    def __post_init__(self) -> None:
        if not self.variable_name:
            raise ValueError("variable_name must not be empty")
        if self.num_rotation_samples <= 0:
            raise ValueError("num_rotation_samples must be positive")
        if self.rotation_angles is not None and len(self.rotation_angles) != 3:
            raise ValueError("rotation_angles must contain exactly three values")
        if self.max_crossings is not None and self.max_crossings < 0:
            raise ValueError("max_crossings must be non-negative or None")


class FingerprintComplexityError(RuntimeError):
    """Raised before exact evaluation when a projection exceeds its safety cap."""


def _package_version() -> str:
    try:
        return version("knotted_graph")
    except PackageNotFoundError:
        return "source"


class FingerprintComputer:
    """Compute Yamada fingerprints with deterministic on-disk caching."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        settings: FingerprintSettings | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.settings = settings or FingerprintSettings()

    def _cache_key(
        self,
        graph_hash: str,
        removed_crosslink_ids: tuple[str, ...],
    ) -> str:
        payload = {
            "embedding_hash": graph_hash,
            "removed_crosslink_ids": removed_crosslink_ids,
            "settings": asdict(self.settings),
            "package_version": _package_version(),
            "cache_schema": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    def compute(
        self,
        graph: nx.MultiGraph,
        *,
        removed_crosslink_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> FingerprintRecord:
        graph_hash = embedding_hash(graph)
        removed = tuple(sorted(str(value) for value in removed_crosslink_ids))
        cache_key = self._cache_key(graph_hash, removed)
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.exists():
            return FingerprintRecord.from_dict(
                json.loads(cache_path.read_text()),
                from_cache=True,
            )

        started = time.perf_counter()
        if graph.number_of_edges() == 0:
            terms = ((0, "1"),)
            record = FingerprintRecord(
                cache_key=cache_key,
                embedding_hash=graph_hash,
                status="ok",
                polynomial="1",
                canonical_terms=terms,
                fingerprint_id=fingerprint_id(terms),
                pd_code="",
                rotation_angles=self.settings.rotation_angles,
                rotation_order=self.settings.rotation_order,
                crossing_count=0,
                runtime_seconds=time.perf_counter() - started,
                removed_crosslink_ids=removed,
                metadata={"empty_core_convention": "Y(empty)=1", **(metadata or {})},
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(record.to_dict(), indent=2, sort_keys=True)
                )
                temporary.replace(cache_path)
            return record
        variable = sp.Symbol(self.settings.variable_name)
        projection = None
        try:
            projection = select_projection(
                graph,
                rotation_angles=self.settings.rotation_angles,
                rotation_order=self.settings.rotation_order,
                num_rotation_samples=self.settings.num_rotation_samples,
            )
            if (
                self.settings.max_crossings is not None
                and projection.num_crossings > self.settings.max_crossings
            ):
                raise FingerprintComplexityError(
                    f"Selected projection has {projection.num_crossings} crossings, "
                    f"exceeding max_crossings={self.settings.max_crossings}. "
                    "Use Repulsor/smoothing, choose explicit projection angles, or "
                    "raise the cap for an intentional exact run."
                )
            polynomial = projection.processor.compute_yamada(
                variable,
                normalize=self.settings.normalize,
                n_jobs=self.settings.n_jobs,
                method=self.settings.method,
            )
            terms = canonical_laurent_terms(polynomial, variable)
            record = FingerprintRecord(
                cache_key=cache_key,
                embedding_hash=graph_hash,
                status="ok",
                polynomial=sp.sstr(sp.expand(polynomial)),
                canonical_terms=terms,
                fingerprint_id=fingerprint_id(terms),
                pd_code=projection.pd_code,
                rotation_angles=projection.rotation_angles,
                rotation_order=projection.rotation_order,
                crossing_count=projection.num_crossings,
                runtime_seconds=time.perf_counter() - started,
                removed_crosslink_ids=removed,
                metadata=dict(metadata or {}),
            )
        except Exception as exc:
            record = FingerprintRecord(
                cache_key=cache_key,
                embedding_hash=graph_hash,
                status="error",
                polynomial=None,
                canonical_terms=(),
                fingerprint_id=None,
                pd_code=projection.pd_code if projection is not None else None,
                rotation_angles=(
                    projection.rotation_angles
                    if projection is not None
                    else self.settings.rotation_angles
                ),
                rotation_order=(
                    projection.rotation_order
                    if projection is not None
                    else self.settings.rotation_order
                ),
                crossing_count=(
                    projection.num_crossings if projection is not None else None
                ),
                runtime_seconds=time.perf_counter() - started,
                removed_crosslink_ids=removed,
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata=dict(metadata or {}),
            )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True))
            temporary.replace(cache_path)
        return record
