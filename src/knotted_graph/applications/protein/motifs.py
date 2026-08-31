"""Local covalent-loop motif detection for crosslinked proteins.

This module is deliberately an adapter rather than an independent minimal-
surface implementation.  When available, it uses the published Topoly 1.1
minimal-surface backend that underlies established protein-lasso analyses.
The dependency is optional, and an unavailable or failed backend is reported
explicitly instead of being interpreted as an absence of lasso motifs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib import import_module, metadata
import json
import re
from typing import Any, Mapping, Sequence, cast

import numpy as np

from knotted_graph.inputs.crosslinks import (
    CrosslinkedProteinInputResult,
    ResidueKey,
)


_PIERCING_TOKEN = re.compile(r"^([+-])(\d+)$")
_LOOP_KINDS = frozenset({"disulfide", "covalent"})


@dataclass(frozen=True)
class LassoDetectionSettings:
    """Numerical and reduction settings passed to the Topoly backend."""

    smoothing: int = 0
    precision: int = 0
    density: int = 1
    minimum_distances: tuple[int, int, int] = (10, 3, 3)
    maximum_adjacent_ca_distance: float = 5.0

    def __post_init__(self) -> None:
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if self.precision not in {0, 1, 2}:
            raise ValueError("precision must be 0 (high), 1 (medium), or 2 (low)")
        if self.density not in {0, 1, 2}:
            raise ValueError("density must be 0 (low), 1 (medium), or 2 (high)")
        if len(self.minimum_distances) != 3 or any(
            value < 0 for value in self.minimum_distances
        ):
            raise ValueError("minimum_distances must contain three non-negative values")
        if self.maximum_adjacent_ca_distance <= 0:
            raise ValueError("maximum_adjacent_ca_distance must be positive")


@dataclass(frozen=True)
class LassoLoopResult:
    """Minimal-surface result for one intra-chain covalent loop."""

    crosslink_id: str
    crosslink_kind: str
    chain_id: str
    endpoint_a: str
    endpoint_b: str
    start_backbone_index: int | None
    end_backbone_index: int | None
    loop_residue_count: int | None
    status: str
    lasso_class: str | None = None
    before_n_indices: tuple[str, ...] = ()
    before_c_indices: tuple[str, ...] = ()
    crossings_n_indices: tuple[str, ...] = ()
    crossings_c_indices: tuple[str, ...] = ()
    before_n_residues: tuple[str, ...] = ()
    before_c_residues: tuple[str, ...] = ()
    crossings_n_residues: tuple[str, ...] = ()
    crossings_c_residues: tuple[str, ...] = ()
    surface_area: float | None = None
    loop_length: float | None = None
    radius_of_gyration: float | None = None
    smoothing_iterations: int | None = None
    issues: tuple[str, ...] = ()

    @property
    def nontrivial(self) -> bool:
        return is_nontrivial_lasso_class(self.lasso_class)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["nontrivial"] = self.nontrivial
        return payload


@dataclass(frozen=True)
class LassoMotifAnalysis:
    """Auditable local-lasso analysis for one parsed protein structure."""

    status: str
    backend: str
    backend_version: str | None
    method: str
    settings: LassoDetectionSettings
    loops: tuple[LassoLoopResult, ...]
    eligible_loop_count: int
    ignored_inter_chain_crosslink_count: int
    ignored_non_loop_crosslink_count: int
    coordinate_gaps: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def nontrivial_lasso_count(self) -> int:
        return sum(loop.nontrivial for loop in self.loops)

    @property
    def local_lasso_motif_signature(self) -> str | None:
        """Return the complete chemistry/class multiset for all analyzed loops."""

        if self.status != "ok":
            return None
        motifs = sorted(
            (loop.crosslink_kind, loop.lasso_class)
            for loop in self.loops
            if loop.status == "ok" and loop.lasso_class is not None
        )
        return json.dumps(motifs, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "method": self.method,
            "settings": asdict(self.settings),
            "eligible_loop_count": self.eligible_loop_count,
            "ignored_inter_chain_crosslink_count": (
                self.ignored_inter_chain_crosslink_count
            ),
            "ignored_non_loop_crosslink_count": self.ignored_non_loop_crosslink_count,
            "coordinate_gaps": list(self.coordinate_gaps),
            "issues": list(self.issues),
            "nontrivial_lasso_count": self.nontrivial_lasso_count,
            "local_lasso_motif_signature": self.local_lasso_motif_signature,
            "loops": [loop.to_dict() for loop in self.loops],
        }


@dataclass(frozen=True)
class LassoDensityRun:
    """Compact result for one minimal-surface mesh density."""

    density: int
    status: str
    signature: str | None
    loop_classes: tuple[tuple[str, str | None], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LassoDensityStabilityAnalysis:
    """Sensitivity of a local-lasso signature to surface mesh density."""

    status: str
    stable: bool | None
    densities: tuple[int, ...]
    runs: tuple[LassoDensityRun, ...]
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stable": self.stable,
            "densities": list(self.densities),
            "runs": [run.to_dict() for run in self.runs],
            "issues": list(self.issues),
        }


def is_nontrivial_lasso_class(value: str | None) -> bool:
    """Return whether a Topoly lasso class denotes at least one piercing."""

    if not value:
        return False
    normalized = value.strip().upper().replace("_", "")
    return normalized not in {"L0", "L+0", "L-0"}


def _load_topoly():
    return import_module("topoly")


def _topoly_version() -> str | None:
    try:
        return metadata.version("topoly")
    except metadata.PackageNotFoundError:
        return None


def _clean(value: object | None, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return "" if text in {".", "?"} else text


def _residue_key(record: Mapping[str, object]) -> ResidueKey:
    return ResidueKey(
        chain_id=_clean(record.get("chain_id"), "?"),
        sequence_id=_clean(record.get("sequence_id"), "?"),
        insertion_code=_clean(record.get("insertion_code")),
    )


def _ordered_backbones(
    protein: CrosslinkedProteinInputResult,
) -> dict[str, tuple[tuple[ResidueKey, np.ndarray], ...]]:
    by_chain: dict[str, list[tuple[ResidueKey, np.ndarray]]] = {}
    seen: set[ResidueKey] = set()
    selected = set(protein.chain_ids)
    backbone_atom = protein.backbone_atom.upper()
    for record in protein.atom_records:
        if _clean(record.get("group"), "ATOM").upper() != "ATOM":
            continue
        if _clean(record.get("atom_name")).upper() != backbone_atom:
            continue
        residue = _residue_key(record)
        if residue.chain_id not in selected or residue in seen:
            continue
        seen.add(residue)
        by_chain.setdefault(residue.chain_id, []).append(
            (residue, np.asarray(record["coord"], dtype=float))
        )
    return {chain: tuple(records) for chain, records in by_chain.items()}


def _coordinate_gaps(
    records: Sequence[tuple[ResidueKey, np.ndarray]],
    maximum_distance: float,
) -> tuple[str, ...]:
    gaps = []
    for (first, first_coord), (second, second_coord) in zip(records, records[1:]):
        reasons = []
        try:
            first_number = int(first.sequence_id)
            second_number = int(second.sequence_id)
        except ValueError:
            first_number = second_number = None
        if (
            first_number is not None
            and second_number is not None
            and second_number - first_number > 1
        ):
            reasons.append("sequence_gap")
        if float(np.linalg.norm(second_coord - first_coord)) > maximum_distance:
            reasons.append("ca_distance_gap")
        if reasons:
            gaps.append(
                f"{first.label}->{second.label}:" + "+".join(reasons)
            )
    return tuple(gaps)


def _tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split())
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _tokens_to_residues(
    tokens: Sequence[str],
    residues: Sequence[ResidueKey],
) -> tuple[str, ...]:
    output = []
    for token in tokens:
        match = _PIERCING_TOKEN.fullmatch(token.strip())
        if match is None:
            output.append(token)
            continue
        index = int(match.group(2))
        if 0 <= index < len(residues):
            output.append(f"{match.group(1)}{residues[index].label}")
        else:
            output.append(token)
    return tuple(output)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _loop_result(
    record,
    start: int | None,
    end: int | None,
    *,
    status: str,
    backend_result: Mapping[str, object] | None = None,
    residues: Sequence[ResidueKey] = (),
    issues: Sequence[str] = (),
) -> LassoLoopResult:
    data = backend_result or {}
    before_n = _tokens(data.get("beforeN"))
    before_c = _tokens(data.get("beforeC"))
    crossings_n = _tokens(data.get("crossingsN"))
    crossings_c = _tokens(data.get("crossingsC"))
    return LassoLoopResult(
        crosslink_id=record.crosslink_id,
        crosslink_kind=record.kind,
        chain_id=record.endpoint_a.residue.chain_id,
        endpoint_a=record.endpoint_a.residue.label,
        endpoint_b=record.endpoint_b.residue.label,
        start_backbone_index=start,
        end_backbone_index=end,
        loop_residue_count=(end - start + 1 if start is not None and end is not None else None),
        status=status,
        lasso_class=(str(data["class"]) if data.get("class") is not None else None),
        before_n_indices=before_n,
        before_c_indices=before_c,
        crossings_n_indices=crossings_n,
        crossings_c_indices=crossings_c,
        before_n_residues=_tokens_to_residues(before_n, residues),
        before_c_residues=_tokens_to_residues(before_c, residues),
        crossings_n_residues=_tokens_to_residues(crossings_n, residues),
        crossings_c_residues=_tokens_to_residues(crossings_c, residues),
        surface_area=_float_or_none(data.get("Area")),
        loop_length=_float_or_none(data.get("loop_length")),
        radius_of_gyration=_float_or_none(data.get("Rg")),
        smoothing_iterations=_int_or_none(data.get("smoothing_iterations")),
        issues=tuple(issues),
    )


def analyze_local_lasso_motifs(
    protein: CrosslinkedProteinInputResult,
    *,
    settings: LassoDetectionSettings | None = None,
) -> LassoMotifAnalysis:
    """Detect local covalent-loop lasso motifs with explicit provenance.

    Only intra-chain disulfide and other covalent links close a loop suitable
    for this analysis.  Inter-chain links are counted but are not silently
    reinterpreted as lasso loops.
    """

    effective = settings or LassoDetectionSettings()
    backbones = _ordered_backbones(protein)
    coordinate_gaps = tuple(
        gap
        for records in backbones.values()
        for gap in _coordinate_gaps(
            records,
            effective.maximum_adjacent_ca_distance,
        )
    )
    ignored_inter_chain = 0
    ignored_non_loop = 0
    eligible_records = []
    for record in protein.crosslinks:
        if record.kind not in _LOOP_KINDS:
            ignored_non_loop += 1
        elif record.endpoint_a.residue.chain_id != record.endpoint_b.residue.chain_id:
            ignored_inter_chain += 1
        else:
            eligible_records.append(record)

    if not eligible_records:
        return LassoMotifAnalysis(
            status="ok",
            backend="topoly",
            backend_version=_topoly_version(),
            method="triangulated_minimal_surface_tail_piercing",
            settings=effective,
            loops=(),
            eligible_loop_count=0,
            ignored_inter_chain_crosslink_count=ignored_inter_chain,
            ignored_non_loop_crosslink_count=ignored_non_loop,
            coordinate_gaps=coordinate_gaps,
        )

    prepared: dict[str, list[tuple[Any, int, int]]] = {}
    loops: list[LassoLoopResult] = []
    incomplete = False
    for record in eligible_records:
        chain_id = record.endpoint_a.residue.chain_id
        chain_records = backbones.get(chain_id, ())
        residues = [residue for residue, _ in chain_records]
        positions = {residue: index for index, residue in enumerate(residues)}
        first = positions.get(record.endpoint_a.residue)
        second = positions.get(record.endpoint_b.residue)
        if first is None or second is None:
            incomplete = True
            loops.append(
                _loop_result(
                    record,
                    first,
                    second,
                    status="missing_backbone_endpoint",
                    issues=("crosslink endpoint lacks the selected backbone atom",),
                )
            )
            continue
        start, end = sorted((first, second))
        if end - start + 1 < 3:
            incomplete = True
            loops.append(
                _loop_result(
                    record,
                    start,
                    end,
                    status="loop_too_short",
                    residues=residues,
                    issues=("a triangulated loop requires at least three residues",),
                )
            )
            continue
        prepared.setdefault(chain_id, []).append((record, start, end))

    if not prepared:
        return LassoMotifAnalysis(
            status="partial",
            backend="topoly",
            backend_version=_topoly_version(),
            method="triangulated_minimal_surface_tail_piercing",
            settings=effective,
            loops=tuple(sorted(loops, key=lambda loop: loop.crosslink_id)),
            eligible_loop_count=len(eligible_records),
            ignored_inter_chain_crosslink_count=ignored_inter_chain,
            ignored_non_loop_crosslink_count=ignored_non_loop,
            coordinate_gaps=coordinate_gaps,
            issues=("no eligible loop had complete, triangulatable coordinates",),
        )

    try:
        topoly = _load_topoly()
    except (ImportError, OSError) as exc:
        for records in prepared.values():
            for record, start, end in records:
                residues = [item[0] for item in backbones[record.chains[0]]]
                loops.append(
                    _loop_result(
                        record,
                        start,
                        end,
                        status="backend_unavailable",
                        residues=residues,
                        issues=(str(exc),),
                    )
                )
        return LassoMotifAnalysis(
            status="backend_unavailable",
            backend="topoly",
            backend_version=None,
            method="triangulated_minimal_surface_tail_piercing",
            settings=effective,
            loops=tuple(sorted(loops, key=lambda loop: loop.crosslink_id)),
            eligible_loop_count=len(eligible_records),
            ignored_inter_chain_crosslink_count=ignored_inter_chain,
            ignored_non_loop_crosslink_count=ignored_non_loop,
            coordinate_gaps=coordinate_gaps,
            issues=(str(exc),),
        )

    backend_failures = []
    for chain_id, records in prepared.items():
        chain_records = backbones[chain_id]
        residues = [residue for residue, _ in chain_records]
        coordinates = [coord.tolist() for _, coord in chain_records]
        unique_intervals = sorted({(start, end) for _, start, end in records})
        try:
            raw_results = topoly.lasso_type(
                coordinates,
                loop_indices=unique_intervals,
                smooth=effective.smoothing,
                precision=effective.precision,
                density=effective.density,
                min_dist=effective.minimum_distances,
                more_info=True,
            )
        except Exception as exc:  # Topoly exposes several backend exception types.
            incomplete = True
            backend_failures.append(f"chain {chain_id}: {type(exc).__name__}: {exc}")
            for record, start, end in records:
                loops.append(
                    _loop_result(
                        record,
                        start,
                        end,
                        status="backend_error",
                        residues=residues,
                        issues=(f"{type(exc).__name__}: {exc}",),
                    )
                )
            continue
        for record, start, end in records:
            result = raw_results.get((start, end))
            if not isinstance(result, Mapping):
                incomplete = True
                loops.append(
                    _loop_result(
                        record,
                        start,
                        end,
                        status="backend_error",
                        residues=residues,
                        issues=("backend returned no structured result for this loop",),
                    )
                )
                continue
            loop_status = "ok"
            loop_issues = (
                (
                    "coordinate gaps are represented by straight segments, matching "
                    "the published LassoProt convention"
                ),
            ) if coordinate_gaps else ()
            loops.append(
                _loop_result(
                    record,
                    start,
                    end,
                    status=loop_status,
                    backend_result=result,
                    residues=residues,
                    issues=loop_issues,
                )
            )

    issues = list(backend_failures)
    if coordinate_gaps:
        issues.append(
            "one or more chains contain coordinate gaps represented by straight "
            "segments; gap locations are retained for sensitivity audits"
        )
    return LassoMotifAnalysis(
        status="partial" if incomplete else "ok",
        backend="topoly",
        backend_version=_topoly_version(),
        method="triangulated_minimal_surface_tail_piercing",
        settings=effective,
        loops=tuple(sorted(loops, key=lambda loop: loop.crosslink_id)),
        eligible_loop_count=len(eligible_records),
        ignored_inter_chain_crosslink_count=ignored_inter_chain,
        ignored_non_loop_crosslink_count=ignored_non_loop,
        coordinate_gaps=coordinate_gaps,
        issues=tuple(issues),
    )


def analyze_lasso_density_stability(
    protein: CrosslinkedProteinInputResult,
    *,
    settings: LassoDetectionSettings | None = None,
    densities: Sequence[int] = (0, 1, 2),
    baseline: LassoMotifAnalysis | None = None,
) -> LassoDensityStabilityAnalysis:
    """Require the complete motif signature to agree across mesh densities."""

    effective = settings or LassoDetectionSettings()
    requested = tuple(dict.fromkeys(int(value) for value in densities))
    if not requested:
        raise ValueError("densities must not be empty")
    if any(value not in {0, 1, 2} for value in requested):
        raise ValueError("densities must contain only 0 (low), 1 (medium), or 2 (high)")

    analyses = []
    for density in requested:
        if baseline is not None and baseline.settings == replace(
            effective, density=density
        ):
            analysis = baseline
        else:
            analysis = analyze_local_lasso_motifs(
                protein,
                settings=replace(effective, density=density),
            )
        analyses.append(analysis)
    runs = tuple(
        LassoDensityRun(
            density=density,
            status=analysis.status,
            signature=analysis.local_lasso_motif_signature,
            loop_classes=tuple(
                (loop.crosslink_id, loop.lasso_class) for loop in analysis.loops
            ),
        )
        for density, analysis in zip(requested, analyses)
    )
    incomplete = [
        run.density
        for run in runs
        if run.status != "ok" or run.signature is None
    ]
    if incomplete:
        return LassoDensityStabilityAnalysis(
            status="incomplete",
            stable=None,
            densities=requested,
            runs=runs,
            issues=(f"incomplete lasso analyses at densities {incomplete}",),
        )
    signatures = {run.signature for run in runs}
    stable = len(signatures) == 1
    return LassoDensityStabilityAnalysis(
        status="stable" if stable else "unstable",
        stable=stable,
        densities=requested,
        runs=runs,
        issues=() if stable else ("complete motif signature changes with mesh density",),
    )
