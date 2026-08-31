"""Resumable manifest-driven protein topology analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from knotted_graph.inputs import load_crosslinked_protein
from knotted_graph.inputs.crosslinks import DEFAULT_CROSSLINK_TYPES
from knotted_graph.layout.repulsive import (
    DecimationOptions,
    DriverConfig,
    ResamplingOptions,
    SolverOptions,
)

from .fingerprint import FingerprintComputer, FingerprintSettings
from .conditioned import (
    AbstractConditionedRobustness,
    ConditionedNullComparison,
    analyze_abstract_conditioned_robustness,
    compare_conditioned_topology_to_null,
)
from .graph import (
    abstract_connectivity_certificate,
    abstract_connectivity_hash,
    abstract_connectivity_isomorphic,
    crosslink_edges,
    extract_crosslink_core,
)
from .null_models import (
    compare_robustness_to_null,
    crosslink_content_signature,
    generate_null_graphs,
    generate_unique_disulfide_null_graphs,
)
from .motifs import (
    LassoDensityStabilityAnalysis,
    LassoMotifAnalysis,
    analyze_lasso_density_stability,
    analyze_local_lasso_motifs,
)
from .models import MinimumGeneratingSetSearch
from .perturbation import (
    analyze_crosslink_perturbations,
    search_minimum_generating_crosslink_sets,
)
from .repulsion import RepulsionTopologyResult, relax_and_analyze_crosslinks
from .statistics import (
    find_pattern_candidate_pairs,
    find_same_connectivity_different_fingerprints,
    find_same_crosslink_content_different_fingerprints,
    find_same_local_lasso_motifs_different_fingerprints,
    summarize_population_robustness,
)
from .visualization import (
    close_figure,
    plot_edge_importance,
    plot_natural_vs_null,
    plot_pair_synergy_heatmap,
    plot_protein_graph_3d,
)


ANALYSIS_SCHEMA_VERSION = 14


@dataclass(frozen=True)
class ProteinManifestEntry:
    sample_id: str
    source: str
    source_format: str | None = None
    pdb_id: str | None = None
    chain_ids: tuple[str, ...] = ()
    model_id: int = 1
    backbone_atom: str = "CA"
    allowed_crosslink_types: tuple[str, ...] = tuple(sorted(DEFAULT_CROSSLINK_TYPES))
    crosslink_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ProteinBatchSettings:
    include_pairs: bool = True
    exact_subsets: str = "auto"
    max_exact_crosslinks: int = 10
    null_replicates: int = 0
    null_seed: int = 0
    null_embedding_mode: str = "coordinate_preserving"
    null_sampling_mode: str = "random_replicates"
    null_repulsion_fallback_steps: int = 0
    null_repulsion_fallback_max_time: float = 1.0
    null_repulsion_fallback_free_special_vertices: bool = False
    null_repulsion_fallback_decimation_passes: int = 8
    null_repulsion_fallback_max_points_per_edge: int | None = None
    repulsion_steps: int = 0
    repulsion_max_time: float = 1.0
    repulsion_free_special_vertices: bool = False
    repulsion_decimation_passes: int = 8
    repulsion_max_points_per_edge: int | None = None
    repulsion_fallback_only: bool = False
    repulsor_root: str | None = None
    allow_repulsor_certificate_only: bool = False
    conditioned_robustness: bool = False
    conditioned_max_subset_order: int | None = None
    minimum_generator_max_retained_crosslinks: int | None = None
    resume: bool = True

    def __post_init__(self) -> None:
        if self.exact_subsets not in {"none", "auto", "all"}:
            raise ValueError("exact_subsets must be one of: none, auto, all")
        if self.max_exact_crosslinks < 0:
            raise ValueError("max_exact_crosslinks must be non-negative")
        if self.null_replicates < 0:
            raise ValueError("null_replicates must be non-negative")
        if self.null_embedding_mode not in {
            "coordinate_preserving",
            "canonical_low_crossing",
        }:
            raise ValueError(
                "null_embedding_mode must be 'coordinate_preserving' or "
                "'canonical_low_crossing'"
            )
        if self.null_sampling_mode not in {
            "random_replicates",
            "unique_disulfide_matchings",
        }:
            raise ValueError(
                "null_sampling_mode must be 'random_replicates' or "
                "'unique_disulfide_matchings'"
            )
        if self.repulsion_steps < 0:
            raise ValueError("repulsion_steps must be non-negative")
        if self.repulsion_max_time <= 0.0:
            raise ValueError("repulsion_max_time must be positive")
        if self.repulsion_decimation_passes < 1:
            raise ValueError("repulsion_decimation_passes must be at least 1")
        if (
            self.repulsion_max_points_per_edge is not None
            and self.repulsion_max_points_per_edge < 3
        ):
            raise ValueError("repulsion_max_points_per_edge must be at least 3")
        if self.null_repulsion_fallback_steps < 0:
            raise ValueError("null_repulsion_fallback_steps must be non-negative")
        if self.null_repulsion_fallback_max_time <= 0.0:
            raise ValueError("null_repulsion_fallback_max_time must be positive")
        if self.null_repulsion_fallback_decimation_passes < 1:
            raise ValueError(
                "null_repulsion_fallback_decimation_passes must be at least 1"
            )
        if (
            self.null_repulsion_fallback_max_points_per_edge is not None
            and self.null_repulsion_fallback_max_points_per_edge < 3
        ):
            raise ValueError(
                "null_repulsion_fallback_max_points_per_edge must be at least 3"
            )
        if (
            self.conditioned_max_subset_order is not None
            and self.conditioned_max_subset_order < 1
        ):
            raise ValueError("conditioned_max_subset_order must be at least 1")
        if (
            self.minimum_generator_max_retained_crosslinks is not None
            and self.minimum_generator_max_retained_crosslinks < 0
        ):
            raise ValueError(
                "minimum_generator_max_retained_crosslinks must be non-negative"
            )


def _split_values(value: str | None) -> tuple[str, ...]:
    if not value or value.strip() in {"", ".", "?"}:
        return ()
    return tuple(part.strip() for part in re.split(r"[,;|]", value) if part.strip())


def read_protein_manifest(path: str | Path) -> list[ProteinManifestEntry]:
    """Read a CSV manifest, resolving local coordinate paths by manifest location."""

    manifest_path = Path(path).resolve()
    entries: list[ProteinManifestEntry] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "source" not in reader.fieldnames:
            raise ValueError("Protein manifest must contain a 'source' column")
        for row_number, row in enumerate(reader, start=2):
            source = (row.get("source") or "").strip()
            if not source:
                raise ValueError(f"Manifest row {row_number} has an empty source")
            source_path = Path(source)
            if source_path.suffix and not source_path.is_absolute():
                source = str((manifest_path.parent / source_path).resolve())
            pdb_id = (row.get("pdb_id") or "").strip() or None
            sample_id = (
                (row.get("sample_id") or "").strip() or pdb_id or Path(source).stem
            )
            metadata_text = (row.get("metadata_json") or "").strip()
            metadata = json.loads(metadata_text) if metadata_text else {}
            allowed = _split_values(row.get("allowed_crosslink_types"))
            entries.append(
                ProteinManifestEntry(
                    sample_id=sample_id,
                    source=source,
                    source_format=(row.get("source_format") or "").strip() or None,
                    pdb_id=pdb_id,
                    chain_ids=_split_values(row.get("chain_ids")),
                    model_id=int((row.get("model_id") or "1").strip()),
                    backbone_atom=(row.get("backbone_atom") or "CA").strip(),
                    allowed_crosslink_types=(
                        allowed if allowed else tuple(sorted(DEFAULT_CROSSLINK_TYPES))
                    ),
                    crosslink_ids=_split_values(row.get("crosslink_ids")),
                    metadata=metadata,
                )
            )
    ids = [entry.sample_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample_id values must be unique")
    return entries


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_") or "sample"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _summary_row(
    entry,
    protein,
    analysis,
    analysis_core,
    lasso_motifs: LassoMotifAnalysis,
    lasso_stability: LassoDensityStabilityAnalysis,
    conditioned: AbstractConditionedRobustness | None,
    bounded_generator: MinimumGeneratingSetSearch | None,
    conditioned_comparison: ConditionedNullComparison | None,
    comparison,
    repulsion: RepulsionTopologyResult | None,
    null_requested: int,
    null_analyzed: int,
    null_successful: int,
    null_failed: int,
    null_ensemble_exhaustive: bool,
    null_ensemble_size: int | None,
) -> dict[str, Any]:
    if not analysis.baseline.success:
        status = "fingerprint_error"
    elif repulsion is not None and repulsion.status not in {"ok", "certificate_only"}:
        status = repulsion.status
    else:
        status = "ok"
    return {
        "sample_id": entry.sample_id,
        "status": status,
        "pdb_id": protein.pdb_id,
        "chain_ids": "|".join(protein.chain_ids),
        "source_format": protein.source_format,
        "crosslink_count": len(analysis.crosslink_ids),
        "successful_single_count": analysis.successful_single_count,
        "failed_single_count": analysis.failed_single_count,
        "topological_fraction": analysis.topological_fraction,
        "robustness_r1": analysis.robustness_r1,
        "cooperative_pair_count": analysis.cooperative_pair_count,
        "baseline_fingerprint_id": analysis.baseline.fingerprint_id,
        "baseline_crossing_count": analysis.baseline.crossing_count,
        "baseline_runtime_seconds": analysis.baseline.runtime_seconds,
        "crosslink_content_signature": crosslink_content_signature(analysis_core),
        "lasso_detection_status": lasso_motifs.status,
        "local_lasso_motif_signature": (
            lasso_motifs.local_lasso_motif_signature
        ),
        "nontrivial_lasso_count": lasso_motifs.nontrivial_lasso_count,
        "lasso_density_stability_status": lasso_stability.status,
        "lasso_density_stable": lasso_stability.stable,
        "abstract_connectivity_hash": abstract_connectivity_hash(analysis_core),
        "abstract_connectivity_class_id": None,
        "abstract_connectivity_isomorphism_verified": False,
        "conditioned_status": conditioned.status if conditioned else "not_requested",
        "baseline_embedding_nontrivial": (
            conditioned.baseline_embedding_nontrivial if conditioned else None
        ),
        "conditioned_state_robustness_r1": (
            conditioned.conditioned_state_robustness_r1 if conditioned else None
        ),
        "entanglement_retention_r1": (
            conditioned.entanglement_retention_r1 if conditioned else None
        ),
        "topology_carrying_edge_count": (
            conditioned.topology_carrying_edge_count if conditioned else None
        ),
        "topology_carrying_edge_fraction": (
            conditioned.topology_carrying_edge_fraction if conditioned else None
        ),
        "has_topology_carrying_edge": (
            conditioned.has_topology_carrying_edge if conditioned else None
        ),
        "conditioned_cooperative_pair_count": (
            conditioned.cooperative_pair_count if conditioned else None
        ),
        "conditioned_maximum_subset_order": (
            conditioned.maximum_subset_order_evaluated if conditioned else None
        ),
        "conditioned_strictly_cooperative_subset_count": (
            conditioned.strictly_cooperative_subset_count if conditioned else None
        ),
        "conditioned_minimum_information_carrying_subset_size": (
            conditioned.minimum_information_carrying_subset_size
            if conditioned
            else None
        ),
        "bounded_generator_status": (
            bounded_generator.status if bounded_generator else "not_requested"
        ),
        "bounded_generator_proven_minimum_count": (
            bounded_generator.proven_minimum_count if bounded_generator else None
        ),
        "bounded_generator_proven_lower_bound": (
            bounded_generator.proven_lower_bound if bounded_generator else None
        ),
        "bounded_generator_maximum_retained_size_evaluated": (
            bounded_generator.maximum_retained_size_evaluated
            if bounded_generator
            else None
        ),
        "conditioned_null_mean_topology_carrying_edge_fraction": (
            conditioned_comparison.null_mean if conditioned_comparison else None
        ),
        "conditioned_null_p_greater_equal": (
            conditioned_comparison.empirical_p_greater_equal
            if conditioned_comparison
            else None
        ),
        "minimum_changed_subset_size": (
            len(analysis.minimum_cardinality_subsets[0])
            if analysis.minimum_cardinality_subsets
            else None
        ),
        "minimum_generating_crosslink_count": (
            analysis.minimum_generating_crosslink_count
        ),
        "minimum_generating_crosslink_status": (
            analysis.minimum_generating_crosslink_status
        ),
        "evaluated_subset_state_count": len(analysis.subsets) + 1,
        "expected_subset_state_count": (
            1 << len(analysis.crosslink_ids)
            if analysis.metadata.get("enumerate_all_subsets")
            else None
        ),
        "null_replicates": null_successful,
        "null_replicates_requested": null_requested,
        "null_replicates_analyzed": null_analyzed,
        "null_replicates_failed": null_failed,
        "null_ensemble_exhaustive": null_ensemble_exhaustive,
        "null_ensemble_size": null_ensemble_size,
        "null_mean_robustness": comparison.null_mean if comparison else None,
        "null_z_score": comparison.z_score if comparison else None,
        "null_p_greater_equal": comparison.empirical_p_greater_equal
        if comparison
        else None,
        "null_p_less_equal": comparison.empirical_p_less_equal if comparison else None,
        "repulsion_status": repulsion.status if repulsion else "not_requested",
        "repulsion_topology_preserved": (
            repulsion.topology_preserved if repulsion else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _annotate_exact_connectivity_classes(details: list[dict[str, Any]]) -> None:
    """Split WL-hash buckets using exact labeled graph isomorphism."""

    buckets: dict[str, list[dict[str, Any]]] = {}
    for detail in details:
        summary = detail.get("summary", {})
        summary["abstract_connectivity_class_id"] = None
        summary["abstract_connectivity_isomorphism_verified"] = False
        certificate = detail.get("abstract_connectivity_certificate")
        screening_hash = summary.get("abstract_connectivity_hash")
        if (
            summary.get("status") == "ok"
            and isinstance(certificate, dict)
            and screening_hash
        ):
            buckets.setdefault(str(screening_hash), []).append(detail)

    for screening_hash, members in sorted(buckets.items()):
        representatives: list[tuple[str, dict[str, Any]]] = []
        for detail in sorted(
            members,
            key=lambda item: str(item["summary"]["sample_id"]),
        ):
            certificate = detail["abstract_connectivity_certificate"]
            class_id = None
            for candidate_id, representative in representatives:
                if abstract_connectivity_isomorphic(certificate, representative):
                    class_id = candidate_id
                    break
            if class_id is None:
                class_id = f"{screening_hash}:iso{len(representatives)}"
                representatives.append((class_id, certificate))
            detail["summary"]["abstract_connectivity_class_id"] = class_id
            detail["summary"]["abstract_connectivity_isomorphism_verified"] = True


def _write_consolidated_outputs(
    output_dir: Path, details: list[dict[str, Any]]
) -> None:
    _annotate_exact_connectivity_classes(details)
    for detail in details:
        sample_id = detail.get("summary", {}).get("sample_id")
        if sample_id:
            _atomic_json(
                output_dir / "analyses" / f"{_safe_sample_id(str(sample_id))}.json",
                detail,
            )
    summaries = [detail["summary"] for detail in details]
    summary_fields = [
        "sample_id",
        "status",
        "pdb_id",
        "chain_ids",
        "source_format",
        "crosslink_count",
        "successful_single_count",
        "failed_single_count",
        "topological_fraction",
        "robustness_r1",
        "cooperative_pair_count",
        "baseline_fingerprint_id",
        "baseline_crossing_count",
        "baseline_runtime_seconds",
        "crosslink_content_signature",
        "lasso_detection_status",
        "local_lasso_motif_signature",
        "nontrivial_lasso_count",
        "lasso_density_stability_status",
        "lasso_density_stable",
        "abstract_connectivity_hash",
        "abstract_connectivity_class_id",
        "abstract_connectivity_isomorphism_verified",
        "conditioned_status",
        "baseline_embedding_nontrivial",
        "conditioned_state_robustness_r1",
        "entanglement_retention_r1",
        "topology_carrying_edge_count",
        "topology_carrying_edge_fraction",
        "has_topology_carrying_edge",
        "conditioned_cooperative_pair_count",
        "conditioned_maximum_subset_order",
        "conditioned_strictly_cooperative_subset_count",
        "conditioned_minimum_information_carrying_subset_size",
        "bounded_generator_status",
        "bounded_generator_proven_minimum_count",
        "bounded_generator_proven_lower_bound",
        "bounded_generator_maximum_retained_size_evaluated",
        "conditioned_null_mean_topology_carrying_edge_fraction",
        "conditioned_null_p_greater_equal",
        "minimum_changed_subset_size",
        "minimum_generating_crosslink_count",
        "minimum_generating_crosslink_status",
        "evaluated_subset_state_count",
        "expected_subset_state_count",
        "null_replicates",
        "null_replicates_requested",
        "null_replicates_analyzed",
        "null_replicates_failed",
        "null_ensemble_exhaustive",
        "null_ensemble_size",
        "null_mean_robustness",
        "null_z_score",
        "null_p_greater_equal",
        "null_p_less_equal",
        "repulsion_status",
        "repulsion_topology_preserved",
        "error_type",
        "error_message",
    ]
    _write_csv(output_dir / "summary.csv", summaries, summary_fields)

    edge_rows = []
    conditioned_edge_rows = []
    conditioned_pair_rows = []
    conditioned_subset_rows = []
    pair_rows = []
    subset_rows = []
    generating_set_rows = []
    bounded_generating_set_rows = []
    for detail in details:
        analysis = detail.get("analysis")
        if analysis is None:
            continue
        sample_id = detail["summary"]["sample_id"]
        crosslink_ids = tuple(analysis["crosslink_ids"])
        subset_rows.append(
            {
                "sample_id": sample_id,
                "removed_crosslink_ids": "",
                "retained_crosslink_ids": "|".join(crosslink_ids),
                "removed_count": 0,
                "retained_count": len(crosslink_ids),
                "changed_from_full": False,
                "status": analysis["baseline"]["status"],
                "fingerprint_id": analysis["baseline"]["fingerprint_id"],
                "crossing_count": analysis["baseline"]["crossing_count"],
                "runtime_seconds": analysis["baseline"]["runtime_seconds"],
                "error_message": analysis["baseline"]["error_message"],
            }
        )
        for record in analysis["subsets"]:
            removed = tuple(record["removed_crosslink_ids"])
            removed_set = set(removed)
            retained = tuple(
                crosslink_id
                for crosslink_id in crosslink_ids
                if crosslink_id not in removed_set
            )
            subset_rows.append(
                {
                    "sample_id": sample_id,
                    "removed_crosslink_ids": "|".join(removed),
                    "retained_crosslink_ids": "|".join(retained),
                    "removed_count": len(removed),
                    "retained_count": len(retained),
                    "changed_from_full": record["changed"],
                    "status": record["fingerprint"]["status"],
                    "fingerprint_id": record["fingerprint"]["fingerprint_id"],
                    "crossing_count": record["fingerprint"]["crossing_count"],
                    "runtime_seconds": record["fingerprint"]["runtime_seconds"],
                    "error_message": record["fingerprint"]["error_message"],
                }
            )
        for generating_set in analysis.get("minimum_generating_crosslink_sets", []):
            generating_set_rows.append(
                {
                    "sample_id": sample_id,
                    "minimum_generating_crosslink_count": len(generating_set),
                    "retained_crosslink_ids": "|".join(generating_set),
                    "status": analysis.get("minimum_generating_crosslink_status"),
                }
            )
        bounded_generator = detail.get("bounded_minimum_generator") or {}
        for generating_set in bounded_generator.get("proven_minimum_sets", []):
            bounded_generating_set_rows.append(
                {
                    "sample_id": sample_id,
                    "proven_minimum_count": bounded_generator.get(
                        "proven_minimum_count"
                    ),
                    "retained_crosslink_ids": "|".join(generating_set),
                    "proven_lower_bound": bounded_generator.get(
                        "proven_lower_bound"
                    ),
                    "status": bounded_generator.get("status"),
                }
            )
        for record in analysis["singles"]:
            edge_rows.append(
                {
                    "sample_id": sample_id,
                    "crosslink_id": record["crosslink_id"],
                    "crosslink_type": record["crosslink_type"],
                    "endpoint_a": record["endpoint_a"],
                    "endpoint_b": record["endpoint_b"],
                    "changed": record["changed"],
                    "status": record["fingerprint"]["status"],
                    "fingerprint_id": record["fingerprint"]["fingerprint_id"],
                    "crossing_count": record["fingerprint"]["crossing_count"],
                    "runtime_seconds": record["fingerprint"]["runtime_seconds"],
                    "error_message": record["fingerprint"]["error_message"],
                }
            )
        conditioned = detail.get("conditioned_robustness") or {}
        conditioned_by_id = {
            record["crosslink_id"]: record
            for record in conditioned.get("singles", [])
        }
        lasso_classes = {
            loop["crosslink_id"]: loop.get("lasso_class")
            for loop in (detail.get("lasso_motifs") or {}).get("loops", [])
        }
        for record in analysis["singles"]:
            conditioned_record = conditioned_by_id.get(record["crosslink_id"])
            if conditioned_record is None:
                continue
            conditioned_edge_rows.append(
                {
                    "sample_id": sample_id,
                    "crosslink_id": record["crosslink_id"],
                    "crosslink_type": record["crosslink_type"],
                    "endpoint_a": record["endpoint_a"],
                    "endpoint_b": record["endpoint_b"],
                    "lasso_class": lasso_classes.get(record["crosslink_id"]),
                    "baseline_embedding_nontrivial": conditioned.get(
                        "baseline_embedding_nontrivial"
                    ),
                    "deleted_state_nontrivial": conditioned_record.get(
                        "deleted_state_nontrivial"
                    ),
                    "information_carrying": conditioned_record.get(
                        "information_carrying"
                    ),
                }
            )
        for record in analysis["pairs"]:
            pair_rows.append(
                {
                    "sample_id": sample_id,
                    "crosslink_i": record["crosslink_i"],
                    "crosslink_j": record["crosslink_j"],
                    "changed": record["changed"],
                    "cooperative": record["cooperative"],
                    "synergy_score": record["synergy_score"],
                    "status": record["fingerprint"]["status"],
                    "fingerprint_id": record["fingerprint"]["fingerprint_id"],
                    "runtime_seconds": record["fingerprint"]["runtime_seconds"],
                    "error_message": record["fingerprint"]["error_message"],
                }
            )
        for record in conditioned.get("pairs", []):
            conditioned_pair_rows.append(
                {
                    "sample_id": sample_id,
                    "crosslink_i": record["crosslink_i"],
                    "crosslink_j": record["crosslink_j"],
                    "deleted_state_nontrivial": record.get(
                        "deleted_state_nontrivial"
                    ),
                    "information_carrying": record.get("information_carrying"),
                    "cooperative": record.get("cooperative"),
                }
            )
        for record in conditioned.get("subsets", []):
            conditioned_subset_rows.append(
                {
                    "sample_id": sample_id,
                    "removed_crosslink_ids": "|".join(
                        record["removed_crosslink_ids"]
                    ),
                    "order": record["order"],
                    "deleted_state_nontrivial": record.get(
                        "deleted_state_nontrivial"
                    ),
                    "information_carrying": record.get("information_carrying"),
                    "strictly_cooperative": record.get("strictly_cooperative"),
                }
            )
    _write_csv(
        output_dir / "edge_impacts.csv",
        edge_rows,
        [
            "sample_id",
            "crosslink_id",
            "crosslink_type",
            "endpoint_a",
            "endpoint_b",
            "changed",
            "status",
            "fingerprint_id",
            "crossing_count",
            "runtime_seconds",
            "error_message",
        ],
    )
    _write_csv(
        output_dir / "conditioned_edge_impacts.csv",
        conditioned_edge_rows,
        [
            "sample_id",
            "crosslink_id",
            "crosslink_type",
            "endpoint_a",
            "endpoint_b",
            "lasso_class",
            "baseline_embedding_nontrivial",
            "deleted_state_nontrivial",
            "information_carrying",
        ],
    )
    _write_csv(
        output_dir / "pair_impacts.csv",
        pair_rows,
        [
            "sample_id",
            "crosslink_i",
            "crosslink_j",
            "changed",
            "cooperative",
            "synergy_score",
            "status",
            "fingerprint_id",
            "runtime_seconds",
            "error_message",
        ],
    )
    _write_csv(
        output_dir / "conditioned_pair_impacts.csv",
        conditioned_pair_rows,
        [
            "sample_id",
            "crosslink_i",
            "crosslink_j",
            "deleted_state_nontrivial",
            "information_carrying",
            "cooperative",
        ],
    )
    _write_csv(
        output_dir / "conditioned_subset_impacts.csv",
        conditioned_subset_rows,
        [
            "sample_id",
            "removed_crosslink_ids",
            "order",
            "deleted_state_nontrivial",
            "information_carrying",
            "strictly_cooperative",
        ],
    )
    _write_csv(
        output_dir / "subset_states.csv",
        subset_rows,
        [
            "sample_id",
            "removed_crosslink_ids",
            "retained_crosslink_ids",
            "removed_count",
            "retained_count",
            "changed_from_full",
            "status",
            "fingerprint_id",
            "crossing_count",
            "runtime_seconds",
            "error_message",
        ],
    )
    _write_csv(
        output_dir / "minimum_generating_sets.csv",
        generating_set_rows,
        [
            "sample_id",
            "minimum_generating_crosslink_count",
            "retained_crosslink_ids",
            "status",
        ],
    )
    _write_csv(
        output_dir / "bounded_minimum_generating_sets.csv",
        bounded_generating_set_rows,
        [
            "sample_id",
            "proven_minimum_count",
            "retained_crosslink_ids",
            "proven_lower_bound",
            "status",
        ],
    )
    patterns = {
        "same_local_lasso_motifs_different_fingerprints": (
            find_same_local_lasso_motifs_different_fingerprints(summaries)
        ),
        "same_crosslink_content_different_fingerprints": (
            find_same_crosslink_content_different_fingerprints(summaries)
        ),
        "same_connectivity_different_fingerprints": (
            find_same_connectivity_different_fingerprints(summaries)
        ),
    }
    _atomic_json(output_dir / "pattern_groups.json", patterns)
    pattern_candidates = find_pattern_candidate_pairs(summaries)
    _write_csv(
        output_dir / "pattern_candidates.csv",
        pattern_candidates,
        [
            "hypothesis",
            "group_key",
            "group_value",
            "sample_i",
            "sample_j",
            "pdb_i",
            "pdb_j",
            "fingerprint_i",
            "fingerprint_j",
        ],
    )
    population = summarize_population_robustness(details)
    _atomic_json(output_dir / "dataset_statistics.json", population)
    _write_csv(
        output_dir / "robustness_by_sample.csv",
        population["per_sample"],
        [
            "sample_id",
            "pdb_id",
            "natural_robustness_r1",
            "null_mean_robustness_r1",
            "natural_minus_null_mean",
            "null_replicates_requested",
            "null_replicates_analyzed",
            "null_replicates_failed",
            "null_ensemble_exhaustive",
            "null_ensemble_size",
        ],
    )
    _write_csv(
        output_dir / "conditioned_robustness_by_sample.csv",
        population["abstract_conditioned"]["per_sample"],
        [
            "sample_id",
            "pdb_id",
            "baseline_embedding_nontrivial",
            "has_topology_carrying_edge",
            "natural_topology_carrying_edge_fraction",
            "null_mean_topology_carrying_edge_fraction",
            "natural_minus_null_mean",
            "successful_conditioned_null_count",
            "null_ensemble_exhaustive",
            "null_ensemble_size",
        ],
    )


def _should_enumerate_exact(settings: ProteinBatchSettings, count: int) -> bool:
    if settings.exact_subsets == "none":
        return False
    if settings.exact_subsets == "all":
        return True
    return count <= settings.max_exact_crosslinks


def _null_summary(
    null_record,
    analysis,
    repulsion: RepulsionTopologyResult | None = None,
    conditioned: AbstractConditionedRobustness | None = None,
) -> dict[str, Any]:
    graph_metadata = null_record.graph.graph
    return {
        "replicate": null_record.replicate,
        "seed": null_record.seed,
        "input_id": analysis.input_id,
        "embedding_mode": graph_metadata.get(
            "null_embedding_mode",
            "coordinate_preserving",
        ),
        "layout_algorithm": graph_metadata.get("canonical_layout_algorithm"),
        "min_non_adjacent_segment_distance": graph_metadata.get(
            "canonical_min_non_adjacent_segment_distance"
        ),
        "issues": list(null_record.issues),
        "null_ensemble_size": graph_metadata.get("null_ensemble_size"),
        "null_ensemble_selected_count": graph_metadata.get(
            "null_ensemble_selected_count"
        ),
        "null_ensemble_exhaustive": bool(
            graph_metadata.get("null_ensemble_exhaustive", False)
        ),
        "null_sampling_without_replacement": bool(
            graph_metadata.get("null_sampling_without_replacement", False)
        ),
        "robustness_r1": analysis.robustness_r1,
        "topological_fraction": analysis.topological_fraction,
        "baseline_status": analysis.baseline.status,
        "baseline_crossing_count": analysis.baseline.crossing_count,
        "baseline_fingerprint_id": analysis.baseline.fingerprint_id,
        "repulsion_status": repulsion.status
        if repulsion is not None
        else "not_requested",
        "repulsion_topology_preserved": (
            repulsion.topology_preserved if repulsion is not None else None
        ),
        "conditioned_status": conditioned.status if conditioned else "not_requested",
        "baseline_embedding_nontrivial": (
            conditioned.baseline_embedding_nontrivial if conditioned else None
        ),
        "topology_carrying_edge_fraction": (
            conditioned.topology_carrying_edge_fraction if conditioned else None
        ),
        "conditioned_state_robustness_r1": (
            conditioned.conditioned_state_robustness_r1 if conditioned else None
        ),
    }


def run_protein_batch(
    entries: Iterable[ProteinManifestEntry],
    output_dir: str | Path,
    *,
    fingerprint_settings: FingerprintSettings | None = None,
    batch_settings: ProteinBatchSettings | None = None,
) -> list[dict[str, Any]]:
    """Run or resume a local batch and write JSON/CSV/PNG artifacts."""

    output_path = Path(output_dir)
    detail_dir = output_path / "analyses"
    figure_dir = output_path / "figures"
    settings = batch_settings or ProteinBatchSettings()
    effective_fingerprint_settings = fingerprint_settings or FingerprintSettings()
    batch_config = asdict(settings)
    batch_config.pop("resume", None)
    run_config = {
        "fingerprint_settings": asdict(effective_fingerprint_settings),
        "batch_settings": batch_config,
    }
    computer = FingerprintComputer(
        output_path / "fingerprint_cache",
        settings=effective_fingerprint_settings,
    )
    details: list[dict[str, Any]] = []
    for entry in entries:
        detail_path = detail_dir / f"{_safe_sample_id(entry.sample_id)}.json"
        if settings.resume and detail_path.exists():
            existing = json.loads(detail_path.read_text())
            manifest_entry = json.loads(json.dumps(asdict(entry), sort_keys=True))
            if (
                existing.get("schema_version") == ANALYSIS_SCHEMA_VERSION
                and existing.get("run_config") == run_config
                and existing.get("manifest_entry") == manifest_entry
            ):
                details.append(existing)
                continue
        try:
            protein = load_crosslinked_protein(
                entry.source,
                source_format=entry.source_format,
                pdb_id=entry.pdb_id,
                chain_ids=entry.chain_ids or None,
                model_id=entry.model_id,
                backbone_atom=entry.backbone_atom,
                allowed_crosslink_types=entry.allowed_crosslink_types,
                crosslink_ids=entry.crosslink_ids or None,
                data_dir=output_path / "coordinate_cache",
                metadata={"sample_id": entry.sample_id, **entry.metadata},
            )
            lasso_motifs = analyze_local_lasso_motifs(protein)
            lasso_stability = analyze_lasso_density_stability(
                protein,
                baseline=lasso_motifs,
            )
            input_core = extract_crosslink_core(protein.graph)
            enumerate_exact = _should_enumerate_exact(
                settings,
                len(crosslink_edges(input_core)),
            )
            repulsion_result = None
            analysis_graph = protein.graph
            driver_config = DriverConfig(
                repulsor_root=(
                    Path(settings.repulsor_root)
                    if settings.repulsor_root is not None
                    else DriverConfig().repulsor_root
                ),
                verbose=False,
            )
            preliminary_analysis = None
            if settings.repulsion_steps > 0 and settings.repulsion_fallback_only:
                preliminary_analysis = analyze_crosslink_perturbations(
                    protein.graph,
                    fingerprinter=computer,
                    include_pairs=settings.include_pairs,
                    enumerate_all_subsets=enumerate_exact,
                    max_exact_crosslinks=settings.max_exact_crosslinks,
                )
            if (
                settings.repulsion_steps > 0
                and (
                    preliminary_analysis is None
                    or not preliminary_analysis.baseline.success
                )
            ):
                repulsion_result = relax_and_analyze_crosslinks(
                    protein.graph,
                    output_path
                    / "repulsion"
                    / _safe_sample_id(entry.sample_id)
                    / "natural",
                    fingerprinter=computer,
                    solver_options=SolverOptions(
                        steps=settings.repulsion_steps,
                        max_time=settings.repulsion_max_time,
                        free_special_vertices=(
                            settings.repulsion_free_special_vertices
                        ),
                    ),
                    decimation_options=DecimationOptions(
                        max_passes=settings.repulsion_decimation_passes,
                        min_points_per_edge=3,
                    ),
                    resampling_options=(
                        ResamplingOptions(
                            max_points_per_edge=(
                                settings.repulsion_max_points_per_edge
                            ),
                            min_points_per_edge=3,
                            allow_downsample=True,
                        )
                        if settings.repulsion_max_points_per_edge is not None
                        else None
                    ),
                    driver_config=driver_config,
                    allow_certificate_only=settings.allow_repulsor_certificate_only,
                    include_pairs=settings.include_pairs,
                    enumerate_all_subsets=enumerate_exact,
                    max_exact_crosslinks=settings.max_exact_crosslinks,
                )
                if repulsion_result.analysis is not None:
                    analysis = repulsion_result.analysis
                    analysis_graph = repulsion_result.relaxed_graph or protein.graph
                else:
                    analysis = analyze_crosslink_perturbations(
                        protein.graph,
                        fingerprinter=computer,
                        include_pairs=settings.include_pairs,
                        enumerate_all_subsets=enumerate_exact,
                        max_exact_crosslinks=settings.max_exact_crosslinks,
                    )
            elif preliminary_analysis is not None:
                analysis = preliminary_analysis
            else:
                analysis = analyze_crosslink_perturbations(
                    protein.graph,
                    fingerprinter=computer,
                    include_pairs=settings.include_pairs,
                    enumerate_all_subsets=enumerate_exact,
                    max_exact_crosslinks=settings.max_exact_crosslinks,
                )
            conditioned = (
                analyze_abstract_conditioned_robustness(
                    analysis_graph,
                    analysis=analysis,
                    fingerprinter=computer,
                    reference_seed=settings.null_seed,
                    include_pairs=settings.include_pairs,
                    max_subset_order=settings.conditioned_max_subset_order,
                )
                if settings.conditioned_robustness
                else None
            )
            bounded_generator = (
                search_minimum_generating_crosslink_sets(
                    analysis_graph,
                    fingerprinter=computer,
                    max_retained_crosslinks=(
                        settings.minimum_generator_max_retained_crosslinks
                    ),
                )
                if settings.minimum_generator_max_retained_crosslinks is not None
                else None
            )
            null_analyses = []
            conditioned_null_analyses = []
            null_summaries = []
            null_failures = []
            if settings.null_sampling_mode == "unique_disulfide_matchings":
                null_records = generate_unique_disulfide_null_graphs(
                    protein,
                    max_nulls=settings.null_replicates,
                    seed=settings.null_seed,
                    embedding_mode=settings.null_embedding_mode,
                )
            else:
                null_records = generate_null_graphs(
                    protein,
                    replicates=settings.null_replicates,
                    seed=settings.null_seed,
                    embedding_mode=settings.null_embedding_mode,
                )
            for null_record in null_records:
                if (
                    settings.repulsion_steps > 0
                    and settings.null_embedding_mode == "coordinate_preserving"
                    and not settings.repulsion_fallback_only
                ):
                    null_repulsion = relax_and_analyze_crosslinks(
                        null_record.graph,
                        output_path
                        / "repulsion"
                        / _safe_sample_id(entry.sample_id)
                        / f"null_{null_record.replicate}",
                        fingerprinter=computer,
                        solver_options=SolverOptions(
                            steps=settings.repulsion_steps,
                            max_time=settings.repulsion_max_time,
                            free_special_vertices=(
                                settings.repulsion_free_special_vertices
                            ),
                        ),
                        decimation_options=DecimationOptions(
                            max_passes=settings.repulsion_decimation_passes,
                            min_points_per_edge=3,
                        ),
                        resampling_options=(
                            ResamplingOptions(
                                max_points_per_edge=(
                                    settings.repulsion_max_points_per_edge
                                ),
                                min_points_per_edge=3,
                                allow_downsample=True,
                            )
                            if settings.repulsion_max_points_per_edge is not None
                            else None
                        ),
                        driver_config=driver_config,
                        allow_certificate_only=settings.allow_repulsor_certificate_only,
                        include_pairs=False,
                        enumerate_all_subsets=False,
                    )
                    if null_repulsion.analysis is not None:
                        null_analyses.append(null_repulsion.analysis)
                        conditioned_null = (
                            analyze_abstract_conditioned_robustness(
                                null_repulsion.relaxed_graph or null_record.graph,
                                analysis=null_repulsion.analysis,
                                fingerprinter=computer,
                                reference_seed=settings.null_seed,
                            )
                            if settings.conditioned_robustness
                            else None
                        )
                        if conditioned_null is not None:
                            conditioned_null_analyses.append(conditioned_null)
                        null_summaries.append(
                            _null_summary(
                                null_record,
                                null_repulsion.analysis,
                                null_repulsion,
                                conditioned_null,
                            )
                        )
                    else:
                        null_failures.append(
                            {
                                "replicate": null_record.replicate,
                                "status": null_repulsion.status,
                                "error_type": null_repulsion.error_type,
                                "error_message": null_repulsion.error_message,
                            }
                        )
                else:
                    null_analysis = analyze_crosslink_perturbations(
                        null_record.graph,
                        fingerprinter=computer,
                        include_pairs=False,
                        enumerate_all_subsets=False,
                    )
                    null_repulsion = None
                    conditioned_graph = null_record.graph
                    if (
                        settings.null_repulsion_fallback_steps > 0
                        and not null_analysis.baseline.success
                        and settings.null_embedding_mode == "coordinate_preserving"
                    ):
                        null_repulsion = relax_and_analyze_crosslinks(
                            null_record.graph,
                            output_path
                            / "repulsion"
                            / _safe_sample_id(entry.sample_id)
                            / f"null_fallback_{null_record.replicate}",
                            fingerprinter=computer,
                            solver_options=SolverOptions(
                                steps=settings.null_repulsion_fallback_steps,
                                max_time=settings.null_repulsion_fallback_max_time,
                                free_special_vertices=(
                                    settings.null_repulsion_fallback_free_special_vertices
                                ),
                            ),
                            decimation_options=DecimationOptions(
                                max_passes=(
                                    settings.null_repulsion_fallback_decimation_passes
                                ),
                                min_points_per_edge=3,
                            ),
                            resampling_options=(
                                ResamplingOptions(
                                    max_points_per_edge=(
                                        settings.null_repulsion_fallback_max_points_per_edge
                                    ),
                                    min_points_per_edge=3,
                                    allow_downsample=True,
                                )
                                if settings.null_repulsion_fallback_max_points_per_edge
                                is not None
                                else None
                            ),
                            driver_config=driver_config,
                            allow_certificate_only=(
                                settings.allow_repulsor_certificate_only
                            ),
                            include_pairs=False,
                            enumerate_all_subsets=False,
                        )
                        if null_repulsion.analysis is not None:
                            null_analysis = null_repulsion.analysis
                            conditioned_graph = (
                                null_repulsion.relaxed_graph or null_record.graph
                            )
                    null_analyses.append(null_analysis)
                    conditioned_null = (
                        analyze_abstract_conditioned_robustness(
                            conditioned_graph,
                            analysis=null_analysis,
                            fingerprinter=computer,
                            reference_seed=settings.null_seed,
                        )
                        if settings.conditioned_robustness
                        else None
                    )
                    if conditioned_null is not None:
                        conditioned_null_analyses.append(conditioned_null)
                    null_summaries.append(
                        _null_summary(
                            null_record,
                            null_analysis,
                            null_repulsion,
                            conditioned=conditioned_null,
                        )
                    )
                    if null_analysis.robustness_r1 is None:
                        null_failures.append(
                            {
                                "replicate": null_record.replicate,
                                "status": "fingerprint_error",
                                "error_type": null_analysis.baseline.error_type,
                                "error_message": null_analysis.baseline.error_message,
                            }
                        )
            comparison = (
                compare_robustness_to_null(analysis, null_analyses)
                if null_analyses
                and analysis.robustness_r1 is not None
                and any(
                    null_analysis.robustness_r1 is not None
                    for null_analysis in null_analyses
                )
                else None
            )
            conditioned_comparison = (
                compare_conditioned_topology_to_null(
                    conditioned,
                    conditioned_null_analyses,
                )
                if conditioned is not None
                and conditioned.topology_carrying_edge_fraction is not None
                and any(
                    result.topology_carrying_edge_fraction is not None
                    for result in conditioned_null_analyses
                )
                else None
            )
            analysis_core = extract_crosslink_core(analysis_graph)
            null_successful = sum(
                null_analysis.robustness_r1 is not None
                for null_analysis in null_analyses
            )
            null_ensemble_exhaustive = bool(null_summaries) and all(
                record.get("null_ensemble_exhaustive") is True
                for record in null_summaries
            )
            ensemble_sizes = {
                int(record["null_ensemble_size"])
                for record in null_summaries
                if record.get("null_ensemble_size") is not None
            }
            null_ensemble_size = (
                next(iter(ensemble_sizes)) if len(ensemble_sizes) == 1 else None
            )
            detail = {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_config": run_config,
                "manifest_entry": asdict(entry),
                "input": {
                    "source_path": str(protein.source_path),
                    "source_url": protein.source_url,
                    "downloaded": protein.downloaded,
                    "issues": protein.issues,
                    "excluded_crosslink_count": len(protein.excluded_crosslinks),
                },
                "summary": _summary_row(
                    entry,
                    protein,
                    analysis,
                    analysis_core,
                    lasso_motifs,
                    lasso_stability,
                    conditioned,
                    bounded_generator,
                    conditioned_comparison,
                    comparison,
                    repulsion_result,
                    settings.null_replicates,
                    len(null_analyses),
                    null_successful,
                    len(null_failures),
                    null_ensemble_exhaustive,
                    null_ensemble_size,
                ),
                "analysis": analysis.to_dict(),
                "abstract_connectivity_certificate": (
                    abstract_connectivity_certificate(analysis_core)
                ),
                "lasso_motifs": lasso_motifs.to_dict(),
                "lasso_density_stability": lasso_stability.to_dict(),
                "conditioned_robustness": (
                    conditioned.to_dict() if conditioned else None
                ),
                "bounded_minimum_generator": (
                    bounded_generator.to_dict() if bounded_generator else None
                ),
                "conditioned_null_comparison": (
                    conditioned_comparison.to_dict()
                    if conditioned_comparison
                    else None
                ),
                "repulsion": repulsion_result.to_dict() if repulsion_result else None,
                "null_comparison": asdict(comparison) if comparison else None,
                "null_summaries": null_summaries,
                "null_failures": null_failures,
            }
            sample_figure_dir = figure_dir / _safe_sample_id(entry.sample_id)
            figure, _ = plot_protein_graph_3d(
                analysis_graph,
                output_path=sample_figure_dir / "crosslinked_graph.png",
            )
            close_figure(figure)
            figure, _ = plot_edge_importance(
                analysis,
                output_path=sample_figure_dir / "edge_importance.png",
            )
            close_figure(figure)
            if analysis.pairs:
                figure, _ = plot_pair_synergy_heatmap(
                    analysis,
                    output_path=sample_figure_dir / "pair_synergy.png",
                )
                close_figure(figure)
            if comparison is not None:
                figure, _ = plot_natural_vs_null(
                    comparison,
                    output_path=sample_figure_dir / "natural_vs_null.png",
                )
                close_figure(figure)
        except Exception as exc:
            detail = {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_config": run_config,
                "manifest_entry": asdict(entry),
                "summary": {
                    "sample_id": entry.sample_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                "analysis": None,
            }
        _atomic_json(detail_path, detail)
        details.append(json.loads(detail_path.read_text()))
        _write_consolidated_outputs(output_path, details)
    _write_consolidated_outputs(output_path, details)
    _atomic_json(
        output_path / "run_config.json",
        {
            **run_config,
            "resume": settings.resume,
            "sample_count": len(details),
        },
    )
    return details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Yamada-fingerprint analysis of protein crosslinks."
    )
    parser.add_argument("manifest", type=Path, help="CSV protein manifest")
    parser.add_argument("output_dir", type=Path, help="Local result directory")
    parser.add_argument("--rotation-samples", type=int, default=10)
    parser.add_argument("--max-crossings", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument(
        "--exact-subsets", choices=("none", "auto", "all"), default="auto"
    )
    parser.add_argument("--max-exact-crosslinks", type=int, default=10)
    parser.add_argument("--null-replicates", type=int, default=0)
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument(
        "--null-embedding-mode",
        choices=("coordinate_preserving", "canonical_low_crossing"),
        default="coordinate_preserving",
    )
    parser.add_argument(
        "--null-sampling-mode",
        choices=("random_replicates", "unique_disulfide_matchings"),
        default="random_replicates",
        help=(
            "random rewires or exact unique intrachain-disulfide matchings "
            "sampled without replacement"
        ),
    )
    parser.add_argument(
        "--null-repulsion-fallback-steps",
        type=int,
        default=0,
        help=(
            "apply certificate-checked Repulsor relaxation only to "
            "coordinate-preserving nulls whose exact baseline exceeds the cap"
        ),
    )
    parser.add_argument(
        "--null-repulsion-fallback-max-time",
        type=float,
        default=1.0,
        help="wall-time budget in seconds for each null Repulsor fallback",
    )
    parser.add_argument(
        "--null-repulsion-fallback-free-special-vertices",
        action="store_true",
        help=(
            "allow crosslink/core vertices to move during certificate-checked "
            "null fallback relaxation"
        ),
    )
    parser.add_argument(
        "--null-repulsion-fallback-decimation-passes",
        type=int,
        default=8,
        help=(
            "maximum conservative shortcut passes after each null fallback "
            "relaxation"
        ),
    )
    parser.add_argument(
        "--null-repulsion-fallback-max-points-per-edge",
        type=int,
        help=(
            "safely pre-decimate long null-graph arcs toward this point cap "
            "before Repulsor fallback"
        ),
    )
    parser.add_argument("--repulsion-steps", type=int, default=0)
    parser.add_argument(
        "--repulsion-max-time",
        type=float,
        default=1.0,
        help="wall-time budget in seconds for each Repulsor preprocessing run",
    )
    parser.add_argument(
        "--repulsion-free-special-vertices",
        action="store_true",
        help="allow crosslink/core vertices to move during safe relaxation",
    )
    parser.add_argument(
        "--repulsion-decimation-passes",
        type=int,
        default=8,
        help="maximum conservative shortcut passes after Repulsor relaxation",
    )
    parser.add_argument(
        "--repulsion-max-points-per-edge",
        type=int,
        help=(
            "safely pre-decimate long graph arcs toward this point cap before "
            "Repulsor"
        ),
    )
    parser.add_argument(
        "--repulsion-fallback-only",
        action="store_true",
        help="skip Repulsor when the original graph is already exactly evaluable",
    )
    parser.add_argument("--repulsor-root", type=Path)
    parser.add_argument("--allow-repulsor-certificate-only", action="store_true")
    parser.add_argument(
        "--conditioned-robustness",
        action="store_true",
        help=(
            "compare natural and single-deletion embeddings with deterministic "
            "low-crossing references of the same abstract graph"
        ),
    )
    parser.add_argument(
        "--conditioned-max-subset-order",
        type=int,
        help=(
            "maximum deletion-subset order for strict conditioned "
            "cooperativity; defaults to two with pair scans and one otherwise"
        ),
    )
    parser.add_argument(
        "--minimum-generator-max-retained-crosslinks",
        type=int,
        help=(
            "rigorously search retained sets from size zero through this bound"
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = read_protein_manifest(args.manifest)
    run_protein_batch(
        entries,
        args.output_dir,
        fingerprint_settings=FingerprintSettings(
            num_rotation_samples=args.rotation_samples,
            max_crossings=args.max_crossings,
            n_jobs=args.n_jobs,
        ),
        batch_settings=ProteinBatchSettings(
            include_pairs=not args.no_pairs,
            exact_subsets=args.exact_subsets,
            max_exact_crosslinks=args.max_exact_crosslinks,
            null_replicates=args.null_replicates,
            null_seed=args.null_seed,
            null_embedding_mode=args.null_embedding_mode,
            null_sampling_mode=args.null_sampling_mode,
            null_repulsion_fallback_steps=args.null_repulsion_fallback_steps,
            null_repulsion_fallback_max_time=(
                args.null_repulsion_fallback_max_time
            ),
            null_repulsion_fallback_free_special_vertices=(
                args.null_repulsion_fallback_free_special_vertices
            ),
            null_repulsion_fallback_decimation_passes=(
                args.null_repulsion_fallback_decimation_passes
            ),
            null_repulsion_fallback_max_points_per_edge=(
                args.null_repulsion_fallback_max_points_per_edge
            ),
            repulsion_steps=args.repulsion_steps,
            repulsion_max_time=args.repulsion_max_time,
            repulsion_free_special_vertices=args.repulsion_free_special_vertices,
            repulsion_decimation_passes=args.repulsion_decimation_passes,
            repulsion_max_points_per_edge=args.repulsion_max_points_per_edge,
            repulsion_fallback_only=args.repulsion_fallback_only,
            repulsor_root=str(args.repulsor_root) if args.repulsor_root else None,
            allow_repulsor_certificate_only=args.allow_repulsor_certificate_only,
            conditioned_robustness=args.conditioned_robustness,
            conditioned_max_subset_order=args.conditioned_max_subset_order,
            minimum_generator_max_retained_crosslinks=(
                args.minimum_generator_max_retained_crosslinks
            ),
            resume=not args.no_resume,
        ),
    )
    print(args.output_dir / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
