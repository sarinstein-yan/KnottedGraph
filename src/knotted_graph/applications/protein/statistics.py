"""Dataset-level summaries for protein topology fingerprints."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from typing import Any, Iterable, Mapping, cast

import numpy as np


def _different_fingerprint_groups(
    rows: Iterable[Mapping[str, object]],
    group_key: str,
) -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or not row.get(group_key):
            continue
        groups[str(row[group_key])].append(row)
    output = []
    for key, members in sorted(groups.items()):
        fingerprints = sorted(
            {
                str(member["baseline_fingerprint_id"])
                for member in members
                if member.get("baseline_fingerprint_id")
            }
        )
        if len(fingerprints) < 2:
            continue
        output.append(
            {
                group_key: key,
                "sample_ids": sorted(str(member["sample_id"]) for member in members),
                "fingerprint_ids": fingerprints,
                "sample_count": len(members),
                "fingerprint_count": len(fingerprints),
            }
        )
    return output


def find_same_motif_different_fingerprints(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Find identical crosslink-content groups with distinct fingerprints."""

    return _different_fingerprint_groups(rows, "motif_signature")


def find_same_crosslink_content_different_fingerprints(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Find equal chemistry/scope count signatures with distinct fingerprints."""

    return _different_fingerprint_groups(rows, "crosslink_content_signature")


def find_same_connectivity_different_fingerprints(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Find abstractly identical graphs with distinct spatial fingerprints."""

    verified = (
        row
        for row in rows
        if row.get("abstract_connectivity_isomorphism_verified") is True
    )
    return _different_fingerprint_groups(verified, "abstract_connectivity_class_id")


def find_same_local_lasso_motifs_different_fingerprints(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Find complete, nontrivial lasso-motif groups with distinct fingerprints."""

    eligible = (
        row
        for row in rows
        if row.get("lasso_detection_status") == "ok"
        and row.get("lasso_density_stable") is True
        and int(cast(Any, row.get("nontrivial_lasso_count") or 0)) > 0
    )
    return _different_fingerprint_groups(eligible, "local_lasso_motif_signature")


def find_pattern_candidate_pairs(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Expand dataset pattern groups into inspectable protein pairs."""

    materialized = list(rows)
    candidates: list[dict[str, object]] = []
    hypotheses = (
        (
            "same_local_lasso_motifs_different_fingerprint",
            "local_lasso_motif_signature",
            True,
        ),
        (
            "same_crosslink_content_different_fingerprint",
            "crosslink_content_signature",
            False,
        ),
        (
            "same_connectivity_different_fingerprint",
            "abstract_connectivity_class_id",
            False,
        ),
    )
    for hypothesis, group_key, require_nontrivial_lasso in hypotheses:
        groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in materialized:
            if (
                row.get("status") == "ok"
                and row.get(group_key)
                and row.get("baseline_fingerprint_id")
                and (
                    hypothesis != "same_connectivity_different_fingerprint"
                    or row.get("abstract_connectivity_isomorphism_verified") is True
                )
                and (
                    not require_nontrivial_lasso
                    or (
                        row.get("lasso_detection_status") == "ok"
                        and row.get("lasso_density_stable") is True
                        and int(
                            cast(Any, row.get("nontrivial_lasso_count") or 0)
                        )
                        > 0
                    )
                )
            ):
                groups[str(row[group_key])].append(row)
        for group_value, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda row: str(row["sample_id"]))
            for first, second in combinations(ordered, 2):
                fingerprint_i = str(first["baseline_fingerprint_id"])
                fingerprint_j = str(second["baseline_fingerprint_id"])
                if fingerprint_i == fingerprint_j:
                    continue
                candidates.append(
                    {
                        "hypothesis": hypothesis,
                        "group_key": group_key,
                        "group_value": group_value,
                        "sample_i": str(first["sample_id"]),
                        "sample_j": str(second["sample_id"]),
                        "pdb_i": first.get("pdb_id"),
                        "pdb_j": second.get("pdb_id"),
                        "fingerprint_i": fingerprint_i,
                        "fingerprint_j": fingerprint_j,
                    }
                )
    return candidates


def _distribution_summary(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return {
            "count": 0,
            "values": [],
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "mean_bootstrap_95_ci": None,
        }
    if len(array) == 1:
        interval = [float(array[0]), float(array[0])]
        standard_deviation = 0.0
    else:
        samples = rng.choice(
            array,
            size=(bootstrap_samples, len(array)),
            replace=True,
        )
        interval = [
            float(value) for value in np.quantile(samples.mean(axis=1), [0.025, 0.975])
        ]
        standard_deviation = float(array.std(ddof=1))
    return {
        "count": int(len(array)),
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": standard_deviation,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean_bootstrap_95_ci": interval,
    }


def _paired_sign_flip_p_value(
    differences: Iterable[float],
    *,
    rng: np.random.Generator,
    monte_carlo_samples: int = 100_000,
) -> tuple[float | None, str | None]:
    array = np.asarray(list(differences), dtype=float)
    array = array[~np.isclose(array, 0.0)]
    if len(array) == 0:
        return 1.0, "all_differences_zero"
    observed = abs(float(array.mean()))
    if len(array) <= 20:
        exceed = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=len(array)):
            statistic = abs(float(np.mean(array * np.asarray(signs))))
            exceed += statistic >= observed - 1e-15
            total += 1
        return float(exceed / total), "exact"

    exceed = 0
    remaining = monte_carlo_samples
    while remaining:
        chunk = min(10_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(chunk, len(array)))
        statistics = np.abs((signs * array).mean(axis=1))
        exceed += int(np.sum(statistics >= observed - 1e-15))
        remaining -= chunk
    return float((exceed + 1) / (monte_carlo_samples + 1)), "monte_carlo"


def summarize_population_robustness(
    details: Iterable[Mapping[str, Any]],
    *,
    seed: int = 2026,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Summarize matched natural/null robustness across a protein cohort.

    This is deliberately descriptive for small cohorts. Population inference
    additionally requires a declared sampling design, nonzero variation, at
    least 20 proteins, and at least 20 successful nulls per matched protein.
    """

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    materialized = list(details)
    rng = np.random.default_rng(seed)
    per_sample = []
    natural_values = []
    null_means = []
    differences = []
    total_requested = 0
    total_analyzed = 0
    total_failed = 0
    for detail in materialized:
        summary = detail.get("summary") or {}
        if summary.get("status") != "ok" or summary.get("robustness_r1") is None:
            continue
        natural = float(summary["robustness_r1"])
        natural_values.append(natural)
        null_values = [
            float(record["robustness_r1"])
            for record in detail.get("null_summaries", [])
            if record.get("robustness_r1") is not None
        ]
        requested = int(summary.get("null_replicates_requested") or 0)
        analyzed = int(summary.get("null_replicates_analyzed") or 0)
        failed = int(summary.get("null_replicates_failed") or 0)
        ensemble_size_value = summary.get("null_ensemble_size")
        ensemble_size = (
            int(ensemble_size_value) if ensemble_size_value is not None else None
        )
        ensemble_exhaustive = bool(summary.get("null_ensemble_exhaustive")) and (
            ensemble_size is not None
            and failed == 0
            and len(null_values) == ensemble_size
        )
        total_requested += requested
        total_analyzed += analyzed
        total_failed += failed
        null_mean = float(np.mean(null_values)) if null_values else None
        difference = natural - null_mean if null_mean is not None else None
        if null_mean is not None:
            null_means.append(null_mean)
            differences.append(difference)
        per_sample.append(
            {
                "sample_id": summary.get("sample_id"),
                "pdb_id": summary.get("pdb_id"),
                "natural_robustness_r1": natural,
                "null_mean_robustness_r1": null_mean,
                "natural_minus_null_mean": difference,
                "null_replicates_requested": requested,
                "null_replicates_analyzed": analyzed,
                "null_replicates_failed": failed,
                "null_ensemble_exhaustive": ensemble_exhaustive,
                "null_ensemble_size": ensemble_size,
            }
        )

    paired_p_value, paired_method = _paired_sign_flip_p_value(
        differences,
        rng=rng,
    )
    matched_count = len(differences)
    replication_threshold_met = matched_count >= 20 and all(
        row["null_replicates_analyzed"] >= 20
        or row["null_ensemble_exhaustive"]
        for row in per_sample
        if row["null_mean_robustness_r1"] is not None
    )
    natural_variation = len(set(natural_values)) > 1
    null_variation = len(set(null_means)) > 1
    nonzero_matched_difference = any(
        not np.isclose(difference, 0.0) for difference in differences
    )
    sampling_design_declared = bool(materialized) and all(
        (detail.get("manifest_entry") or {})
        .get("metadata", {})
        .get("population_sampling_design")
        in {"representative", "sequence_redundancy_controlled"}
        for detail in materialized
        if (detail.get("summary") or {}).get("status") == "ok"
    )
    population_inference_ready = (
        replication_threshold_met
        and sampling_design_declared
        and natural_variation
        and null_variation
        and nonzero_matched_difference
    )
    limitations = []
    if matched_count < 20:
        limitations.append("fewer_than_20_matched_proteins")
    if any(
        row["null_mean_robustness_r1"] is not None
        and row["null_replicates_analyzed"] < 20
        and not row["null_ensemble_exhaustive"]
        for row in per_sample
    ):
        limitations.append("fewer_than_20_successful_nulls_for_some_proteins")
    if total_failed:
        limitations.append("null_fingerprint_failures_present")
    if not sampling_design_declared:
        limitations.append("population_sampling_design_not_declared")
    if not natural_variation:
        limitations.append("zero_variance_natural_robustness")
    if not null_variation:
        limitations.append("zero_variance_matched_null_means")
    if not nonzero_matched_difference:
        limitations.append("all_matched_differences_zero")

    conditioned_per_sample = []
    conditioned_values = []
    conditioned_null_means = []
    conditioned_differences = []
    conditioned_baseline_nontrivial_count = 0
    conditioned_carrying_sample_count = 0
    for detail in materialized:
        summary = detail.get("summary") or {}
        natural_conditioned = summary.get("topology_carrying_edge_fraction")
        baseline_nontrivial = summary.get("baseline_embedding_nontrivial")
        if summary.get("status") != "ok" or natural_conditioned is None:
            continue
        natural_conditioned = float(natural_conditioned)
        conditioned_values.append(natural_conditioned)
        conditioned_baseline_nontrivial_count += baseline_nontrivial is True
        conditioned_carrying_sample_count += bool(
            summary.get("has_topology_carrying_edge") is True
        )
        null_conditioned_values = [
            float(record["topology_carrying_edge_fraction"])
            for record in detail.get("null_summaries", [])
            if record.get("topology_carrying_edge_fraction") is not None
        ]
        null_conditioned_mean = (
            float(np.mean(null_conditioned_values))
            if null_conditioned_values
            else None
        )
        conditioned_difference = (
            natural_conditioned - null_conditioned_mean
            if null_conditioned_mean is not None
            else None
        )
        if null_conditioned_mean is not None:
            conditioned_null_means.append(null_conditioned_mean)
            conditioned_differences.append(conditioned_difference)
        conditioned_per_sample.append(
            {
                "sample_id": summary.get("sample_id"),
                "pdb_id": summary.get("pdb_id"),
                "baseline_embedding_nontrivial": baseline_nontrivial,
                "has_topology_carrying_edge": summary.get(
                    "has_topology_carrying_edge"
                ),
                "natural_topology_carrying_edge_fraction": natural_conditioned,
                "null_mean_topology_carrying_edge_fraction": (
                    null_conditioned_mean
                ),
                "natural_minus_null_mean": conditioned_difference,
                "successful_conditioned_null_count": len(null_conditioned_values),
                "null_ensemble_exhaustive": bool(
                    summary.get("null_ensemble_exhaustive")
                    and summary.get("null_ensemble_size") is not None
                    and len(null_conditioned_values)
                    == int(summary["null_ensemble_size"])
                ),
                "null_ensemble_size": summary.get("null_ensemble_size"),
            }
        )
    conditioned_p_value, conditioned_p_method = _paired_sign_flip_p_value(
        conditioned_differences,
        rng=rng,
    )
    conditioned_count = len(conditioned_values)
    conditioned_matched_count = len(conditioned_differences)
    conditioned_replication_threshold_met = (
        conditioned_matched_count >= 20
        and all(
            row["successful_conditioned_null_count"] >= 20
            or row["null_ensemble_exhaustive"]
            for row in conditioned_per_sample
            if row["null_mean_topology_carrying_edge_fraction"] is not None
        )
    )
    conditioned_natural_variation = len(set(conditioned_values)) > 1
    conditioned_null_variation = len(set(conditioned_null_means)) > 1
    conditioned_nonzero_difference = any(
        not np.isclose(value, 0.0) for value in conditioned_differences
    )
    conditioned_population_inference_ready = (
        conditioned_replication_threshold_met
        and sampling_design_declared
        and conditioned_natural_variation
        and conditioned_null_variation
        and conditioned_nonzero_difference
    )

    return {
        "schema_version": 3,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "input_sample_count": len(materialized),
        "successful_natural_sample_count": len(natural_values),
        "matched_sample_count": matched_count,
        "natural_robustness": _distribution_summary(
            natural_values,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        ),
        "matched_null_mean_robustness": _distribution_summary(
            null_means,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        ),
        "natural_minus_matched_null_mean": _distribution_summary(
            differences,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        ),
        "paired_sign_flip_p_value": paired_p_value,
        "paired_sign_flip_method": paired_method,
        "null_replicates_requested": total_requested,
        "null_replicates_analyzed": total_analyzed,
        "null_replicates_failed": total_failed,
        "replication_threshold_met": replication_threshold_met,
        "sampling_design_declared": sampling_design_declared,
        "informative_variation_present": (
            natural_variation and null_variation and nonzero_matched_difference
        ),
        "population_inference_ready": population_inference_ready,
        "claim_ready": population_inference_ready,
        "limitations": limitations,
        "per_sample": per_sample,
        "abstract_conditioned": {
            "successful_natural_sample_count": conditioned_count,
            "baseline_nontrivial_sample_count": (
                conditioned_baseline_nontrivial_count
            ),
            "baseline_nontrivial_sample_fraction": (
                conditioned_baseline_nontrivial_count / conditioned_count
                if conditioned_count
                else None
            ),
            "topology_carrying_edge_sample_count": (
                conditioned_carrying_sample_count
            ),
            "topology_carrying_edge_sample_fraction": (
                conditioned_carrying_sample_count / conditioned_count
                if conditioned_count
                else None
            ),
            "matched_sample_count": conditioned_matched_count,
            "natural_topology_carrying_edge_fraction": _distribution_summary(
                conditioned_values,
                rng=rng,
                bootstrap_samples=bootstrap_samples,
            ),
            "matched_null_mean_topology_carrying_edge_fraction": (
                _distribution_summary(
                    conditioned_null_means,
                    rng=rng,
                    bootstrap_samples=bootstrap_samples,
                )
            ),
            "natural_minus_matched_null_mean": _distribution_summary(
                conditioned_differences,
                rng=rng,
                bootstrap_samples=bootstrap_samples,
            ),
            "paired_sign_flip_p_value": conditioned_p_value,
            "paired_sign_flip_method": conditioned_p_method,
            "replication_threshold_met": conditioned_replication_threshold_met,
            "sampling_design_declared": sampling_design_declared,
            "informative_variation_present": (
                conditioned_natural_variation
                and conditioned_null_variation
                and conditioned_nonzero_difference
            ),
            "population_inference_ready": conditioned_population_inference_ready,
            "claim_ready": conditioned_population_inference_ready,
            "per_sample": conditioned_per_sample,
        },
    }
