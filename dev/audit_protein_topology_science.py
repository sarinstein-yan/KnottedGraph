"""Verify the frozen protein-topology Directions 1–6 result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(results_root: Path) -> dict[str, Any]:
    pattern = results_root / "pattern_validation_v1"
    higher = results_root / "higher_order_validation_v1"
    disulfide_pairs = results_root / "disulfide_pair_validation_v1"
    minimum = results_root / "minimum_generator_5osq_v1"
    population = results_root / "population_conditioned_recovered_v1"
    population_sensitivity = results_root / "population_conditioned_v1"
    recovery = results_root / "complexity_recovery_v1"

    pattern_summary = _rows(pattern / "summary.csv")
    pattern_states = _rows(pattern / "subset_states.csv")
    pattern_groups = _json(pattern / "pattern_groups.json")
    _require(len(pattern_summary) == 21, "Direction 1/3/4/5 cohort is not 21")
    _require(
        all(row["status"] == "ok" for row in pattern_summary),
        "Pattern-validation natural fingerprint failure",
    )
    _require(
        all(row["minimum_generating_crosslink_status"] == "ok" for row in pattern_summary),
        "Direction 3 minimum is not proven for every pattern protein",
    )
    _require(
        all(
            row["minimum_generating_crosslink_count"] == row["crosslink_count"]
            for row in pattern_summary
        ),
        "A pattern protein has a smaller topology-generating set",
    )
    _require(
        len(pattern_states) == 136
        and all(row["status"] == "ok" for row in pattern_states),
        "Direction 1 pattern lattices are incomplete",
    )
    lasso_groups = pattern_groups["same_local_lasso_motifs_different_fingerprints"]
    connectivity_groups = pattern_groups["same_connectivity_different_fingerprints"]
    _require(len(lasso_groups) == 6, "Direction 4 does not have six lasso groups")
    _require(
        len(connectivity_groups) == 2,
        "Direction 5 does not have two exact connectivity groups",
    )

    higher_detail = _json(higher / "analyses" / "5OSQ_A.json")
    conditioned = higher_detail["conditioned_robustness"]
    higher_subsets = _rows(higher / "conditioned_subset_impacts.csv")
    strict_pairs = [
        row
        for row in higher_subsets
        if row["strictly_cooperative"] == "True" and row["order"] == "2"
    ]
    strict_triples = [
        row
        for row in higher_subsets
        if row["strictly_cooperative"] == "True" and row["order"] == "3"
    ]
    _require(conditioned["status"] == "ok", "5OSQ conditioned analysis failed")
    _require(
        conditioned["successful_subset_count"] == 364
        and conditioned["failed_subset_count"] == 0,
        "5OSQ pair/triple scan is incomplete",
    )
    _require(
        conditioned["topology_carrying_edge_count"] == 3,
        "5OSQ carrying-edge count changed",
    )
    _require(
        len(strict_pairs) == 19 and len(strict_triples) == 3,
        "Direction 2 strict pair/triple counts changed",
    )

    disulfide_pair_summary = _rows(disulfide_pairs / "summary.csv")
    disulfide_pair_impacts = _rows(
        disulfide_pairs / "conditioned_subset_impacts.csv"
    )
    disulfide_pair_config = _json(disulfide_pairs / "run_config.json")
    _require(
        len(disulfide_pair_summary) == 82
        and all(
            row["status"] == "ok"
            and row["conditioned_status"] == "ok"
            and row["conditioned_maximum_subset_order"] == "2"
            for row in disulfide_pair_summary
        ),
        "The 82-protein disulfide pair-validation run is incomplete",
    )
    disulfide_pair_keys = {
        (row["sample_id"], row["removed_crosslink_ids"])
        for row in disulfide_pair_impacts
    }
    expected_pair_counts = {
        row["sample_id"]: math.comb(int(row["crosslink_count"]), 2)
        for row in disulfide_pair_summary
    }
    observed_pair_counts = Counter(
        row["sample_id"] for row in disulfide_pair_impacts
    )
    _require(
        len(disulfide_pair_impacts) == 207
        and len(disulfide_pair_keys) == 207
        and dict(observed_pair_counts) == expected_pair_counts
        and all(row["order"] == "2" for row in disulfide_pair_impacts),
        "The primary disulfide pair table is not 207 unique order-two subsets",
    )
    pair_batch_settings = disulfide_pair_config["batch_settings"]
    pair_fingerprint_settings = disulfide_pair_config["fingerprint_settings"]
    _require(
        pair_batch_settings["conditioned_max_subset_order"] == 2
        and pair_batch_settings["null_replicates"] == 0
        and pair_fingerprint_settings["max_crossings"] == 40
        and pair_fingerprint_settings["num_rotation_samples"] == 32,
        "The disulfide pair-validation configuration changed",
    )
    _require(
        not any(
            row["strictly_cooperative"] == "True"
            for row in disulfide_pair_impacts
        )
        and sum(
            row["information_carrying"] == "True"
            for row in disulfide_pair_impacts
        )
        == 6,
        "The primary disulfide pair result changed",
    )

    minimum_detail = _json(minimum / "analyses" / "5OSQ_A.json")
    generator = minimum_detail["bounded_minimum_generator"]
    _require(generator["status"] == "proven", "5OSQ minimum is not proven")
    _require(
        generator["successful_state_count"] == 8192
        and generator["failed_state_count"] == 0,
        "5OSQ complete retained-state lattice is incomplete",
    )
    _require(
        generator["proven_minimum_count"] == 13
        and len(generator["proven_minimum_sets"]) == 1,
        "5OSQ m_top result changed",
    )

    population_summary = _rows(population / "summary.csv")
    population_stats = _json(population / "dataset_statistics.json")
    population_conditioned = population_stats["abstract_conditioned"]
    _require(len(population_summary) == 114, "Population cohort is not 114")
    _require(
        all(
            row["status"] == "ok"
            and row["conditioned_status"] == "ok"
            and int(row["null_replicates_failed"]) == 0
            for row in population_summary
        ),
        "Population natural, conditioned, or null failure present",
    )
    null_analyzed = sum(int(row["null_replicates_analyzed"]) for row in population_summary)
    _require(null_analyzed == 398, "Selected unique-null count is not 398")
    for gate in (
        "replication_threshold_met",
        "sampling_design_declared",
        "informative_variation_present",
        "population_inference_ready",
        "claim_ready",
    ):
        _require(population_conditioned[gate] is True, f"Conditioned gate {gate} failed")
    difference = population_conditioned["natural_minus_matched_null_mean"]
    _require(
        math.isclose(difference["mean"], -0.07848420585262691),
        "Population effect size changed",
    )
    _require(
        difference["mean_bootstrap_95_ci"][1] < 0.0,
        "Population bootstrap interval includes zero",
    )
    _require(
        population_conditioned["paired_sign_flip_p_value"] < 0.001,
        "Population paired sign-flip p-value is not below 0.001",
    )
    per_sample_differences = [
        float(row["natural_minus_null_mean"])
        for row in population_conditioned["per_sample"]
    ]
    difference_sign_counts = {
        "negative": sum(value < 0.0 for value in per_sample_differences),
        "zero": sum(value == 0.0 for value in per_sample_differences),
        "positive": sum(value > 0.0 for value in per_sample_differences),
    }
    _require(
        difference_sign_counts == {"negative": 32, "zero": 78, "positive": 4},
        "Population per-protein effect-sign pattern changed",
    )
    leave_one_out_means = [
        sum(per_sample_differences[:index] + per_sample_differences[index + 1 :])
        / (len(per_sample_differences) - 1)
        for index in range(len(per_sample_differences))
    ]
    _require(
        max(leave_one_out_means) < 0.0,
        "Population effect sign is not stable under leave-one-protein-out analysis",
    )

    sensitivity_summary = _rows(population_sensitivity / "summary.csv")
    sensitivity_stats = _json(population_sensitivity / "dataset_statistics.json")
    sensitivity_conditioned = sensitivity_stats["abstract_conditioned"]
    _require(len(sensitivity_summary) == 82, "Nested sensitivity cohort is not 82")
    _require(
        all(
            row["status"] == "ok"
            and row["conditioned_status"] == "ok"
            and int(row["null_replicates_failed"]) == 0
            for row in sensitivity_summary
        ),
        "Nested sensitivity natural, conditioned, or null failure present",
    )
    sensitivity_nulls = sum(
        int(row["null_replicates_analyzed"]) for row in sensitivity_summary
    )
    _require(sensitivity_nulls == 297, "Nested sensitivity null count is not 297")
    sensitivity_difference = sensitivity_conditioned[
        "natural_minus_matched_null_mean"
    ]
    _require(
        math.isclose(sensitivity_difference["mean"], -0.06391896721165014),
        "Nested sensitivity effect size changed",
    )
    _require(
        sensitivity_difference["mean_bootstrap_95_ci"][1] < 0.0
        and sensitivity_conditioned["paired_sign_flip_p_value"] < 0.001,
        "Nested sensitivity inference no longer supports the primary result",
    )

    recovery_summary = _rows(recovery / "summary.csv")
    recovery_pairs = _rows(recovery / "conditioned_subset_impacts.csv")
    _require(len(recovery_summary) == 32, "Complexity-recovery cohort is not 32")
    _require(
        all(
            row["status"] == "ok" and row["conditioned_status"] == "ok"
            for row in recovery_summary
        ),
        "A high-complexity candidate was not recovered",
    )
    _require(
        len(recovery_pairs) == 85
        and not any(row["strictly_cooperative"] == "True" for row in recovery_pairs),
        "Recovered disulfide pair result changed",
    )
    all_disulfide_pairs = disulfide_pair_impacts + recovery_pairs
    _require(
        len(all_disulfide_pairs) == 292
        and not any(
            row["strictly_cooperative"] == "True" for row in all_disulfide_pairs
        ),
        "Combined 292-pair disulfide survey result changed",
    )

    return {
        "direction_1": {"pattern_states": 136, "five_osq_states": 8192},
        "direction_2": {
            "five_osq_strict_pairs": 19,
            "five_osq_strict_triples": 3,
            "disulfide_pairs": len(all_disulfide_pairs),
            "primary_disulfide_information_carrying_pairs": 6,
            "disulfide_strict_pairs": 0,
        },
        "direction_3": {"five_osq_m_top": 13},
        "direction_4": {"same_lasso_groups": 6},
        "direction_5": {"same_connectivity_groups": 2},
        "direction_6": {
            "natural_samples": 114,
            "selected_unique_nulls": null_analyzed,
            "mean_natural_minus_null": difference["mean"],
            "paired_sign_flip_p_value": population_conditioned[
                "paired_sign_flip_p_value"
            ],
            "per_protein_difference_sign_counts": difference_sign_counts,
            "leave_one_out_mean_range": [
                min(leave_one_out_means),
                max(leave_one_out_means),
            ],
            "claim_ready": population_conditioned["claim_ready"],
        },
        "direction_6_nested_sensitivity": {
            "natural_samples": 82,
            "selected_unique_nulls": sensitivity_nulls,
            "mean_natural_minus_null": sensitivity_difference["mean"],
            "paired_sign_flip_p_value": sensitivity_conditioned[
                "paired_sign_flip_p_value"
            ],
        },
        "complexity_recovery": {
            "samples": 32,
            "conditioned_pairs": 85,
            "strict_pairs": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        nargs="?",
        type=Path,
        default=Path("results/protein_topology"),
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.results_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
