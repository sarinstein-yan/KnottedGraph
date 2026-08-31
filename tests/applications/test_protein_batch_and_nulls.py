import csv
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from knotted_graph.applications.protein import (
    FingerprintRecord,
    ProteinBatchSettings,
    ProteinManifestEntry,
    canonicalize_null_embedding,
    compare_robustness_to_null,
    crosslink_content_signature,
    crosslink_motif_signature,
    embedding_hash,
    find_pattern_candidate_pairs,
    find_same_crosslink_content_different_fingerprints,
    find_same_connectivity_different_fingerprints,
    find_same_local_lasso_motifs_different_fingerprints,
    find_same_motif_different_fingerprints,
    generate_unique_disulfide_null_graphs,
    randomize_crosslinks,
    read_protein_manifest,
    run_protein_batch,
    summarize_population_robustness,
)
from knotted_graph.inputs import CrosslinkEndpoint, CrosslinkRecord, ResidueKey


def _endpoint(sequence_id: int, residue_name: str = "CYS") -> CrosslinkEndpoint:
    atom_name = "SG" if residue_name == "CYS" else residue_name
    return CrosslinkEndpoint(
        residue=ResidueKey("A", str(sequence_id)),
        residue_name=residue_name,
        atom_name=atom_name,
    )


def _atom_records():
    records = []
    for sequence_id in range(1, 6):
        records.append(
            {
                "group": "ATOM",
                "atom_name": "CA",
                "residue_name": "CYS",
                "chain_id": "A",
                "sequence_id": str(sequence_id),
                "insertion_code": "",
                "coord": np.array([float(sequence_id), 0.0, 0.0]),
            }
        )
    return records


def test_crosslink_null_rewiring_is_reproducible_and_preserves_chemistry():
    source = [
        CrosslinkRecord(
            crosslink_id="native:1",
            kind="disulfide",
            endpoint_a=_endpoint(1),
            endpoint_b=_endpoint(3),
            source_record="SSBOND",
        )
    ]

    first = randomize_crosslinks(source, _atom_records(), chain_ids=["A"], seed=19)
    second = randomize_crosslinks(source, _atom_records(), chain_ids=["A"], seed=19)

    assert first == second
    assert first[0].kind == "disulfide"
    assert first[0].endpoint_a.residue_name == "CYS"
    assert first[0].endpoint_b.residue_name == "CYS"
    separation = abs(
        int(first[0].endpoint_a.residue.sequence_id)
        - int(first[0].endpoint_b.residue.sequence_id)
    )
    assert abs(separation - 2) <= 1
    assert frozenset(
        (first[0].endpoint_a.residue, first[0].endpoint_b.residue)
    ) != frozenset((source[0].endpoint_a.residue, source[0].endpoint_b.residue))


def test_disulfide_null_uses_a_non_native_perfect_endpoint_matching():
    source = [
        CrosslinkRecord(
            crosslink_id=f"native:{index}",
            kind="disulfide",
            endpoint_a=_endpoint(first),
            endpoint_b=_endpoint(second),
            source_record="SSBOND",
        )
        for index, (first, second) in enumerate(((1, 4), (2, 5), (3, 6)), 1)
    ]
    atom_records = _atom_records() + [
        {
            "group": "ATOM",
            "atom_name": "CA",
            "residue_name": "CYS",
            "chain_id": "A",
            "sequence_id": "6",
            "insertion_code": "",
            "coord": np.array([6.0, 0.0, 0.0]),
        }
    ]

    randomized = randomize_crosslinks(
        source,
        atom_records,
        chain_ids=["A"],
        seed=23,
    )

    original_pairs = {
        frozenset((record.endpoint_a.residue, record.endpoint_b.residue))
        for record in source
    }
    randomized_pairs = {
        frozenset((record.endpoint_a.residue, record.endpoint_b.residue))
        for record in randomized
    }
    randomized_endpoints = [
        endpoint.residue
        for record in randomized
        for endpoint in (record.endpoint_a, record.endpoint_b)
    ]
    assert randomized_pairs != original_pairs
    assert len(set(randomized_endpoints)) == 6
    assert set(randomized_endpoints) == {
        endpoint.residue
        for record in source
        for endpoint in (record.endpoint_a, record.endpoint_b)
    }
    assert all(
        record.source_record == "null_disulfide_perfect_matching"
        for record in randomized
    )


def test_unique_disulfide_null_enumerates_two_edge_ensemble_without_duplicates():
    source = [
        CrosslinkRecord(
            crosslink_id=f"native:{index}",
            kind="disulfide",
            endpoint_a=_endpoint(first),
            endpoint_b=_endpoint(second),
            source_record="SSBOND",
        )
        for index, (first, second) in enumerate(((1, 4), (2, 3)), 1)
    ]
    source_graph = nx.MultiGraph()
    source_graph.graph["input_id"] = "TEST_A_crosslinked"
    protein = SimpleNamespace(
        crosslinks=source,
        atom_records=_atom_records(),
        chain_ids=("A",),
        backbone_atom="CA",
        pdb_id="TEST",
        source_format="pdb",
        graph=source_graph,
    )

    nulls = generate_unique_disulfide_null_graphs(
        protein,
        max_nulls=20,
        seed=2026,
    )

    assert len(nulls) == 2
    pair_sets = {
        frozenset(
            frozenset((record.endpoint_a.residue, record.endpoint_b.residue))
            for record in null.crosslinks
        )
        for null in nulls
    }
    assert len(pair_sets) == 2
    assert all(null.graph.graph["null_ensemble_size"] == 2 for null in nulls)
    assert all(null.graph.graph["null_ensemble_exhaustive"] for null in nulls)
    assert all(
        record.source_record == "exact_unique_disulfide_perfect_matching"
        for null in nulls
        for record in null.crosslinks
    )


def test_null_repulsion_fallback_settings_validate():
    settings = ProteinBatchSettings(
        null_repulsion_fallback_steps=500,
        null_repulsion_fallback_max_time=10.0,
        null_repulsion_fallback_free_special_vertices=True,
        null_repulsion_fallback_decimation_passes=16,
        null_repulsion_fallback_max_points_per_edge=32,
    )

    assert settings.null_repulsion_fallback_free_special_vertices
    assert settings.null_repulsion_fallback_decimation_passes == 16
    assert settings.null_repulsion_fallback_max_points_per_edge == 32
    with pytest.raises(ValueError, match="decimation_passes must be at least 1"):
        ProteinBatchSettings(null_repulsion_fallback_decimation_passes=0)

    repulsion = ProteinBatchSettings(
        repulsion_steps=500,
        repulsion_max_time=10.0,
        repulsion_free_special_vertices=True,
        repulsion_decimation_passes=16,
        repulsion_max_points_per_edge=32,
        repulsion_fallback_only=True,
    )
    assert repulsion.repulsion_free_special_vertices
    assert repulsion.repulsion_decimation_passes == 16
    assert repulsion.repulsion_max_points_per_edge == 32
    assert repulsion.repulsion_fallback_only
    with pytest.raises(ValueError, match="repulsion_max_time must be positive"):
        ProteinBatchSettings(repulsion_max_time=0.0)
    with pytest.raises(ValueError, match="repulsion_decimation_passes"):
        ProteinBatchSettings(repulsion_decimation_passes=0)
    with pytest.raises(ValueError, match="max_points_per_edge must be at least 3"):
        ProteinBatchSettings(repulsion_max_points_per_edge=2)
    with pytest.raises(ValueError, match="max_points_per_edge must be at least 3"):
        ProteinBatchSettings(null_repulsion_fallback_max_points_per_edge=2)


def test_canonical_null_embedding_is_deterministic_and_has_clearance():
    graph = nx.MultiGraph()
    for index in range(4):
        graph.add_node(index, pos=np.asarray([float(index), 0.0, 0.0]))
    for index, (u, v) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))):
        graph.add_edge(
            u,
            v,
            key=f"edge:{index}",
            pts=np.vstack([graph.nodes[u]["pos"], graph.nodes[v]["pos"]]),
            edge_kind="backbone" if index < 4 else "crosslink",
            crosslink_id="x" if index == 4 else None,
            crosslink_type="covalent" if index == 4 else None,
        )

    first = canonicalize_null_embedding(graph, seed=123)
    second = canonicalize_null_embedding(graph, seed=123)

    assert embedding_hash(first) == embedding_hash(second)
    assert first.graph["null_embedding_mode"] == "canonical_low_crossing"
    assert first.graph["canonical_min_non_adjacent_segment_distance"] > 0.0
    assert all(len(data["pts"]) == 7 for *_, data in first.edges(data=True))


def test_null_robustness_comparison_uses_finite_sample_correction():
    natural = SimpleNamespace(robustness_r1=0.8)
    nulls = [SimpleNamespace(robustness_r1=value) for value in (0.2, 0.5, 0.9)]

    comparison = compare_robustness_to_null(natural, nulls)

    assert comparison.null_mean == pytest.approx(0.5333333333)
    assert comparison.empirical_p_greater_equal == pytest.approx(0.5)
    assert comparison.empirical_p_less_equal == pytest.approx(0.75)


def test_pattern_group_discovery_separates_the_two_hypotheses():
    rows = [
        {
            "sample_id": "a",
            "status": "ok",
            "motif_signature": "same-motif",
            "lasso_detection_status": "ok",
            "lasso_density_stable": True,
            "local_lasso_motif_signature": '[["disulfide","L+1N"]]',
            "nontrivial_lasso_count": 1,
            "crosslink_content_signature": "same-content",
            "abstract_connectivity_hash": "same-connectivity",
            "abstract_connectivity_class_id": "same-connectivity:iso0",
            "abstract_connectivity_isomorphism_verified": True,
            "baseline_fingerprint_id": "fp-1",
        },
        {
            "sample_id": "b",
            "status": "ok",
            "motif_signature": "same-motif",
            "lasso_detection_status": "ok",
            "lasso_density_stable": True,
            "local_lasso_motif_signature": '[["disulfide","L+1N"]]',
            "nontrivial_lasso_count": 1,
            "crosslink_content_signature": "same-content",
            "abstract_connectivity_hash": "different-connectivity",
            "abstract_connectivity_class_id": "different-connectivity:iso0",
            "abstract_connectivity_isomorphism_verified": True,
            "baseline_fingerprint_id": "fp-2",
        },
        {
            "sample_id": "c",
            "status": "ok",
            "motif_signature": "different-motif",
            "lasso_detection_status": "ok",
            "lasso_density_stable": True,
            "local_lasso_motif_signature": '[["disulfide","L+2C"]]',
            "nontrivial_lasso_count": 1,
            "crosslink_content_signature": "different-content",
            "abstract_connectivity_hash": "same-connectivity",
            "abstract_connectivity_class_id": "same-connectivity:iso0",
            "abstract_connectivity_isomorphism_verified": True,
            "baseline_fingerprint_id": "fp-3",
        },
    ]

    motif_groups = find_same_motif_different_fingerprints(rows)
    content_groups = find_same_crosslink_content_different_fingerprints(rows)
    connectivity_groups = find_same_connectivity_different_fingerprints(rows)
    lasso_groups = find_same_local_lasso_motifs_different_fingerprints(rows)

    assert motif_groups[0]["sample_ids"] == ["a", "b"]
    assert content_groups[0]["sample_ids"] == ["a", "b"]
    assert connectivity_groups[0]["sample_ids"] == ["a", "c"]
    assert lasso_groups[0]["sample_ids"] == ["a", "b"]
    candidates = find_pattern_candidate_pairs(rows)
    assert {
        (record["hypothesis"], record["sample_i"], record["sample_j"])
        for record in candidates
    } == {
        ("same_local_lasso_motifs_different_fingerprint", "a", "b"),
        ("same_crosslink_content_different_fingerprint", "a", "b"),
        ("same_connectivity_different_fingerprint", "a", "c"),
    }


def test_population_summary_keeps_matched_values_and_small_sample_warning():
    details = [
        {
            "summary": {
                "sample_id": "a",
                "pdb_id": "TEST",
                "status": "ok",
                "robustness_r1": 0.75,
                "null_replicates_requested": 2,
                "null_replicates_analyzed": 2,
                "null_replicates_failed": 0,
            },
            "null_summaries": [
                {"robustness_r1": 0.25},
                {"robustness_r1": 0.5},
            ],
        },
        {
            "summary": {
                "sample_id": "failed",
                "status": "fingerprint_error",
                "robustness_r1": None,
            }
        },
    ]

    summary = summarize_population_robustness(
        details,
        seed=12,
        bootstrap_samples=50,
    )

    assert summary["successful_natural_sample_count"] == 1
    assert summary["matched_sample_count"] == 1
    assert summary["natural_robustness"]["mean"] == 0.75
    assert summary["matched_null_mean_robustness"]["mean"] == 0.375
    assert summary["natural_minus_matched_null_mean"]["mean"] == 0.375
    assert summary["paired_sign_flip_p_value"] == 1.0
    assert not summary["replication_threshold_met"]
    assert not summary["population_inference_ready"]
    assert not summary["claim_ready"]
    assert summary["abstract_conditioned"]["successful_natural_sample_count"] == 0
    assert "fewer_than_20_matched_proteins" in summary["limitations"]
    assert "population_sampling_design_not_declared" in summary["limitations"]


def test_population_summary_accepts_exhaustive_small_null_ensembles():
    details = []
    for index in range(20):
        natural = 0.25 if index % 2 else 0.75
        natural_conditioned = 0.0 if index % 2 else 0.5
        details.append(
            {
                "manifest_entry": {
                    "metadata": {
                        "population_sampling_design": (
                            "sequence_redundancy_controlled"
                        )
                    }
                },
                "summary": {
                    "sample_id": f"sample-{index}",
                    "pdb_id": "TEST",
                    "status": "ok",
                    "robustness_r1": natural,
                    "baseline_embedding_nontrivial": natural_conditioned > 0,
                    "topology_carrying_edge_fraction": natural_conditioned,
                    "has_topology_carrying_edge": natural_conditioned > 0,
                    "null_replicates_requested": 20,
                    "null_replicates_analyzed": 2,
                    "null_replicates_failed": 0,
                    "null_ensemble_exhaustive": True,
                    "null_ensemble_size": 2,
                },
                "null_summaries": [
                    {
                        "robustness_r1": 0.1 + 0.1 * (index % 2),
                        "topology_carrying_edge_fraction": 0.0,
                    },
                        {
                            "robustness_r1": 0.2 + 0.1 * (index % 2),
                            "topology_carrying_edge_fraction": (
                                0.25 + 0.25 * (index % 2)
                            ),
                        },
                ],
            }
        )

    summary = summarize_population_robustness(
        details,
        seed=12,
        bootstrap_samples=50,
    )

    assert summary["replication_threshold_met"]
    assert summary["population_inference_ready"]
    assert summary["abstract_conditioned"]["replication_threshold_met"]
    assert summary["abstract_conditioned"]["population_inference_ready"]
    assert "fewer_than_20_successful_nulls_for_some_proteins" not in (
        summary["limitations"]
    )


def test_crosslink_content_signature_counts_chemistry_and_chain_scope():
    graph = nx.MultiGraph()
    graph.add_node("a", chain_id="A")
    graph.add_node("b", chain_id="A")
    graph.add_node("c", chain_id="B")
    graph.add_edge("a", "b", edge_kind="crosslink", crosslink_type="disulfide")
    graph.add_edge("a", "c", edge_kind="crosslink", crosslink_type="covalent")

    signature = crosslink_content_signature(graph)

    assert "disulfide" in signature
    assert "intra_chain" in signature
    assert "inter_chain" in signature
    assert crosslink_motif_signature(graph) == signature


def test_manifest_parser_resolves_relative_coordinate_path(tmp_path):
    coordinates = tmp_path / "small.pdb"
    coordinates.write_text("END\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,source,source_format,pdb_id,chain_ids,allowed_crosslink_types\n"
        "sample,small.pdb,pdb,TEST,A,disulfide|metal_coordination\n"
    )

    entries = read_protein_manifest(manifest)

    assert entries == [
        ProteinManifestEntry(
            sample_id="sample",
            source=str(coordinates.resolve()),
            source_format="pdb",
            pdb_id="TEST",
            chain_ids=("A",),
            allowed_crosslink_types=("disulfide", "metal_coordination"),
        )
    ]


def test_batch_writes_resumable_json_csv_and_figures(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    pdb_path = repository / "pdb-cache" / "1AOC.pdb"

    def fake_compute(self, graph, *, removed_crosslink_ids=(), metadata=None):
        removed = tuple(sorted(removed_crosslink_ids))
        terms = ((0, "baseline" if not removed else "removed:" + "|".join(removed)),)
        return FingerprintRecord(
            cache_key="|".join(removed) or "baseline",
            embedding_hash="hash",
            status="ok",
            polynomial=terms[0][1],
            canonical_terms=terms,
            fingerprint_id=terms[0][1],
            pd_code="",
            rotation_angles=None,
            rotation_order="ZYX",
            crossing_count=0,
            runtime_seconds=0.0,
            removed_crosslink_ids=removed,
        )

    monkeypatch.setattr(
        "knotted_graph.applications.protein.fingerprint.FingerprintComputer.compute",
        fake_compute,
    )
    output_dir = tmp_path / "results"
    entry = ProteinManifestEntry(
        sample_id="1AOC_A",
        source=str(pdb_path),
        pdb_id="1AOC",
        chain_ids=("A",),
        allowed_crosslink_types=("disulfide",),
    )
    settings = ProteinBatchSettings(
        include_pairs=False,
        exact_subsets="none",
        resume=True,
    )

    first = run_protein_batch([entry], output_dir, batch_settings=settings)
    second = run_protein_batch([entry], output_dir, batch_settings=settings)
    changed = run_protein_batch(
        [entry],
        output_dir,
        batch_settings=ProteinBatchSettings(
            include_pairs=True,
            exact_subsets="none",
            null_replicates=1,
            null_embedding_mode="canonical_low_crossing",
            repulsion_steps=1,
            repulsion_fallback_only=True,
            conditioned_robustness=True,
            conditioned_max_subset_order=3,
            minimum_generator_max_retained_crosslinks=1,
            resume=True,
        ),
    )

    assert first[0]["summary"]["status"] == "ok"
    assert first[0]["summary"]["abstract_connectivity_isomorphism_verified"]
    assert second == first
    assert changed[0]["run_config"]["batch_settings"]["include_pairs"] is True
    assert len(changed[0]["analysis"]["pairs"]) == 28
    assert changed[0]["null_summaries"][0]["embedding_mode"] == (
        "canonical_low_crossing"
    )
    assert changed[0]["null_summaries"][0]["min_non_adjacent_segment_distance"] > 0.0
    assert changed[0]["conditioned_robustness"]["status"] in {"ok", "partial"}
    assert changed[0]["conditioned_null_comparison"] is not None
    assert (output_dir / "analyses" / "1AOC_A.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "edge_impacts.csv").exists()
    assert (output_dir / "conditioned_edge_impacts.csv").exists()
    assert (output_dir / "conditioned_pair_impacts.csv").exists()
    assert (output_dir / "conditioned_subset_impacts.csv").exists()
    assert (output_dir / "subset_states.csv").exists()
    assert (output_dir / "minimum_generating_sets.csv").exists()
    assert (output_dir / "bounded_minimum_generating_sets.csv").exists()
    assert (output_dir / "pattern_groups.json").exists()
    assert (output_dir / "pattern_candidates.csv").exists()
    assert (output_dir / "dataset_statistics.json").exists()
    assert (output_dir / "robustness_by_sample.csv").exists()
    assert (output_dir / "conditioned_robustness_by_sample.csv").exists()
    assert (output_dir / "figures" / "1AOC_A" / "crosslinked_graph.png").exists()
    with (output_dir / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["crosslink_count"] == "8"
