from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from knotted_graph.applications.protein import (
    analyze_lasso_density_stability,
    analyze_local_lasso_motifs,
    is_nontrivial_lasso_class,
)
from knotted_graph.inputs import CrosslinkEndpoint, CrosslinkRecord, ResidueKey


def _protein(*, with_crosslink=True, gap=False):
    sequence_ids = [1, 2, 3, 4, 5, 6] if not gap else [1, 2, 3, 10, 11, 12]
    atom_records = [
        {
            "group": "ATOM",
            "atom_name": "CA",
            "chain_id": "A",
            "sequence_id": str(sequence_id),
            "insertion_code": "",
            "coord": np.asarray([float(index), 0.0, 0.0]),
        }
        for index, sequence_id in enumerate(sequence_ids)
    ]
    crosslinks = []
    if with_crosslink:
        crosslinks.append(
            CrosslinkRecord(
                crosslink_id="ssbond:1",
                kind="disulfide",
                endpoint_a=CrosslinkEndpoint(
                    ResidueKey("A", str(sequence_ids[1])), "CYS", "SG"
                ),
                endpoint_b=CrosslinkEndpoint(
                    ResidueKey("A", str(sequence_ids[4])), "CYS", "SG"
                ),
                source_record="SSBOND",
            )
        )
    return SimpleNamespace(
        pdb_id="TEST",
        source_format="pdb",
        source_path=Path("test.pdb"),
        source_url=None,
        downloaded=False,
        model_id=1,
        chain_ids=("A",),
        backbone_atom="CA",
        allowed_crosslink_types=("disulfide",),
        crosslinks=crosslinks,
        excluded_crosslinks=[],
        atom_records=atom_records,
        graph=nx.MultiGraph(),
        issues=[],
        metadata={},
    )


class _FakeTopoly:
    @staticmethod
    def lasso_type(
        coordinates,
        loop_indices,
        *,
        smooth,
        precision,
        density,
        min_dist,
        more_info,
    ):
        assert len(coordinates) == 6
        assert loop_indices == [(1, 4)]
        assert (smooth, precision, min_dist, more_info) == (
            0,
            0,
            (10, 3, 3),
            True,
        )
        assert density in {0, 1, 2}
        return {
            (1, 4): {
                "class": "L+1N",
                "beforeN": ["+0", "-1", "+2"],
                "beforeC": [],
                "crossingsN": ["+2"],
                "crossingsC": [],
                "Area": 12.5,
                "loop_length": 8.0,
                "Rg": 2.0,
                "smoothing_iterations": 0,
            }
        }


def test_lasso_adapter_builds_stable_signature_and_maps_residue_ids(monkeypatch):
    monkeypatch.setattr(
        "knotted_graph.applications.protein.motifs._load_topoly",
        lambda: _FakeTopoly,
    )
    monkeypatch.setattr(
        "knotted_graph.applications.protein.motifs._topoly_version",
        lambda: "test-version",
    )

    result = analyze_local_lasso_motifs(_protein())

    assert result.status == "ok"
    assert result.backend_version == "test-version"
    assert result.local_lasso_motif_signature == '[["disulfide","L+1N"]]'
    assert result.nontrivial_lasso_count == 1
    assert result.loops[0].crossings_n_indices == ("+2",)
    assert result.loops[0].crossings_n_residues == ("+A:3",)
    assert result.loops[0].surface_area == 12.5


def test_coordinate_gaps_are_audited_without_becoming_false_negatives(monkeypatch):
    monkeypatch.setattr(
        "knotted_graph.applications.protein.motifs._load_topoly",
        lambda: _FakeTopoly,
    )

    result = analyze_local_lasso_motifs(_protein(gap=True))

    assert result.status == "ok"
    assert result.local_lasso_motif_signature == '[["disulfide","L+1N"]]'
    assert result.coordinate_gaps == ("A:3->A:10:sequence_gap",)
    assert "straight segments" in result.issues[0]


def test_backend_failure_is_not_interpreted_as_no_lasso(monkeypatch):
    def unavailable():
        raise ImportError("topoly is not installed")

    monkeypatch.setattr(
        "knotted_graph.applications.protein.motifs._load_topoly", unavailable
    )

    result = analyze_local_lasso_motifs(_protein())

    assert result.status == "backend_unavailable"
    assert result.local_lasso_motif_signature is None
    assert result.loops[0].status == "backend_unavailable"


def test_no_eligible_covalent_loop_has_an_empty_complete_signature():
    result = analyze_local_lasso_motifs(_protein(with_crosslink=False))

    assert result.status == "ok"
    assert result.local_lasso_motif_signature == "[]"
    assert result.nontrivial_lasso_count == 0


def test_nontrivial_lasso_class_recognizes_topoly_classes():
    assert not is_nontrivial_lasso_class(None)
    assert not is_nontrivial_lasso_class("L0")
    assert is_nontrivial_lasso_class("L+1N")
    assert is_nontrivial_lasso_class("LL+1,-1")


def test_density_stability_requires_all_complete_signatures_to_match(monkeypatch):
    monkeypatch.setattr(
        "knotted_graph.applications.protein.motifs._load_topoly",
        lambda: _FakeTopoly,
    )
    protein = _protein()
    baseline = analyze_local_lasso_motifs(protein)

    stable = analyze_lasso_density_stability(protein, baseline=baseline)

    assert stable.status == "stable"
    assert stable.stable is True
    assert [run.density for run in stable.runs] == [0, 1, 2]


def test_density_stability_exposes_an_unstable_boundary_case(monkeypatch):
    class DensitySensitiveTopoly(_FakeTopoly):
        @staticmethod
        def lasso_type(*args, **kwargs):
            result = _FakeTopoly.lasso_type(*args, **kwargs)
            if kwargs["density"] == 2:
                result[(1, 4)]["class"] = "L0"
                result[(1, 4)]["crossingsN"] = []
            return result

    monkeypatch.setattr(
        "knotted_graph.applications.protein.motifs._load_topoly",
        lambda: DensitySensitiveTopoly,
    )

    result = analyze_lasso_density_stability(_protein())

    assert result.status == "unstable"
    assert result.stable is False
    assert result.runs[-1].signature == '[["disulfide","L0"]]'
