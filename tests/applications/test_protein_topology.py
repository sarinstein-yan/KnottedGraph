from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import sympy as sp

from knotted_graph.applications.protein import (
    abstract_connectivity_certificate,
    abstract_connectivity_isomorphic,
    FingerprintComputer,
    FingerprintRecord,
    FingerprintSettings,
    analyze_abstract_conditioned_robustness,
    analyze_crosslink_perturbations,
    canonical_laurent_terms,
    embedding_hash,
    extract_crosslink_core,
    remove_crosslinks,
    search_minimum_generating_crosslink_sets,
)
from knotted_graph.core.embedding import validate_embedding


def _impact_graph(crosslink_ids=("x1", "x2", "x3")) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_edge(
        "u",
        "v",
        key="backbone",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [2.0, 0.0, 0.0]]),
        edge_kind="backbone",
        chain_id="A",
        residue_keys=[],
    )
    for index, crosslink_id in enumerate(crosslink_ids, 1):
        graph.add_edge(
            "u",
            "v",
            key=f"crosslink:{crosslink_id}",
            pts=np.array(
                [[0.0, 0.0, 0.0], [1.0, float(index), 0.2 * index], [2.0, 0.0, 0.0]]
            ),
            edge_kind="crosslink",
            crosslink_id=crosslink_id,
            crosslink_type="disulfide",
            endpoint_a={"residue": {"chain_id": "A", "sequence_id": "1"}},
            endpoint_b={"residue": {"chain_id": "A", "sequence_id": "9"}},
        )
    graph.graph.update(
        input_id="TEST_A_crosslinked",
        pdb_id="TEST",
        chain_ids=("A",),
    )
    return graph


def test_remove_crosslink_suppresses_backbone_anchor_geometry():
    graph = nx.MultiGraph()
    graph.add_node("a", pos=np.array([0.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_node("x", pos=np.array([1.0, 0.0, 0.0]), node_type="crosslink_residue")
    graph.add_node("b", pos=np.array([2.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_edge(
        "a",
        "x",
        key="ba",
        pts=np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [1.0, 0.0, 0.0]]),
        edge_kind="backbone",
        chain_id="A",
    )
    graph.add_edge(
        "x",
        "b",
        key="bb",
        pts=np.array([[1.0, 0.0, 0.0], [1.5, -0.1, 0.0], [2.0, 0.0, 0.0]]),
        edge_kind="backbone",
        chain_id="A",
    )
    graph.add_edge(
        "x",
        "b",
        key="crosslink:x1",
        pts=np.array([[1.0, 0.0, 0.0], [1.4, 0.5, 0.2], [2.0, 0.0, 0.0]]),
        edge_kind="crosslink",
        crosslink_id="x1",
        crosslink_type="disulfide",
    )

    perturbed = remove_crosslinks(graph, ["x1"])

    assert set(perturbed.nodes) == {"a", "b"}
    assert perturbed.number_of_edges() == 1
    assert validate_embedding(perturbed) == []
    points = next(iter(perturbed.edges(data="pts")))[2]
    np.testing.assert_allclose(
        points,
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.1, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, -0.1, 0.0],
            [2.0, 0.0, 0.0],
        ],
    )


def test_crosslink_core_prunes_open_protein_terminus():
    graph = nx.MultiGraph()
    graph.add_node("a", pos=np.array([0.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_node("x", pos=np.array([1.0, 0.0, 0.0]), node_type="crosslink_residue")
    graph.add_node("b", pos=np.array([2.0, 0.0, 0.0]), node_type="crosslink_residue")
    graph.add_edge(
        "a",
        "x",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_kind="backbone",
    )
    graph.add_edge(
        "x",
        "b",
        pts=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        edge_kind="backbone",
    )
    graph.add_edge(
        "x",
        "b",
        key="crosslink:x1",
        pts=np.array([[1.0, 0.0, 0.0], [1.5, 0.5, 0.2], [2.0, 0.0, 0.0]]),
        edge_kind="crosslink",
        crosslink_id="x1",
        crosslink_type="disulfide",
    )

    core = extract_crosslink_core(graph)

    assert set(core.nodes) == {"x", "b"}
    assert core.number_of_edges() == 2
    assert core.graph["core_crosslink_ids"] == ("x1",)
    assert validate_embedding(core) == []


def test_crosslink_core_removes_bridge_between_cyclic_blocks():
    graph = nx.MultiGraph()
    for index, node in enumerate(("a", "b", "c", "d")):
        graph.add_node(node, pos=np.array([float(index), 0.0, 0.0]))
    for u, v, crosslink_id in (("a", "b", "x1"), ("c", "d", "x2")):
        start = graph.nodes[u]["pos"]
        end = graph.nodes[v]["pos"]
        graph.add_edge(u, v, pts=np.vstack([start, end]), edge_kind="backbone")
        graph.add_edge(
            u,
            v,
            key=f"crosslink:{crosslink_id}",
            pts=np.vstack([start, (start + end) / 2 + [0.0, 0.5, 0.2], end]),
            edge_kind="crosslink",
            crosslink_id=crosslink_id,
            crosslink_type="disulfide",
        )
    graph.add_edge(
        "b",
        "c",
        key="between-cycles",
        pts=np.vstack([graph.nodes["b"]["pos"], graph.nodes["c"]["pos"]]),
        edge_kind="backbone",
    )

    core = extract_crosslink_core(graph)

    assert nx.number_connected_components(core) == 2
    assert not core.has_edge("b", "c", "between-cycles")
    assert core.graph["core_kind"] == "bridgeless_crosslink_supported_core"
    assert validate_embedding(core) == []


def test_embedding_hash_is_order_independent_and_geometry_sensitive():
    graph = _impact_graph(("x1",))
    reordered = nx.MultiGraph()
    for node in reversed(list(graph.nodes)):
        reordered.add_node(node, **graph.nodes[node])
    for u, v, key, data in reversed(list(graph.edges(keys=True, data=True))):
        copied = dict(data)
        copied["pts"] = np.asarray(data["pts"]).copy()
        reordered.add_edge(u, v, key=key, **copied)

    assert embedding_hash(graph) == embedding_hash(reordered)
    reordered["u"]["v"]["crosslink:x1"]["pts"][1, 2] += 0.1
    assert embedding_hash(graph) != embedding_hash(reordered)


def test_abstract_connectivity_certificate_supports_exact_labeled_isomorphism():
    first = _impact_graph(("x1", "x2"))
    relabeled = nx.relabel_nodes(first, {"u": "left", "v": "right"}, copy=True)

    assert abstract_connectivity_isomorphic(first, relabeled)
    assert abstract_connectivity_isomorphic(
        abstract_connectivity_certificate(first),
        abstract_connectivity_certificate(relabeled),
    )

    changed_label = relabeled.copy()
    for _, _, data in changed_label.edges(data=True):
        if data.get("edge_kind") == "crosslink":
            data["crosslink_type"] = "covalent"
            break
    assert not abstract_connectivity_isomorphic(first, changed_label)


def test_abstract_conditioned_robustness_removes_abstract_graph_change():
    class ConditionedFingerprinter:
        def compute(
            self,
            graph,
            *,
            removed_crosslink_ids=(),
            metadata=None,
        ):
            is_reference = graph.graph.get("null_embedding_mode") == (
                "canonical_low_crossing"
            )
            is_entangled = not is_reference and removed_crosslink_ids != ("x1",)
            terms = ((0, "entangled" if is_entangled else "reference"),)
            return FingerprintRecord(
                cache_key="|".join(removed_crosslink_ids) or "baseline",
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
                removed_crosslink_ids=removed_crosslink_ids,
                metadata=metadata or {},
            )

    result = analyze_abstract_conditioned_robustness(
        _impact_graph(("x1", "x2")),
        fingerprinter=ConditionedFingerprinter(),
    )

    assert result.status == "ok"
    assert result.baseline_embedding_nontrivial is True
    assert result.conditioned_state_robustness_r1 == 0.5
    assert result.entanglement_retention_r1 == 0.5
    assert result.topology_carrying_edge_count == 1
    assert result.topology_carrying_edge_fraction == 0.5
    assert result.has_topology_carrying_edge is True
    assert [record.information_carrying for record in result.singles] == [True, False]


def test_abstract_conditioned_pair_detects_cooperative_topology_loss():
    class CooperativeFingerprinter:
        def compute(
            self,
            graph,
            *,
            removed_crosslink_ids=(),
            metadata=None,
        ):
            is_reference = graph.graph.get("null_embedding_mode") == (
                "canonical_low_crossing"
            )
            is_entangled = not is_reference and len(removed_crosslink_ids) < 2
            terms = ((0, "entangled" if is_entangled else "reference"),)
            return FingerprintRecord(
                cache_key="|".join(removed_crosslink_ids) or "baseline",
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
                removed_crosslink_ids=removed_crosslink_ids,
                metadata=metadata or {},
            )

    result = analyze_abstract_conditioned_robustness(
        _impact_graph(("x1", "x2")),
        fingerprinter=CooperativeFingerprinter(),
        include_pairs=True,
    )

    assert [record.information_carrying for record in result.singles] == [False, False]
    assert result.cooperative_pair_count == 1
    assert result.successful_pair_count == 1
    assert result.failed_pair_count == 0
    assert result.pairs[0].information_carrying is True
    assert result.pairs[0].cooperative is True


def test_abstract_conditioned_subset_detects_strict_triple_cooperativity():
    class TripleCooperativeFingerprinter:
        def compute(
            self,
            graph,
            *,
            removed_crosslink_ids=(),
            metadata=None,
        ):
            is_reference = graph.graph.get("null_embedding_mode") == (
                "canonical_low_crossing"
            )
            is_entangled = not is_reference and len(removed_crosslink_ids) < 3
            terms = ((0, "entangled" if is_entangled else "reference"),)
            return FingerprintRecord(
                cache_key="|".join(removed_crosslink_ids) or "baseline",
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
                removed_crosslink_ids=removed_crosslink_ids,
                metadata=metadata or {},
            )

    result = analyze_abstract_conditioned_robustness(
        _impact_graph(("x1", "x2", "x3")),
        fingerprinter=TripleCooperativeFingerprinter(),
        max_subset_order=3,
    )

    triple = next(record for record in result.subsets if record.order == 3)
    assert all(record.information_carrying is False for record in result.singles)
    assert all(record.information_carrying is False for record in result.pairs)
    assert triple.information_carrying is True
    assert triple.strictly_cooperative is True
    assert result.maximum_subset_order_evaluated == 3
    assert result.strictly_cooperative_subset_count == 1
    assert result.minimum_information_carrying_subset_size == 3
    assert result.minimum_information_carrying_subsets == (("x1", "x2", "x3"),)


def test_abstract_conditioned_baseline_error_short_circuits_subsets():
    class ReferenceFailureFingerprinter:
        def compute(
            self,
            graph,
            *,
            removed_crosslink_ids=(),
            metadata=None,
        ):
            is_reference = graph.graph.get("null_embedding_mode") == (
                "canonical_low_crossing"
            )
            return FingerprintRecord(
                cache_key="|".join(removed_crosslink_ids) or "baseline",
                embedding_hash="hash",
                status="error" if is_reference else "ok",
                polynomial=None if is_reference else "observed",
                canonical_terms=() if is_reference else ((0, "observed"),),
                fingerprint_id=None if is_reference else "observed",
                pd_code="",
                rotation_angles=None,
                rotation_order="ZYX",
                crossing_count=0,
                runtime_seconds=0.0,
                removed_crosslink_ids=removed_crosslink_ids,
                error_type="ReferenceFailure" if is_reference else None,
                error_message="intentional" if is_reference else None,
                metadata=metadata or {},
            )

    result = analyze_abstract_conditioned_robustness(
        _impact_graph(("x1", "x2", "x3")),
        fingerprinter=ReferenceFailureFingerprinter(),
        max_subset_order=3,
    )

    assert result.status == "baseline_error"
    assert result.maximum_subset_order_evaluated == 0
    assert result.states == (result.baseline,)
    assert result.subsets == ()


def test_bounded_minimum_generator_search_proves_first_matching_level():
    class GeneratorFingerprinter:
        def compute(
            self,
            graph,
            *,
            removed_crosslink_ids=(),
            metadata=None,
        ):
            terms = (
                ((0, "full"),)
                if removed_crosslink_ids in ((), ("x1", "x3"))
                else ((0, "other"),)
            )
            return FingerprintRecord(
                cache_key="|".join(removed_crosslink_ids) or "baseline",
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
                removed_crosslink_ids=removed_crosslink_ids,
                metadata=metadata or {},
            )

    result = search_minimum_generating_crosslink_sets(
        _impact_graph(("x1", "x2", "x3")),
        fingerprinter=GeneratorFingerprinter(),
        max_retained_crosslinks=2,
    )

    assert result.status == "proven"
    assert result.proven_minimum_count == 1
    assert result.proven_minimum_sets == (("x2",),)
    assert result.proven_lower_bound == 1
    assert result.maximum_retained_size_evaluated == 1
    assert result.successful_state_count == 4


def test_bounded_minimum_generator_search_reports_rigorous_lower_bound():
    class FullOnlyFingerprinter:
        def compute(
            self,
            graph,
            *,
            removed_crosslink_ids=(),
            metadata=None,
        ):
            terms = ((0, "full" if not removed_crosslink_ids else "other"),)
            return FingerprintRecord(
                cache_key="|".join(removed_crosslink_ids) or "baseline",
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
                removed_crosslink_ids=removed_crosslink_ids,
                metadata=metadata or {},
            )

    result = search_minimum_generating_crosslink_sets(
        _impact_graph(("x1", "x2", "x3")),
        fingerprinter=FullOnlyFingerprinter(),
        max_retained_crosslinks=1,
    )

    assert result.status == "none_up_to_bound"
    assert result.proven_minimum_count is None
    assert result.proven_lower_bound == 2
    assert result.maximum_retained_size_evaluated == 1


def test_canonical_laurent_terms_accepts_negative_exponents():
    variable = sp.Symbol("A")

    terms = canonical_laurent_terms(
        2 * variable**3 - variable + 4 / variable**2 + variable**3,
        variable,
    )

    assert terms == ((-2, "4"), (1, "-1"), (3, "3"))


def test_fingerprint_computer_uses_canonical_disk_cache(tmp_path, monkeypatch):
    calls = []

    def fake_select(graph, **kwargs):
        calls.append((graph, kwargs))
        processor = SimpleNamespace(
            compute_yamada=lambda variable, **compute_kwargs: variable + 2 / variable
        )
        return SimpleNamespace(
            processor=processor,
            pd_code="V[0,1,2]",
            rotation_angles=(0.1, 0.2, 0.3),
            rotation_order="ZYX",
            num_crossings=2,
        )

    monkeypatch.setattr(
        "knotted_graph.applications.protein.fingerprint.select_projection",
        fake_select,
    )
    computer = FingerprintComputer(
        tmp_path,
        settings=FingerprintSettings(rotation_angles=(0.1, 0.2, 0.3)),
    )

    first = computer.compute(_impact_graph(("x1",)))
    second = computer.compute(_impact_graph(("x1",)))

    assert first.success
    assert first.canonical_terms == ((-1, "2"), (1, "1"))
    assert not first.from_cache
    assert second.from_cache
    assert second.same_fingerprint(first)
    assert len(calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_fingerprint_crossing_cap_records_projection_without_exact_evaluation(
    tmp_path,
    monkeypatch,
):
    processor = SimpleNamespace(
        compute_yamada=lambda *args, **kwargs: pytest.fail(
            "exact evaluator must not run"
        )
    )
    monkeypatch.setattr(
        "knotted_graph.applications.protein.fingerprint.select_projection",
        lambda graph, **kwargs: SimpleNamespace(
            processor=processor,
            pd_code="X[too,many,crossings]",
            rotation_angles=(0.1, 0.2, 0.3),
            rotation_order="ZYX",
            num_crossings=7,
        ),
    )
    computer = FingerprintComputer(
        tmp_path,
        settings=FingerprintSettings(max_crossings=6),
    )

    record = computer.compute(_impact_graph(("x1",)))

    assert not record.success
    assert record.error_type == "FingerprintComplexityError"
    assert record.crossing_count == 7
    assert record.pd_code == "X[too,many,crossings]"


def test_empty_crosslink_core_has_explicit_trivial_fingerprint(tmp_path):
    empty = nx.MultiGraph()
    computer = FingerprintComputer(tmp_path)

    record = computer.compute(empty, removed_crosslink_ids=("x1",))

    assert record.success
    assert record.polynomial == "1"
    assert record.canonical_terms == ((0, "1"),)
    assert record.metadata["empty_core_convention"] == "Y(empty)=1"


class _SubsetFingerprinter:
    changed_subsets = {("x3",), ("x1", "x2")}

    def compute(self, graph, *, removed_crosslink_ids=(), metadata=None):
        removed = tuple(sorted(removed_crosslink_ids))
        terms = (
            ((0, "changed"),) if removed in self.changed_subsets else ((0, "baseline"),)
        )
        return FingerprintRecord(
            cache_key="/".join(removed) or "baseline",
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


def test_perturbation_analysis_finds_single_and_cooperative_effects():
    analysis = analyze_crosslink_perturbations(
        _impact_graph(),
        fingerprinter=_SubsetFingerprinter(),
        include_pairs=True,
        enumerate_all_subsets=True,
        max_exact_crosslinks=3,
    )

    assert [record.changed for record in analysis.singles] == [False, False, True]
    cooperative = [record for record in analysis.pairs if record.cooperative]
    assert [(record.crosslink_i, record.crosslink_j) for record in cooperative] == [
        ("x1", "x2")
    ]
    assert analysis.topological_fraction == pytest.approx(1 / 3)
    assert analysis.robustness_r1 == pytest.approx(2 / 3)
    assert set(analysis.minimal_changed_subsets) == {("x3",), ("x1", "x2")}
    assert analysis.minimum_cardinality_subsets == [("x3",)]
    assert analysis.minimum_generating_crosslink_count == 0
    assert analysis.minimum_generating_crosslink_sets == [()]
    assert analysis.minimum_generating_crosslink_status == "ok"
    assert len(analysis.subsets) == 7


def test_minimum_generating_set_is_retained_complement_not_removed_subset():
    class GeneratingFingerprinter(_SubsetFingerprinter):
        changed_subsets = {
            ("x1",),
            ("x2",),
            ("x3",),
            ("x1", "x2"),
            ("x1", "x3"),
            ("x1", "x2", "x3"),
        }

    analysis = analyze_crosslink_perturbations(
        _impact_graph(),
        fingerprinter=GeneratingFingerprinter(),
        enumerate_all_subsets=True,
        max_exact_crosslinks=3,
    )

    # Removing x2+x3 leaves x1 and reproduces the full fingerprint.  The
    # generating set is therefore the retained complement (x1), not (x2,x3).
    assert analysis.minimum_generating_crosslink_count == 1
    assert analysis.minimum_generating_crosslink_sets == [("x1",)]
    assert analysis.minimum_generating_crosslink_status == "ok"


def test_minimum_generating_set_requires_complete_enumeration():
    analysis = analyze_crosslink_perturbations(
        _impact_graph(),
        fingerprinter=_SubsetFingerprinter(),
        enumerate_all_subsets=False,
    )

    assert analysis.minimum_generating_crosslink_count is None
    assert analysis.minimum_generating_crosslink_sets == []
    assert analysis.minimum_generating_crosslink_status == "not_enumerated"


def test_exact_subset_limit_is_enforced():
    with pytest.raises(ValueError, match="exceeding max_exact_crosslinks"):
        analyze_crosslink_perturbations(
            _impact_graph(),
            fingerprinter=_SubsetFingerprinter(),
            enumerate_all_subsets=True,
            max_exact_crosslinks=2,
        )
