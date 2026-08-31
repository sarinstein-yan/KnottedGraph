from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from knotted_graph.applications.protein import (
    FingerprintRecord,
    check_repulsor_availability,
    relax_and_analyze_crosslinks,
)
from knotted_graph.layout.repulsive import DriverConfig


def _graph() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]), node_type="chain_endpoint")
    graph.add_edge(
        "u",
        "v",
        key="backbone",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [2.0, 0.0, 0.0]]),
        edge_kind="backbone",
    )
    graph.add_edge(
        "u",
        "v",
        key="crosslink:x1",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.2], [2.0, 0.0, 0.0]]),
        edge_kind="crosslink",
        crosslink_id="x1",
        crosslink_type="disulfide",
    )
    graph.graph.update(input_id="demo", chain_ids=("A",), fingerprint_group="native")
    return graph


class _GraphPropertyFingerprinter:
    def compute(self, graph, *, removed_crosslink_ids=(), metadata=None):
        removed = tuple(sorted(removed_crosslink_ids))
        value = str(graph.graph.get("fingerprint_group", "native"))
        terms = ((0, value),)
        return FingerprintRecord(
            cache_key=value + "|" + "|".join(removed),
            embedding_hash="hash",
            status="ok",
            polynomial=value,
            canonical_terms=terms,
            fingerprint_id=value,
            pd_code="",
            rotation_angles=None,
            rotation_order="ZYX",
            crossing_count=0,
            runtime_seconds=0.0,
            removed_crosslink_ids=removed,
        )


class _InitiallyCappedFingerprinter(_GraphPropertyFingerprinter):
    def compute(self, graph, *, removed_crosslink_ids=(), metadata=None):
        if graph.graph.get("fingerprint_group") == "native":
            return FingerprintRecord(
                cache_key="capped",
                embedding_hash="hash",
                status="error",
                polynomial=None,
                canonical_terms=(),
                fingerprint_id=None,
                pd_code="X[many]",
                rotation_angles=None,
                rotation_order="ZYX",
                crossing_count=40,
                runtime_seconds=0.0,
                removed_crosslink_ids=tuple(sorted(removed_crosslink_ids)),
                error_type="FingerprintComplexityError",
                error_message="crossing cap",
            )
        return super().compute(
            graph,
            removed_crosslink_ids=removed_crosslink_ids,
            metadata=metadata,
        )


def test_repulsor_relaxation_is_validated_before_deletion_scan(tmp_path, monkeypatch):
    captured = {}

    def fake_relax(graph, workspace, **kwargs):
        captured["graph"] = graph
        captured["kwargs"] = kwargs
        relaxed = graph.copy()
        relaxed["u"]["v"]["crosslink:x1"]["pts"][1, 2] += 0.4
        return SimpleNamespace(graph=relaxed, metadata={"solver": "fake"})

    monkeypatch.setattr(
        "knotted_graph.applications.protein.repulsion.relax_spatial_graph",
        fake_relax,
    )

    result = relax_and_analyze_crosslinks(
        _graph(),
        tmp_path / "layout",
        fingerprinter=_GraphPropertyFingerprinter(),
        include_pairs=False,
    )

    assert result.status == "ok"
    assert result.topology_preserved is True
    assert result.analysis is not None
    assert len(result.analysis.singles) == 1
    assert result.analysis.metadata["repulsor"]["topology_preserved"] is True
    assert captured["graph"].graph["core_kind"] == (
        "bridgeless_crosslink_supported_core"
    )
    assert captured["kwargs"]["pin_graph_nodes"] is False
    assert captured["kwargs"]["decimation_options"].min_points_per_edge == 3


def test_repulsor_fingerprint_mismatch_stops_perturbations(tmp_path, monkeypatch):
    def fake_relax(graph, workspace, **kwargs):
        relaxed = graph.copy()
        relaxed.graph["fingerprint_group"] = "changed"
        return SimpleNamespace(graph=relaxed, metadata={"solver": "fake"})

    monkeypatch.setattr(
        "knotted_graph.applications.protein.repulsion.relax_spatial_graph",
        fake_relax,
    )

    result = relax_and_analyze_crosslinks(
        _graph(),
        tmp_path / "layout",
        fingerprinter=_GraphPropertyFingerprinter(),
    )

    assert result.status == "topology_mismatch"
    assert result.topology_preserved is False
    assert result.analysis is None
    assert result.error_type == "TopologyMismatch"


def test_repulsor_layout_error_is_a_structured_result(tmp_path, monkeypatch):
    def fake_relax(graph, workspace, **kwargs):
        raise FileNotFoundError("missing native dependency")

    monkeypatch.setattr(
        "knotted_graph.applications.protein.repulsion.relax_spatial_graph",
        fake_relax,
    )

    result = relax_and_analyze_crosslinks(
        _graph(),
        tmp_path / "layout",
        fingerprinter=_GraphPropertyFingerprinter(),
    )

    assert result.status == "layout_error"
    assert result.error_type == "FileNotFoundError"
    assert "missing native dependency" in result.error_message


def test_valid_safe_step_certificate_can_gate_initially_capped_graph(
    tmp_path,
    monkeypatch,
):
    def fake_relax(graph, workspace, **kwargs):
        relaxed = graph.copy()
        relaxed.graph["fingerprint_group"] = "relaxed"
        return SimpleNamespace(
            graph=relaxed,
            metadata={"certificate": {"valid": True}},
        )

    monkeypatch.setattr(
        "knotted_graph.applications.protein.repulsion.relax_spatial_graph",
        fake_relax,
    )

    result = relax_and_analyze_crosslinks(
        _graph(),
        tmp_path / "layout",
        fingerprinter=_InitiallyCappedFingerprinter(),
        allow_certificate_only=True,
        include_pairs=False,
    )

    assert result.status == "certificate_only"
    assert result.topology_preserved is None
    assert result.analysis is not None
    assert result.analysis.metadata["repulsor"]["validation_mode"] == (
        "repulsor_safe_step_certificate"
    )


def test_repulsor_availability_check_does_not_compile(tmp_path):
    root = tmp_path / "Repulsor"
    root.mkdir()
    (root / "Repulsor.hpp").write_text("// test")
    config = DriverConfig(
        repulsor_root=root,
        driver_source=Path(__file__),
        driver_binary=tmp_path / "not-built",
        verbose=False,
    )

    availability = check_repulsor_availability(config)

    assert availability.available
    assert availability.header_exists
    assert availability.driver_source_exists
    assert not availability.driver_binary_exists
