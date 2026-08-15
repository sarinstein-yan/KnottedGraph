import networkx as nx
import numpy as np
import pytest

from knotted_graph.core.embedding import (
    EmbeddingValidationError,
    contract_short_edges,
    ensure_embedding,
    is_embedding,
    validate_embedding,
)


def _two_node_graph() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.graph["domain"] = "demo"
    graph.add_node("u", pos=[0.0, 0.0, 0.0], label="U")
    graph.add_node("v", pos=[1.0, 0.0, 0.0], label="V")
    return graph


def test_ensure_embedding_adds_missing_edge_polyline_without_mutating_input():
    graph = _two_node_graph()
    graph.add_edge("u", "v", key="edge", color="red")

    embedded = ensure_embedding(graph)

    assert "pts" not in graph.edges["u", "v", "edge"]
    assert embedded.graph["domain"] == "demo"
    assert embedded.nodes["u"]["label"] == "U"
    assert embedded.edges["u", "v", "edge"]["color"] == "red"
    np.testing.assert_allclose(
        embedded.edges["u", "v", "edge"]["pts"],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )


def test_ensure_embedding_orients_reversed_edge_polyline_and_preserves_key():
    graph = _two_node_graph()
    graph.add_edge(
        "u",
        "v",
        key="upper",
        pts=np.array([[1.0, 0.0, 0.0], [0.5, 0.25, 0.0], [0.0, 0.0, 0.0]]),
    )

    embedded = ensure_embedding(graph)

    assert set(embedded.edges(keys=True)) == {("u", "v", "upper")}
    np.testing.assert_allclose(embedded.edges["u", "v", "upper"]["pts"][0], graph.nodes["u"]["pos"])
    np.testing.assert_allclose(embedded.edges["u", "v", "upper"]["pts"][-1], graph.nodes["v"]["pos"])


def test_validate_embedding_reports_coordinate_and_endpoint_issues():
    graph = nx.MultiGraph()
    graph.add_node("bad", pos=[0.0, 0.0])
    graph.add_node("v", pos=[1.0, 0.0, 0.0])
    graph.add_edge("bad", "v", key="edge", pts=[[0.0, 0.0, 0.0]])

    issues = validate_embedding(graph)

    assert any("node 'bad' pos must be a 3D point" in issue for issue in issues)
    assert any("edge ('bad', 'v', 'edge') pts must contain at least two points" in issue for issue in issues)
    assert not is_embedding(graph)
    with pytest.raises(EmbeddingValidationError, match="node 'bad'"):
        ensure_embedding(graph)


def test_validate_embedding_rejects_polyline_endpoint_mismatch():
    graph = _two_node_graph()
    graph.add_edge("u", "v", key="edge", pts=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    assert validate_embedding(graph) == [
        "edge ('u', 'v', 'edge') endpoints do not match node positions"
    ]


def test_contract_short_edges_relinks_incident_edge_endpoints():
    graph = nx.MultiGraph()
    graph.add_node("a", pos=[0.0, 0.0, 0.0])
    graph.add_node("b", pos=[0.1, 0.0, 0.0])
    graph.add_node("c", pos=[1.0, 0.0, 0.0])
    graph.add_edge("a", "b", key="short", pts=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    graph.add_edge("b", "c", key="long", pts=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])

    contracted = contract_short_edges(graph, min_length=0.2)

    assert contracted.number_of_nodes() == 2
    assert contracted.number_of_edges() == 1
    assert validate_embedding(contracted) == []

    kept = next(node for node in contracted if node != "c")
    np.testing.assert_allclose(contracted.nodes[kept]["pos"], [0.05, 0.0, 0.0])
    edge = next(iter(contracted.edges(data=True)))
    np.testing.assert_allclose(edge[2]["pts"][0], [0.05, 0.0, 0.0])
    np.testing.assert_allclose(edge[2]["pts"][-1], [1.0, 0.0, 0.0])
