import networkx as nx
import numpy as np
import pytest

from knotted_graph.core.embedding import (
    EmbeddingValidationError,
    contract_short_edges,
    ensure_embedding,
    is_embedding,
    simplify_edges,
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


def test_simplify_edges_preserves_acyclic_path_geometry_and_metadata():
    graph = nx.MultiGraph(project="path-demo")
    graph.add_node("a", pos=[0.0, 0.0, 0.0], role="start")
    graph.add_node("b", pos=[1.0, 0.5, 0.0], role="bend")
    graph.add_node("c", pos=[2.0, 0.0, 0.0], role="end")
    graph.add_edge(
        "a",
        "b",
        key="first",
        pts=[[0.0, 0.0, 0.0], [0.4, 0.3, 0.0], [1.0, 0.5, 0.0]],
        segment_id="left",
    )
    graph.add_edge(
        "b",
        "c",
        key="second",
        pts=[[1.0, 0.5, 0.0], [1.6, 0.3, 0.0], [2.0, 0.0, 0.0]],
        segment_id="right",
    )

    simplified = simplify_edges(graph)

    assert simplified.graph["project"] == "path-demo"
    assert simplified.number_of_nodes() == 3
    assert simplified.number_of_edges() == 2
    assert {data["role"] for _, data in simplified.nodes(data=True)} == {"start", "bend", "end"}
    assert {data["segment_id"] for *_, data in simplified.edges(data=True)} == {"left", "right"}
    assert sorted(len(data["pts"]) for *_, data in simplified.edges(data=True)) == [3, 3]
    assert validate_embedding(simplified) == []


def test_simplify_edges_preserves_branched_tree_edges_and_metadata():
    graph = nx.MultiGraph(system="tree-demo")
    positions = {
        "center": [0.0, 0.0, 0.0],
        "north": [0.0, 1.0, 0.0],
        "east": [1.0, 0.0, 0.0],
        "west": [-1.0, 0.0, 0.0],
    }
    for node, pos in positions.items():
        graph.add_node(node, pos=pos, source_id=node)
    for branch_id, leaf in enumerate(("north", "east", "west"), start=1):
        graph.add_edge("center", leaf, branch_id=branch_id)

    simplified = simplify_edges(graph)

    assert simplified.graph["system"] == "tree-demo"
    assert simplified.number_of_nodes() == 4
    assert simplified.number_of_edges() == 3
    assert {data["source_id"] for _, data in simplified.nodes(data=True)} == set(positions)
    assert {data["branch_id"] for *_, data in simplified.edges(data=True)} == {1, 2, 3}
    assert all(np.asarray(data["pts"]).shape == (2, 3) for *_, data in simplified.edges(data=True))
    assert validate_embedding(simplified) == []


def test_simplify_edges_preserves_acyclic_component_beside_cycle():
    graph = nx.MultiGraph(project="mixed-components")
    positions = {
        "c0": [0.0, 0.0, 0.0],
        "c1": [1.0, 0.0, 0.0],
        "c2": [0.5, 1.0, 0.0],
        "p0": [3.0, 0.0, 0.0],
        "p1": [4.0, 0.0, 0.0],
    }
    for node, pos in positions.items():
        graph.add_node(node, pos=pos, source_id=node)
    for u, v in (("c0", "c1"), ("c1", "c2"), ("c2", "c0")):
        graph.add_edge(u, v, component="cycle")
    graph.add_edge("p0", "p1", key="path", component="path", sample="keep-me")

    simplified = simplify_edges(graph)

    assert simplified.graph["project"] == "mixed-components"
    assert nx.number_connected_components(simplified) == 2
    assert simplified.number_of_edges() == 2
    path_edges = [
        data for *_, data in simplified.edges(data=True) if data.get("component") == "path"
    ]
    assert len(path_edges) == 1
    assert path_edges[0]["sample"] == "keep-me"
    np.testing.assert_allclose(path_edges[0]["pts"], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert validate_embedding(simplified) == []
