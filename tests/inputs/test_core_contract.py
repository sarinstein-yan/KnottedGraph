import csv

import networkx as nx
import numpy as np

from knotted_graph.core.embedding import validate_embedding
from knotted_graph.inputs import (
    from_coordinate_chain,
    from_lammps_dump,
    from_spatial_graph_csv,
    write_lammps_dump,
)


def _assert_core_graph_contract(graph: nx.MultiGraph) -> None:
    assert isinstance(graph, nx.MultiGraph)
    assert validate_embedding(graph) == []
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    for node, data in graph.nodes(data=True):
        pos = np.asarray(data.get("pos"), dtype=float)
        assert pos.shape == (3,), node
        assert np.isfinite(pos).all(), node

    for u, v, key, data in graph.edges(keys=True, data=True):
        pts = np.asarray(data.get("pts"), dtype=float)
        assert pts.ndim == 2, (u, v, key)
        assert pts.shape[1] == 3, (u, v, key)
        assert pts.shape[0] >= 2, (u, v, key)
        assert np.isfinite(pts).all(), (u, v, key)

        u_pos = np.asarray(graph.nodes[u]["pos"], dtype=float)
        v_pos = np.asarray(graph.nodes[v]["pos"], dtype=float)
        direct = np.allclose(pts[0], u_pos) and np.allclose(pts[-1], v_pos)
        reverse = np.allclose(pts[0], v_pos) and np.allclose(pts[-1], u_pos)
        assert direct or reverse, (u, v, key)


def test_coordinate_chain_adapter_outputs_core_graph_contract():
    result = from_coordinate_chain(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        input_id="curve",
    )

    _assert_core_graph_contract(result.graph)


def test_polymer_adapter_outputs_core_graph_contract(tmp_path):
    path = tmp_path / "chain.dump"
    write_lammps_dump(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        path,
        molecule_id=7,
    )

    result = from_lammps_dump(path, molecule_id=7)

    _assert_core_graph_contract(result.graph)


def test_spatial_graph_csv_preserves_parallel_edge_keys_and_contract(tmp_path):
    nodes = tmp_path / "nodes.csv"
    edges = tmp_path / "edges.csv"

    with nodes.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [
                ["node_id", "x", "y", "z"],
                ["u", "0", "0", "0"],
                ["v", "1", "0", "0"],
            ]
        )

    with edges.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [
                ["edge_id", "source", "target", "points_json"],
                ["upper", "u", "v", "[[0, 0, 0], [0.5, 0.5, 0], [1, 0, 0]]"],
                ["lower", "u", "v", "[[0, 0, 0], [0.5, -0.5, 0], [1, 0, 0]]"],
            ]
        )

    result = from_spatial_graph_csv(nodes, edges)

    _assert_core_graph_contract(result.graph)
    assert set(result.graph.edges(keys=True)) == {
        ("u", "v", "upper"),
        ("u", "v", "lower"),
    }
