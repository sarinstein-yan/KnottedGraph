import networkx as nx
import numpy as np
import pytest

from knotted_graph.core import ensure_embedding
from knotted_graph.extraction import (
    skeleton_image_to_graph,
    topology_aware_skeleton_image_to_graph,
)


def _trivalent_t_skeleton(size=25):
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    image[2 : c + 1, c, c] = True
    image[c : size - 2, c, c] = True
    image[c, c : size - 2, c] = True
    return image


def _diamond_ring(size=17):
    """A planar digital cycle with exactly two 26-neighbours per voxel."""
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    z = c
    radius = 5
    vertices = [
        (c - radius, c),
        (c, c + radius),
        (c + radius, c),
        (c, c - radius),
    ]
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
        steps = max(abs(x1 - x0), abs(y1 - y0))
        for step in range(steps):
            alpha = step / steps
            x = int(round(x0 + alpha * (x1 - x0)))
            y = int(round(y0 + alpha * (y1 - y0)))
            image[x, y, z] = True
    return image


def _two_diamond_rings(size=31):
    image = np.zeros((size, size, size), dtype=bool)
    for z in (8, 22):
        c = size // 2
        radius = 5
        vertices = [
            (c - radius, c),
            (c, c + radius),
            (c + radius, c),
            (c, c - radius),
        ]
        for (x0, y0), (x1, y1) in zip(
            vertices,
            vertices[1:] + vertices[:1],
        ):
            steps = max(abs(x1 - x0), abs(y1 - y0))
            for step in range(steps):
                alpha = step / steps
                x = int(round(x0 + alpha * (x1 - x0)))
                y = int(round(y0 + alpha * (y1 - y0)))
                image[x, y, z] = True
    return image


def test_topology_aware_collapses_voxel_junction_blob_with_valence_hint():
    graph = topology_aware_skeleton_image_to_graph(
        _trivalent_t_skeleton(),
        max_junction_degree=3,
    )
    assert isinstance(graph, nx.MultiGraph)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert sorted(dict(graph.degree()).values()) == [1, 1, 1, 3]
    ensure_embedding(graph, copy=False, normalize=False)


def test_topology_aware_edge_endpoints_match_node_positions():
    graph = topology_aware_skeleton_image_to_graph(
        _trivalent_t_skeleton(),
        max_junction_degree=3,
    )
    for u, v, data in graph.edges(data=True):
        points = np.asarray(data["pts"], dtype=float)
        u_pos = np.asarray(graph.nodes[u]["pos"], dtype=float)
        v_pos = np.asarray(graph.nodes[v]["pos"], dtype=float)
        direct = np.array_equal(points[0], u_pos) and np.array_equal(
            points[-1], v_pos
        )
        reverse = np.array_equal(points[0], v_pos) and np.array_equal(
            points[-1], u_pos
        )
        assert direct or reverse


def test_default_sparse_output_matches_poly2graph_exactly():
    image = _trivalent_t_skeleton()
    expected = skeleton_image_to_graph(image, backend="poly2graph")
    actual = skeleton_image_to_graph(image)

    assert list(expected.nodes()) == list(actual.nodes())
    assert list(expected.edges(keys=True)) == list(actual.edges(keys=True))
    for node in expected.nodes():
        assert np.array_equal(
            expected.nodes[node]["pos"],
            actual.nodes[node]["pos"],
        )
    for u, v, key in expected.edges(keys=True):
        assert np.array_equal(
            expected[u][v][key]["pts"],
            actual[u][v][key]["pts"],
        )
        assert expected[u][v][key]["weight"] == actual[u][v][key]["weight"]


def test_topology_aware_pure_ring_is_closed_self_loop():
    image = _diamond_ring()
    foreground = np.argwhere(image)
    foreground_set = {tuple(p) for p in foreground}
    for point in foreground:
        neighbours = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    q = tuple(point + np.array([dx, dy, dz]))
                    neighbours += q in foreground_set
        assert neighbours == 2

    graph = topology_aware_skeleton_image_to_graph(image)
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 1
    u, v, data = next(iter(graph.edges(data=True)))
    assert u == v
    points = np.asarray(data["pts"], dtype=float)
    node_pos = np.asarray(graph.nodes[u]["pos"], dtype=float)
    assert np.array_equal(points[0], node_pos)
    assert np.array_equal(points[-1], node_pos)
    ensure_embedding(graph, copy=False, normalize=False)


def test_disconnected_ring_components_are_all_preserved():
    graph = topology_aware_skeleton_image_to_graph(_two_diamond_rings())
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 2
    assert nx.number_connected_components(graph) == 2
    assert all(u == v for u, v in graph.edges())


def test_auto_dispatch_uses_topology_aware_for_3d(monkeypatch):
    import poly2graph

    def fail_if_called(_image):
        raise AssertionError("poly2graph should not be used for 3-D auto dispatch")

    monkeypatch.setattr(poly2graph, "skeleton2graph", fail_if_called)
    graph = skeleton_image_to_graph(_diamond_ring())
    assert graph.number_of_edges() == 1


def test_explicit_poly2graph_backend_remains_available():
    image = _diamond_ring()
    graph = skeleton_image_to_graph(image, backend="poly2graph")
    assert isinstance(graph, nx.MultiGraph)


def test_topology_aware_rejects_non_3d_input():
    with pytest.raises(ValueError, match="three-dimensional"):
        topology_aware_skeleton_image_to_graph(np.zeros((8, 8), dtype=bool))


def test_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        skeleton_image_to_graph(
            np.zeros((8, 8, 8), dtype=bool),
            backend="unknown",
        )
