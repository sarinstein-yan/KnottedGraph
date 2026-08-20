import networkx as nx
import numpy as np
import pytest

from knotted_graph.core import ensure_embedding
from knotted_graph.extraction import (
    skeleton_image_to_graph,
    topology_aware_skeleton_image_to_graph,
)


def _trivalent_t_skeleton(size=15):
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    # Three long arms meeting at one physical junction. Under 26-connectivity,
    # voxels adjacent to the junction form a small multi-voxel branch blob.
    image[2 : c + 1, c, c] = True
    image[c : size - 2, c, c] = True
    image[c, c : size - 2, c] = True
    return image


def _square_ring(size=15):
    image = np.zeros((size, size, size), dtype=bool)
    z = size // 2
    lo, hi = 3, size - 4
    image[lo : hi + 1, lo, z] = True
    image[lo : hi + 1, hi, z] = True
    image[lo, lo : hi + 1, z] = True
    image[hi, lo : hi + 1, z] = True
    return image


def test_topology_aware_collapses_voxel_junction_blob():
    graph = topology_aware_skeleton_image_to_graph(
        _trivalent_t_skeleton(),
        junction_hops=2,
        max_junction_degree=3,
    )
    assert isinstance(graph, nx.MultiGraph)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert sorted(dict(graph.degree()).values()) == [1, 1, 1, 3]
    ensure_embedding(graph, copy=False, normalize=False)


def test_topology_aware_edge_endpoints_match_collapsed_node_positions():
    graph = topology_aware_skeleton_image_to_graph(
        _trivalent_t_skeleton(),
        junction_hops=2,
        max_junction_degree=3,
    )
    for u, v, data in graph.edges(data=True):
        points = np.asarray(data["pts"], dtype=float)
        u_pos = np.asarray(graph.nodes[u]["pos"], dtype=float)
        v_pos = np.asarray(graph.nodes[v]["pos"], dtype=float)
        direct = np.array_equal(points[0], u_pos) and np.array_equal(points[-1], v_pos)
        reverse = np.array_equal(points[0], v_pos) and np.array_equal(points[-1], u_pos)
        assert direct or reverse


def test_topology_aware_pure_ring_is_closed_self_loop():
    graph = topology_aware_skeleton_image_to_graph(_square_ring(), junction_hops=1)
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 1
    u, v, data = next(iter(graph.edges(data=True)))
    assert u == v
    points = np.asarray(data["pts"], dtype=float)
    node_pos = np.asarray(graph.nodes[u]["pos"], dtype=float)
    assert np.array_equal(points[0], node_pos)
    assert np.array_equal(points[-1], node_pos)
    ensure_embedding(graph, copy=False, normalize=False)


def test_topology_aware_rejects_non_3d_input():
    with pytest.raises(ValueError, match="three-dimensional"):
        topology_aware_skeleton_image_to_graph(np.zeros((8, 8), dtype=bool))


def test_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        skeleton_image_to_graph(np.zeros((8, 8, 8), dtype=bool), backend="unknown")
