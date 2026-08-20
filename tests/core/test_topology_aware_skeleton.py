import networkx as nx
import numpy as np
import pytest
from skimage import morphology

from knotted_graph.core import ensure_embedding
from knotted_graph.extraction import (
    skeleton_image_to_graph,
    skeletonize_volume,
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
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    z = c
    radius = 5
    vertices = [(c - radius, c), (c, c + radius), (c + radius, c), (c, c - radius)]
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
        vertices = [(c - radius, c), (c, c + radius), (c + radius, c), (c, c - radius)]
        for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
            steps = max(abs(x1 - x0), abs(y1 - y0))
            for step in range(steps):
                alpha = step / steps
                x = int(round(x0 + alpha * (x1 - x0)))
                y = int(round(y0 + alpha * (y1 - y0)))
                image[x, y, z] = True
    return image


def test_canonical_extractor_collapses_digital_junction_with_valence_hint():
    graph = skeleton_image_to_graph(_trivalent_t_skeleton(), max_junction_degree=3)
    assert isinstance(graph, nx.MultiGraph)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert sorted(dict(graph.degree()).values()) == [1, 1, 1, 3]
    ensure_embedding(graph, copy=False, normalize=False)


def test_alias_and_canonical_extractor_are_identical():
    image = _trivalent_t_skeleton()
    expected = skeleton_image_to_graph(image, max_junction_degree=3)
    actual = topology_aware_skeleton_image_to_graph(image, max_junction_degree=3)
    assert nx.is_isomorphic(expected, actual)
    for u, v, data in actual.edges(data=True):
        points = np.asarray(data["pts"], dtype=float)
        u_pos = np.asarray(actual.nodes[u]["pos"], dtype=float)
        v_pos = np.asarray(actual.nodes[v]["pos"], dtype=float)
        assert (
            np.array_equal(points[0], u_pos) and np.array_equal(points[-1], v_pos)
        ) or (
            np.array_equal(points[0], v_pos) and np.array_equal(points[-1], u_pos)
        )


def test_cropped_extraction_preserves_global_coordinates():
    base = _diamond_ring()
    padded = np.pad(base, ((11, 23), (7, 19), (5, 17)), mode="constant")
    graph = skeleton_image_to_graph(padded)
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 1
    node = next(iter(graph.nodes()))
    expected = np.asarray(next(iter(skeleton_image_to_graph(base).nodes(data=True)))[1]["pos"])
    actual = np.asarray(graph.nodes[node]["pos"])
    assert np.array_equal(actual, expected + np.array([11, 7, 5]))


def test_package_and_module_share_same_public_functions():
    import knotted_graph.extraction as package_api
    import knotted_graph.extraction.skeleton as module_api

    assert package_api.skeleton_image_to_graph is module_api.skeleton_image_to_graph
    assert package_api.skeletonize_volume is module_api.skeletonize_volume
    assert (
        package_api.topology_aware_skeleton_image_to_graph
        is module_api.topology_aware_skeleton_image_to_graph
    )


def test_public_extractor_routes_to_optimizer(monkeypatch):
    import knotted_graph.extraction.skeleton as module_api

    called = {}

    def fake_extract(image, **kwargs):
        called["shape"] = np.asarray(image).shape
        called["kwargs"] = kwargs
        return nx.MultiGraph()

    monkeypatch.setattr(module_api, "_optimized_extract", fake_extract)
    result = module_api.skeleton_image_to_graph(
        np.zeros((9, 10, 11), dtype=bool),
        max_junction_degree=3,
        adaptive_max_hops=2,
        anomaly_ratio=0.12,
    )
    assert isinstance(result, nx.MultiGraph)
    assert called == {
        "shape": (9, 10, 11),
        "kwargs": {
            "max_junction_degree": 3,
            "adaptive_max_hops": 2,
            "anomaly_ratio": 0.12,
        },
    }


def test_pure_ring_is_closed_self_loop():
    graph = skeleton_image_to_graph(_diamond_ring())
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 1
    u, v, data = next(iter(graph.edges(data=True)))
    assert u == v
    points = np.asarray(data["pts"], dtype=float)
    node_pos = np.asarray(graph.nodes[u]["pos"], dtype=float)
    assert np.array_equal(points[0], node_pos)
    assert np.array_equal(points[-1], node_pos)
    ensure_embedding(graph, copy=False, normalize=False)


def test_disconnected_ring_components_are_preserved():
    graph = skeleton_image_to_graph(_two_diamond_rings())
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 2
    assert nx.number_connected_components(graph) == 2
    assert all(u == v for u, v in graph.edges())


def test_skeletonize_volume_matches_full_lee_and_preserves_global_coordinates():
    mask = np.zeros((64, 61, 59), dtype=bool)
    mask[37:53, 31:47, 28:44] = True
    mask[42:48, 26:52, 33:39] = True
    expected = morphology.skeletonize(mask, method="lee")
    actual = skeletonize_volume(mask)
    assert np.array_equal(actual, expected)


def test_skeletonize_volume_rejects_empty_or_non_3d_input():
    with pytest.raises(ValueError, match="three-dimensional"):
        skeletonize_volume(np.zeros((8, 8), dtype=bool))
    with pytest.raises(ValueError, match="does not contain"):
        skeletonize_volume(np.zeros((8, 8, 8), dtype=bool))


def test_extractor_rejects_non_3d_input():
    with pytest.raises(ValueError, match="three-dimensional"):
        skeleton_image_to_graph(np.zeros((8, 8), dtype=bool))
