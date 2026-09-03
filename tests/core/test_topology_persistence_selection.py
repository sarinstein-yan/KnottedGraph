import networkx as nx
import numpy as np

from knotted_graph.extraction import skeleton_image_to_graph
from knotted_graph.extraction._topology_optimized import (
    _best_clean_candidate,
    _core_fingerprint,
)


def _summary(graph: nx.Graph):
    full = nx.MultiGraph(graph)
    reduced = nx.MultiGraph(graph)
    return full, reduced, True, _core_fingerprint(reduced), True


def _fragmented_trivalent_t(size: int = 25) -> np.ndarray:
    """Digital T with one local voxel that creates a spurious zero-hop loop."""
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    image[2 : c + 1, c, c] = True
    image[c : size - 2, c, c] = True
    image[c, c : size - 2, c] = True
    image[c + 1, c + 1, c + 1] = True
    return image


def _five_valent_star(size: int = 31) -> np.ndarray:
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    image[3 : size - 3, c, c] = True
    image[c, 3 : size - 3, c] = True
    image[c, c, c : size - 3] = True
    return image


def _two_close_trivalent_junctions(size: int = 29) -> np.ndarray:
    image = np.zeros((size, size, size), dtype=bool)
    c = size // 2
    left = 10
    right = 14
    image[3 : size - 3, c, c] = True
    image[left, 3 : c + 1, c] = True
    image[right, c : size - 3, c] = True
    return image


def _embedded_signature(graph: nx.MultiGraph):
    nodes = tuple(
        (
            node,
            tuple(np.asarray(data["pos"], dtype=float).tolist()),
        )
        for node, data in graph.nodes(data=True)
    )
    edges = tuple(
        (
            u,
            v,
            key,
            tuple(
                tuple(row)
                for row in np.asarray(data["pts"], dtype=float).tolist()
            ),
        )
        for u, v, key, data in graph.edges(keys=True, data=True)
    )
    return nodes, edges


def test_fallback_persistence_uses_isomorphism_not_degree_fingerprint():
    # K3,3 and the triangular prism both have 6 vertices, 9 edges, and degree
    # sequence (3,3,3,3,3,3), but they are not isomorphic. A fingerprint-only
    # vote would incorrectly merge all four candidates into one topology class.
    k33 = nx.complete_bipartite_graph(3, 3)
    prism = nx.circular_ladder_graph(3)
    assert _core_fingerprint(nx.MultiGraph(k33)) == _core_fingerprint(
        nx.MultiGraph(prism)
    )
    assert not nx.is_isomorphic(k33, prism)

    candidates = [
        (1, _summary(prism)),
        (2, _summary(k33)),
        (3, _summary(k33)),
        (4, _summary(k33)),
    ]
    selected = _best_clean_candidate(candidates)
    assert selected is not None
    assert nx.is_isomorphic(selected[1], nx.MultiGraph(k33))
    assert not nx.is_isomorphic(selected[1], nx.MultiGraph(prism))


def test_singleton_candidate_is_not_treated_as_persistence():
    selected = _best_clean_candidate([(1, _summary(nx.path_graph(4)))])
    assert selected is None


def test_unknown_degree_persistence_repairs_fragmented_digital_junction():
    graph = skeleton_image_to_graph(_fragmented_trivalent_t())
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert sorted(dict(graph.degree()).values()) == [1, 1, 1, 3]
    assert not any(u == v for u, v in graph.edges())


def test_unknown_degree_persistence_preserves_genuine_high_valence():
    graph = skeleton_image_to_graph(_five_valent_star())
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 5
    assert sorted(dict(graph.degree()).values()) == [1, 1, 1, 1, 1, 5]


def test_unknown_degree_persistence_does_not_overmerge_close_junctions():
    graph = skeleton_image_to_graph(_two_close_trivalent_junctions())
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 5
    assert sorted(dict(graph.degree()).values()) == [1, 1, 1, 1, 3, 3]


def test_unknown_degree_persistence_is_deterministic():
    image = _fragmented_trivalent_t()
    first = skeleton_image_to_graph(image)
    second = skeleton_image_to_graph(image)
    assert _embedded_signature(first) == _embedded_signature(second)
