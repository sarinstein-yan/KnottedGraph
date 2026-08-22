import networkx as nx

from knotted_graph.extraction._topology_optimized import (
    _best_clean_candidate,
    _core_fingerprint,
)


def _summary(graph: nx.Graph):
    reduced = nx.MultiGraph(graph)
    return reduced, True, _core_fingerprint(reduced), True


def test_fallback_persistence_uses_isomorphism_not_degree_fingerprint():
    # K3,3 and the triangular prism both have 6 vertices, 9 edges, and degree
    # sequence (3,3,3,3,3,3), but they are not isomorphic.  A fingerprint-only
    # vote would incorrectly merge all four candidates into one topology class.
    k33 = nx.complete_bipartite_graph(3, 3)
    prism = nx.circular_ladder_graph(3)
    assert _core_fingerprint(nx.MultiGraph(k33)) == _core_fingerprint(nx.MultiGraph(prism))
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
