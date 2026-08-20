"""Persistence-based topology repair for sparse 3-D skeleton graphs.

The first sparse optimizer could return a clean-looking degree-valid graph even
when a voxelized junction was still split into several graph vertices. The
selector here therefore requires topology persistence across adjacent
junction-zone scales whenever one-hop expansion could still merge surviving
node zones. If no persistent valence-valid correction is demonstrated, it fails
closed to the zero-radius graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from . import _legacy_skeleton as _legacy_sparse


def _prepared_components(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> list[tuple[np.ndarray, list[list[int]]]]:
    """Prepare connected components once for repeated multi-scale traces."""
    prepared: list[tuple[np.ndarray, list[list[int]]]] = []
    for indices in _legacy_sparse._adjacency_components(adjacency):
        remap = {old: index for index, old in enumerate(indices)}
        local_coords = coords[np.asarray(indices, dtype=np.intp)]
        local_adjacency = [
            [remap[v] for v in adjacency[old]]
            for old in indices
        ]
        prepared.append((local_coords, local_adjacency))
    return prepared


def _trace_prepared(
    prepared: list[tuple[np.ndarray, list[list[int]]]],
    hops: int,
) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    next_id = 0
    for local_coords, local_adjacency in prepared:
        local = _legacy_sparse._trace_component(
            local_coords,
            local_adjacency,
            junction_hops=hops,
        )
        mapping = {node: node + next_id for node in local.nodes()}
        local = nx.relabel_nodes(local, mapping, copy=True)
        graph = nx.compose(graph, local)
        next_id += local.number_of_nodes()

    graph.remove_nodes_from(
        [node for node, degree in graph.degree() if degree == 0]
    )
    return graph


def _core_fingerprint(graph: nx.MultiGraph) -> tuple:
    components = []
    for nodes in nx.connected_components(graph):
        part = graph.subgraph(nodes)
        components.append(
            (
                part.number_of_nodes(),
                part.number_of_edges(),
                tuple(sorted(int(degree) for _, degree in part.degree())),
            )
        )
    return tuple(sorted(components))


def _anomaly_from_reduced(
    reduced: nx.MultiGraph,
    ratio: float,
) -> int:
    count = 0
    pairs: set[tuple[object, object]] = set()
    for u, v in reduced.edges():
        if u != v and reduced.degree(u) >= 3 and reduced.degree(v) >= 3:
            pairs.add(tuple(sorted((u, v), key=repr)))

    for u, v in pairs:
        pair_lengths = [
            float(data.get("weight", 0.0))
            for data in reduced.get_edge_data(u, v, default={}).values()
        ]
        refs_u: list[float] = []
        refs_v: list[float] = []

        for a, b, _, data in reduced.edges(u, keys=True, data=True):
            other = b if a == u else a
            if other not in (u, v):
                refs_u.append(float(data.get("weight", 0.0)))
        for a, b, _, data in reduced.edges(v, keys=True, data=True):
            other = b if a == v else a
            if other not in (u, v):
                refs_v.append(float(data.get("weight", 0.0)))

        if not refs_u or not refs_v:
            continue
        scale = min(float(np.median(refs_u)), float(np.median(refs_v)))
        if scale <= 0:
            continue
        if min(pair_lengths) / scale < ratio:
            count += 1

    return count


def _diagnostic_summary(
    graph: nx.MultiGraph,
    *,
    max_degree: int,
    anomaly_ratio: float,
) -> tuple[nx.MultiGraph, bool, tuple, bool]:
    """Return reduced topology, cleanliness, fingerprint and one-hop safety."""
    pruned = _legacy_sparse._remove_leaves_for_diagnostic(graph)
    reduced = _legacy_sparse._reduced_weighted(pruned)
    max_observed_degree = max(
        (degree for _, degree in reduced.degree()),
        default=0,
    )
    anomaly_count = _anomaly_from_reduced(reduced, anomaly_ratio)
    clean = max_observed_degree <= max_degree and anomaly_count == 0

    survivors = set(pruned.nodes())
    has_mergeable_short_edge = any(
        u != v
        and u in survivors
        and v in survivors
        and len(data.get("pts", ())) < 5
        for u, v, data in graph.edges(data=True)
    )
    one_hop_safe = not has_mergeable_short_edge

    return reduced, clean, _core_fingerprint(reduced), one_hop_safe


def constrained_persistent_extract(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    max_degree: int,
    max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Return a persistent valence-valid correction or fail closed to base."""
    prepared = _prepared_components(coords, adjacency)

    def build(hops: int):
        graph = _trace_prepared(prepared, hops)
        reduced, clean, fingerprint, one_hop_safe = _diagnostic_summary(
            graph,
            max_degree=max_degree,
            anomaly_ratio=anomaly_ratio,
        )
        return graph, reduced, clean, fingerprint, one_hop_safe

    base = build(0)
    if base[2] and base[4]:
        return base[0]

    previous = base
    for hops in range(1, max_hops + 1):
        current = build(hops)
        if (
            previous[2]
            and current[2]
            and previous[3] == current[3]
            and nx.is_isomorphic(previous[1], current[1])
        ):
            return previous[0]
        previous = current

    return base[0]
