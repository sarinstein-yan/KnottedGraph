"""Persistence-based topology selection for sparse 3-D skeleton graphs.

A digital junction can occupy several voxels and thereby create a locally
fragmented graph. The selector traces the same sparse skeleton at adjacent
junction-zone scales and prefers graph structures that persist. Persistence is
available with or without a valence prior: a supplied maximum degree is treated
as an additional physical constraint, not as a prerequisite for multi-scale
selection. Candidate geometry is validated before selection so a topologically
plausible trace cannot later fail because an embedded edge has collapsed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ._tracing import (
    adjacency_components,
    reduced_weighted,
    remove_leaves_for_diagnostic,
    trace_component,
)


def _prepared_components(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> list[tuple[np.ndarray, list[list[int]]]]:
    """Prepare connected components once for repeated multi-scale traces."""
    prepared: list[tuple[np.ndarray, list[list[int]]]] = []
    for indices in adjacency_components(adjacency):
        remap = {old: index for index, old in enumerate(indices)}
        local_coords = coords[np.asarray(indices, dtype=np.intp)]
        local_adjacency = [[remap[v] for v in adjacency[old]] for old in indices]
        prepared.append((local_coords, local_adjacency))
    return prepared


def _trace_prepared(
    prepared: list[tuple[np.ndarray, list[list[int]]]],
    hops: int,
) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    next_id = 0
    for local_coords, local_adjacency in prepared:
        local = trace_component(
            local_coords,
            local_adjacency,
            junction_hops=hops,
        )
        mapping = {node: node + next_id for node in local.nodes()}
        local = nx.relabel_nodes(local, mapping, copy=True)
        graph = nx.compose(graph, local)
        next_id += local.number_of_nodes()
    graph.remove_nodes_from([node for node, degree in graph.degree() if degree == 0])
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


def _anomaly_from_reduced(reduced: nx.MultiGraph, ratio: float) -> int:
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


def _drop_consecutive_duplicates(
    points: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return a polyline with adjacent duplicate samples removed."""
    if len(points) == 0:
        return points
    keep = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep[-1]])) > tolerance:
            keep.append(index)
    return points[np.asarray(keep, dtype=np.intp)]


def _embedded_geometry_safe(graph: nx.MultiGraph) -> bool:
    """Reject traces whose edge geometry would collapse during normalization.

    In particular, a self-loop must contain a genuine excursion away from its
    node before returning to that node. Merely having a self-loop in the
    abstract MultiGraph is not enough: a two-endpoint loop whose endpoints snap
    to the same vertex is not an embedded edge and later normalization correctly
    rejects it. Detecting that here lets the multi-scale selector try another
    junction radius instead of returning a doomed candidate.
    """
    positions: dict[object, np.ndarray] = {}
    for node, data in graph.nodes(data=True):
        pos = np.asarray(data.get("pos"), dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            return False
        positions[node] = pos

    for u, v, data in graph.edges(data=True):
        points = np.asarray(data.get("pts"), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            return False
        if not np.isfinite(points).all():
            return False

        start = positions[u]
        end = positions[v]
        normalized = points.copy()
        if u == v:
            normalized[0] = start
            normalized[-1] = start
        else:
            forward = float(
                np.linalg.norm(points[0] - start)
                + np.linalg.norm(points[-1] - end)
            )
            reverse = float(
                np.linalg.norm(points[0] - end)
                + np.linalg.norm(points[-1] - start)
            )
            if reverse < forward:
                normalized = normalized[::-1].copy()
            normalized[0] = start
            normalized[-1] = end

        normalized = _drop_consecutive_duplicates(normalized)
        if len(normalized) < 2:
            return False
        if u == v:
            if len(normalized) < 3:
                return False
            if not np.any(
                np.linalg.norm(normalized[1:-1] - start, axis=1) > 1e-10
            ):
                return False
    return True


def _diagnostic_summary(
    graph: nx.MultiGraph,
    *,
    max_degree: int | None,
    anomaly_ratio: float,
) -> tuple[nx.MultiGraph, bool, tuple, bool]:
    """Return comparison topology, cleanliness, fingerprint and local safety.

    With a valence prior, the historical leaf-pruned diagnostic core is retained
    so constrained benchmark behaviour is not weakened by terminal voxel
    artefacts. Without a valence prior, leaves may be scientifically meaningful,
    so persistence compares the full graph after degree-two chain reduction.
    """
    if max_degree is None:
        diagnostic_graph = nx.MultiGraph(graph)
    else:
        diagnostic_graph = remove_leaves_for_diagnostic(graph)

    reduced = reduced_weighted(diagnostic_graph)
    max_observed_degree = max(
        (degree for _, degree in reduced.degree()),
        default=0,
    )
    anomaly_count = _anomaly_from_reduced(reduced, anomaly_ratio)
    valence_ok = max_degree is None or max_observed_degree <= max_degree
    clean = (
        graph.number_of_nodes() > 0
        and valence_ok
        and anomaly_count == 0
        and _embedded_geometry_safe(graph)
    )

    if max_degree is None:
        # Without a valence prior, a physically real pair of nearby junctions
        # must not be merged merely because its connecting polyline contains
        # fewer than a fixed number of voxel samples. The scale-relative
        # junction anomaly test above remains active; persistence itself decides
        # whether the nearby-junction topology is stable.
        one_hop_safe = True
    else:
        # Preserve the constrained selector's existing short-edge diagnostic.
        survivors = set(diagnostic_graph.nodes())
        has_mergeable_short_edge = any(
            u != v
            and u in survivors
            and v in survivors
            and len(data.get("pts", ())) < 5
            for u, v, data in graph.edges(data=True)
        )
        one_hop_safe = not has_mergeable_short_edge

    return reduced, clean, _core_fingerprint(reduced), one_hop_safe


def _best_clean_candidate(
    candidates,
    *,
    min_occurrences: int = 2,
):
    """Choose the most persistent isomorphism class, then the smallest scale.

    The coarse fingerprint is only a cheap prefilter. Degree sequences do not
    determine graph topology (for example, K3,3 and the triangular prism have
    the same node/edge counts and degree sequence), so recurrence is counted
    only after an actual multigraph-isomorphism check.

    A class must occur at least ``min_occurrences`` times. A singleton candidate
    is not persistence evidence and therefore cannot displace the zero-radius
    fallback by itself.
    """
    if min_occurrences < 1:
        raise ValueError("min_occurrences must be positive")
    if not candidates:
        return None

    classes: list[list[tuple[int, tuple]]] = []
    for candidate in candidates:
        _hops, summary = candidate
        for group in classes:
            representative = group[0][1]
            if (
                representative[3] == summary[3]
                and nx.is_isomorphic(representative[1], summary[1])
            ):
                group.append(candidate)
                break
        else:
            classes.append([candidate])

    eligible = [
        group for group in classes
        if len(group) >= min_occurrences
    ]
    if not eligible:
        return None

    best_group = min(
        eligible,
        key=lambda group: (-len(group), min(item[0] for item in group)),
    )
    return min(best_group, key=lambda item: item[0])[1]


def persistent_extract(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    max_degree: int | None = None,
    max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Return the smallest well-supported multi-scale skeleton reconstruction.

    Every scale must carry nondegenerate embedded edge geometry and pass local
    digital-junction anomaly checks. When ``max_degree`` is supplied, satisfying
    that valence ceiling is an additional validity condition. Otherwise no
    valence is inferred.

    Consecutive valid isomorphic scales provide the strongest persistence signal
    and the smaller-scale representative is returned. If no consecutive pair is
    available, the most recurrent valid isomorphism class is selected, again
    preferring its smallest scale. A topology observed only once is not treated
    as persistent; if no recurring valid class exists, the direct zero-radius
    graph is returned as the conservative fallback.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")
    if max_degree is not None and max_degree < 1:
        raise ValueError("max_degree must be positive when supplied")

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
    clean_candidates = []
    if base[2] and base[4]:
        clean_candidates.append((0, base))

    previous = base
    for hops in range(1, max_hops + 1):
        current = build(hops)
        if current[2] and current[4]:
            clean_candidates.append((hops, current))
        if (
            previous[2]
            and previous[4]
            and current[2]
            and current[4]
            and previous[3] == current[3]
            and nx.is_isomorphic(previous[1], current[1])
        ):
            return previous[0]
        previous = current

    fallback = _best_clean_candidate(
        clean_candidates,
        min_occurrences=2,
    )
    if fallback is not None:
        return fallback[0]
    return base[0]


def constrained_persistent_extract(
    coords: np.ndarray,
    adjacency: list[list[int]],
    *,
    max_degree: int,
    max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Compatibility wrapper for persistence with a required valence prior."""
    return persistent_extract(
        coords,
        adjacency,
        max_degree=max_degree,
        max_hops=max_hops,
        anomaly_ratio=anomaly_ratio,
    )
