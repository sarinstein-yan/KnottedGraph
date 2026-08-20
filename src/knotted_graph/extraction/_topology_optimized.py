"""High-performance topology-aware 3-D skeleton extraction.

This module contains two improvements over the first sparse prototype:

1. foreground discovery is restricted to the occupied axis-aligned bounding box,
   while returned coordinates remain in the original global voxel frame;
2. valence-constrained junction repair is accepted only after topology persists
   across adjacent graph-distance scales unless one-hop expansion is certified
   unable to merge any surviving node zones.

The second rule fixes clean-looking split-junction failures that cannot be
identified from degree bounds alone (for example, Pappus and truncated
-tetrahedral test skeletons).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from . import skeleton as _legacy_sparse
from ._sparse_compat import trace_zero_radius_compatible

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def sparse_voxel_adjacency_cropped(
    image: np.ndarray,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build 26-neighbour adjacency after cropping empty image margins.

    Cropping changes only the index-search domain. ``coords`` are shifted back
    into the original image frame, so extracted embedded graphs are byte-for-byte
    compatible with the uncropped sparse parser.
    """
    occupied = [
        np.flatnonzero(image.any(axis=(1, 2))),
        np.flatnonzero(image.any(axis=(0, 2))),
        np.flatnonzero(image.any(axis=(0, 1))),
    ]
    if any(len(indices) == 0 for indices in occupied):
        return np.empty((0, 3), dtype=np.intp), []

    starts = np.asarray([int(indices[0]) for indices in occupied], dtype=np.intp)
    stops = np.asarray([int(indices[-1]) + 1 for indices in occupied], dtype=np.intp)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))
    crop = image[slices]

    flat = np.flatnonzero(crop)
    shape = crop.shape
    local_coords = np.column_stack(np.unravel_index(flat, shape)).astype(
        np.intp,
        copy=False,
    )
    coords = local_coords + starts
    strides = np.asarray((shape[1] * shape[2], shape[2], 1), dtype=np.int64)
    adjacency: list[list[int]] = [[] for _ in range(flat.size)]

    # Positive offsets only: every undirected pair is emitted exactly once.
    # Offsets are already lexicographic, so appending the forward and reverse
    # entries preserves the historical 3x3x3 neighbour-order semantics without
    # a per-voxel Python sort.
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        if (dx, dy, dz) <= (0, 0, 0):
            continue

        query = flat + dx * strides[0] + dy * strides[1] + dz
        positions = np.searchsorted(flat, query)
        valid = positions < flat.size
        left = np.flatnonzero(valid)
        right = positions[valid]
        exact = flat[right] == query[valid]
        left = left[exact]
        right = right[exact]
        if left.size == 0:
            continue

        wanted = np.asarray((dx, dy, dz), dtype=np.intp)
        actual = local_coords[right] - local_coords[left]
        keep = np.all(actual == wanted, axis=1)
        for u, v in zip(left[keep].tolist(), right[keep].tolist()):
            adjacency[u].append(v)
            adjacency[v].append(u)

    return coords, adjacency


def _prepared_components(
    coords: np.ndarray,
    adjacency: list[list[int]],
) -> list[tuple[np.ndarray, list[list[int]]]]:
    prepared: list[tuple[np.ndarray, list[list[int]]]] = []
    for indices in _legacy_sparse._adjacency_components(adjacency):
        remap = {old: index for index, old in enumerate(indices)}
        local_coords = coords[np.asarray(indices, dtype=np.intp)]
        local_adjacency = [
            [remap[v] for v in adjacency[old] if v in remap]
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

    # If every surviving inter-node raw edge contains at least three internal
    # edge voxels (>=5 points including node endpoints), expanding each node zone
    # by one hop cannot make two distinct surviving node zones touch. The clean
    # topology is therefore certified stable for the next scale and a second
    # trace is unnecessary.
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
    """Fail closed unless a valence-valid topology is stable across scales."""
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

    # No persistent valence-valid correction was demonstrated. Preserve the
    # zero-radius topology rather than forcing a topology-changing guess.
    return base[0]


def extract(
    image: np.ndarray,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Production optimized extraction entry point."""
    image = np.asarray(image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")
    if adaptive_max_hops < 0:
        raise ValueError("adaptive_max_hops must be non-negative")
    if max_junction_degree is not None and max_junction_degree < 1:
        raise ValueError("max_junction_degree must be positive")

    coords, adjacency = sparse_voxel_adjacency_cropped(image)
    if max_junction_degree is None:
        return trace_zero_radius_compatible(coords, adjacency)

    return constrained_persistent_extract(
        coords,
        adjacency,
        max_degree=max_junction_degree,
        max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
