"""Geometry-preserving graph operations for protein perturbations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import networkx as nx
import numpy as np

from knotted_graph.core.embedding import (
    drop_consecutive_duplicates,
    ensure_embedding,
    oriented_edge_polyline,
)


@dataclass(frozen=True, order=True)
class CrosslinkEdgeRef:
    crosslink_id: str
    u: Any
    v: Any
    key: Any
    crosslink_type: str
    endpoint_a: str
    endpoint_b: str


def _node_label(node: Any) -> str:
    label = getattr(node, "label", None)
    if label is not None:
        return str(label)
    return repr(node)


def crosslink_edges(graph: nx.MultiGraph) -> list[CrosslinkEdgeRef]:
    """Return stable references to all physical crosslink edges."""

    refs: list[CrosslinkEdgeRef] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("edge_kind") != "crosslink":
            continue
        crosslink_id = str(data.get("crosslink_id", key))
        endpoint_a = data.get("endpoint_a", {})
        endpoint_b = data.get("endpoint_b", {})
        refs.append(
            CrosslinkEdgeRef(
                crosslink_id=crosslink_id,
                u=u,
                v=v,
                key=key,
                crosslink_type=str(data.get("crosslink_type", "other")),
                endpoint_a=str(
                    endpoint_a.get("residue", {}).get("chain_id", _node_label(u))
                    + ":"
                    + endpoint_a.get("residue", {}).get("sequence_id", "")
                    if isinstance(endpoint_a, dict)
                    else _node_label(u)
                ),
                endpoint_b=str(
                    endpoint_b.get("residue", {}).get("chain_id", _node_label(v))
                    + ":"
                    + endpoint_b.get("residue", {}).get("sequence_id", "")
                    if isinstance(endpoint_b, dict)
                    else _node_label(v)
                ),
            )
        )
    refs.sort(key=lambda ref: ref.crosslink_id)
    ids = [ref.crosslink_id for ref in refs]
    if len(ids) != len(set(ids)):
        raise ValueError("crosslink_id values must be unique within a protein graph")
    return refs


def _merged_residue_keys(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
) -> list[dict[str, Any]]:
    keys_a = list(data_a.get("residue_keys", []))
    keys_b = list(data_b.get("residue_keys", []))
    if keys_a and keys_b and keys_a[-1] == keys_b[0]:
        return keys_a + keys_b[1:]
    return keys_a + keys_b


def suppress_backbone_degree_two_nodes(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Suppress loopless degree-two backbone nodes while preserving polylines."""

    result = ensure_embedding(graph, copy=True, normalize=True)
    counter = 0
    changed = True
    while changed:
        changed = False
        for node in list(result.nodes):
            if result.degree(node) != 2:
                continue
            incident = list(result.edges(node, keys=True, data=True))
            if len(incident) != 2 or any(u == v for u, v, _, _ in incident):
                continue
            if any(data.get("edge_kind") != "backbone" for *_, data in incident):
                continue

            oriented: list[tuple[Any, np.ndarray, dict[str, Any]]] = []
            for u, v, key, data in incident:
                neighbor = v if u == node else u
                points = oriented_edge_polyline(result, neighbor, node, key, data)
                oriented.append((neighbor, points, data))
            (neighbor_a, points_a, data_a), (neighbor_b, points_b, data_b) = oriented
            merged_points = drop_consecutive_duplicates(
                np.vstack([points_a, points_b[::-1][1:]])
            )
            if len(merged_points) < 2:
                continue
            chain_a = data_a.get("chain_id")
            chain_b = data_b.get("chain_id")
            chain_id = chain_a if chain_a == chain_b else None
            result.remove_node(node)
            result.add_edge(
                neighbor_a,
                neighbor_b,
                key=f"backbone:suppressed:{counter}",
                pts=merged_points,
                edge_kind="backbone",
                chain_id=chain_id,
                residue_keys=_merged_residue_keys(data_a, data_b),
                suppressed_node_count=(
                    int(data_a.get("suppressed_node_count", 0))
                    + int(data_b.get("suppressed_node_count", 0))
                    + 1
                ),
            )
            counter += 1
            changed = True
            break
    return result


def remove_crosslinks(
    graph: nx.MultiGraph,
    crosslink_ids: Iterable[str],
    *,
    suppress_degree_two: bool = True,
) -> nx.MultiGraph:
    """Delete selected crosslink edges without deleting backbone geometry."""

    selected = frozenset(str(value) for value in crosslink_ids)
    available = {ref.crosslink_id for ref in crosslink_edges(graph)}
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"Unknown crosslink IDs: {unknown}")
    result = ensure_embedding(graph, copy=True, normalize=True)
    for u, v, key, data in list(result.edges(keys=True, data=True)):
        if (
            data.get("edge_kind") == "crosslink"
            and str(data.get("crosslink_id", key)) in selected
        ):
            result.remove_edge(u, v, key)
    for node, data in list(result.nodes(data=True)):
        if result.degree(node) == 0 and data.get("node_type") == "metal_center":
            result.remove_node(node)
    if suppress_degree_two:
        result = suppress_backbone_degree_two_nodes(result)
    result.graph["removed_crosslink_ids"] = tuple(sorted(selected))
    result.graph["crosslink_count"] = len(crosslink_edges(result))
    result.graph["crosslink_ids"] = tuple(
        ref.crosslink_id for ref in crosslink_edges(result)
    )
    return result


def extract_crosslink_core(
    graph: nx.MultiGraph,
    *,
    remove_bridges: bool = True,
) -> nx.MultiGraph:
    """Return the bridgeless cyclic core supported by physical crosslinks.

    Open protein termini are bridge tails.  Because a single bridge makes the
    Yamada polynomial vanish, the workflow first extracts the embedded 2-core
    and then removes any remaining single graph bridges between cyclic blocks.
    The same reduction is recomputed after every crosslink deletion.
    """

    result = ensure_embedding(graph, copy=True, normalize=True)
    original_node_count = result.number_of_nodes()
    original_edge_count = result.number_of_edges()
    original_crosslink_ids = tuple(ref.crosslink_id for ref in crosslink_edges(result))
    while result.number_of_nodes():
        leaves = [node for node in result.nodes if result.degree(node) <= 1]
        if not leaves:
            break
        result.remove_nodes_from(leaves)
    removed_bridge_crosslink_ids: list[str] = []
    if remove_bridges and result.number_of_edges():
        simple = nx.Graph(result)
        for u, v in list(nx.bridges(simple)):
            if result.number_of_edges(u, v) != 1:
                continue
            key, data = next(iter(result[u][v].items()))
            if data.get("edge_kind") == "crosslink":
                removed_bridge_crosslink_ids.append(str(data.get("crosslink_id", key)))
            result.remove_edge(u, v, key)
        result.remove_nodes_from(
            [node for node in list(result.nodes) if result.degree(node) == 0]
        )
    if result.number_of_edges():
        result = suppress_backbone_degree_two_nodes(result)
    else:
        result.remove_nodes_from(list(result.nodes))
    retained_crosslink_ids = tuple(ref.crosslink_id for ref in crosslink_edges(result))
    result.graph.update(
        core_kind="bridgeless_crosslink_supported_core",
        core_remove_bridges=remove_bridges,
        core_removed_bridge_crosslink_ids=tuple(sorted(removed_bridge_crosslink_ids)),
        pre_core_node_count=original_node_count,
        pre_core_edge_count=original_edge_count,
        pre_core_crosslink_ids=original_crosslink_ids,
        core_crosslink_ids=retained_crosslink_ids,
        core_excluded_crosslink_ids=tuple(
            sorted(set(original_crosslink_ids) - set(retained_crosslink_ids))
        ),
    )
    return result


def embedding_hash(graph: nx.MultiGraph, *, decimals: int = 10) -> str:
    """Return an iteration-order-independent hash of topology and geometry."""

    if graph.number_of_edges() == 0:
        return hashlib.sha256(b"empty_spatial_graph").hexdigest()
    normalized = ensure_embedding(graph, copy=True, normalize=True)
    node_labels = {node: _node_label(node) for node in normalized.nodes}
    nodes = sorted(
        (
            node_labels[node],
            np.round(np.asarray(data["pos"], dtype=float), decimals).tolist(),
            str(data.get("node_type", "")),
        )
        for node, data in normalized.nodes(data=True)
    )
    edges = []
    for u, v, key, data in normalized.edges(keys=True, data=True):
        label_u = node_labels[u]
        label_v = node_labels[v]
        points = oriented_edge_polyline(normalized, u, v, key, data)
        if label_v < label_u or (
            label_u == label_v
            and tuple(points[-1].tolist()) < tuple(points[0].tolist())
        ):
            label_u, label_v = label_v, label_u
            points = points[::-1]
        edges.append(
            (
                label_u,
                label_v,
                str(key),
                str(data.get("edge_kind", "")),
                str(data.get("crosslink_id", "")),
                np.round(points, decimals).tolist(),
            )
        )
    payload = json.dumps(
        {"nodes": nodes, "edges": sorted(edges, key=lambda item: item[:5])},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _expanded_abstract_graph(graph: nx.MultiGraph) -> nx.Graph:
    """Encode labeled multiedges and loops in a simple graph."""

    expanded = nx.Graph()
    original_nodes: dict[Any, tuple[str, int]] = {}
    for index, (node, data) in enumerate(graph.nodes(data=True)):
        expanded_node = ("node", index)
        original_nodes[node] = expanded_node
        node_type = str(data.get("node_type", "residue"))
        expanded.add_node(expanded_node, label=f"node:{node_type}")
    for edge_index, (u, v, data) in enumerate(graph.edges(data=True)):
        edge_node = ("edge", edge_index)
        edge_kind = str(data.get("edge_kind", ""))
        crosslink_type = str(data.get("crosslink_type", ""))
        expanded.add_node(edge_node, label=f"edge:{edge_kind}:{crosslink_type}")
        expanded.add_edge(edge_node, original_nodes[u])
        expanded.add_edge(edge_node, original_nodes[v])
        if u == v:
            port = ("loop_port", edge_index)
            expanded.add_node(port, label="loop_port")
            expanded.add_edge(edge_node, port)
            expanded.add_edge(port, original_nodes[u])
    return expanded


def abstract_connectivity_hash(graph: nx.MultiGraph) -> str:
    """Hash abstract multigraph connectivity while ignoring coordinates.

    This Weisfeiler--Lehman hash is a fast candidate filter.  Hash equality is
    not itself an exact isomorphism certificate; use
    :func:`abstract_connectivity_isomorphic` for verification.
    """

    expanded = _expanded_abstract_graph(graph)
    return nx.weisfeiler_lehman_graph_hash(expanded, node_attr="label", iterations=5)


def abstract_connectivity_certificate(graph: nx.MultiGraph) -> dict[str, Any]:
    """Serialize the labeled expansion used for exact isomorphism checks."""

    expanded = _expanded_abstract_graph(graph)
    node_ids = {node: index for index, node in enumerate(expanded.nodes)}
    return {
        "encoding": "labeled_edge_node_expansion_v1",
        "nodes": [
            {"id": node_ids[node], "label": str(data.get("label", ""))}
            for node, data in expanded.nodes(data=True)
        ],
        "edges": sorted(
            [node_ids[first], node_ids[second]]
            for first, second in expanded.edges
        ),
    }


def _certificate_graph(certificate: dict[str, Any]) -> nx.Graph:
    if certificate.get("encoding") != "labeled_edge_node_expansion_v1":
        raise ValueError("Unsupported abstract-connectivity certificate encoding")
    graph = nx.Graph()
    for node in certificate.get("nodes", []):
        graph.add_node(int(node["id"]), label=str(node.get("label", "")))
    for first, second in certificate.get("edges", []):
        graph.add_edge(int(first), int(second))
    return graph


def abstract_connectivity_isomorphic(
    first: nx.MultiGraph | dict[str, Any],
    second: nx.MultiGraph | dict[str, Any],
) -> bool:
    """Verify exact labeled abstract isomorphism, including loops/multiedges."""

    first_expanded = (
        _certificate_graph(first)
        if isinstance(first, dict)
        else _expanded_abstract_graph(first)
    )
    second_expanded = (
        _certificate_graph(second)
        if isinstance(second, dict)
        else _expanded_abstract_graph(second)
    )
    node_match = nx.algorithms.isomorphism.categorical_node_match("label", "")
    return nx.is_isomorphic(first_expanded, second_expanded, node_match=node_match)
