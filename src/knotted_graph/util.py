import numpy as np
import sympy as sp
import networkx as nx
from rdp import rdp, pldist

import logging
from typing import List, Tuple, Iterable, Set, Optional, Any, Sequence, TypeVar
from numpy.typing import NDArray, ArrayLike
kSymbols = TypeVar('kSymbols', bound=Tuple[sp.Symbol, sp.Symbol, sp.Symbol])

__all__ = [
    "PT",
    "is_PT_symmetric",
    "is_trivalent",
    "idx_to_coord",
    "total_edge_pts",
    "smooth_edges",
    "remove_leaf_nodes",
    "simplify_edges",
]


# PT-operator
def PT(
    h: sp.Matrix, 
    k_symbols: Optional[kSymbols] = None,
) -> sp.Matrix:
    """Apply the PT-symmetry operation to a 2x2 matrix.

    Parameters
    ----------
    h : sp.Matrix
        The 2x2 matrix to be transformed.
    k_symbols : (sp.Symbol, sp.Symbol, sp.Symbol), optional
        The momentum space coordinates, by default None, which assumes
        only momentum symbols are present in the matrix.

    Returns
    -------
    sp.Matrix
        The transformed 2x2 matrix.
    """    
    if k_symbols is None:
        k_symbols = sorted(h.free_symbols, key=lambda s: s.name)
    sx = sp.Matrix([[0, 1], [1, 0]])
    return sx * sp.conjugate(h).xreplace({k: -k for k in k_symbols}) * sx


def is_PT_symmetric(
        h: sp.Matrix, 
        k_symbols: Optional[kSymbols] = None
    ) -> bool:
    """Check if a 2x2 Operator is PT-symmetric.

    Parameters
    ----------
    h : sp.Matrix
        The 2x2 matrix to be checked.
    k_symbols : (sp.Symbol, sp.Symbol, sp.Symbol), optional
        The momentum space coordinates, by default None, which assumes
        only momentum symbols are present in the matrix.

    Returns
    -------
    bool
        True if the operator is PT-symmetric, False otherwise.
    """
    return sp.simplify(h - PT(h, k_symbols=k_symbols)) == sp.zeros(2, 2)


def is_trivalent(G):
    """
    Check if a graph is trivalent -- if all vertices have degree <= 3.
    
    Parameters:
      G : networkx.MultiGraph / networkx.Graph
         The undirected graph to check.
    
    Returns:
      True if the graph is trivalent, False otherwise.
    """
    degs = nx.degree(G)
    return all(degree <= 3 for node, degree in degs)


def idx_to_coord(
        indices: ArrayLike,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> NDArray:
        """Convert an array of 3D image indices to spatial coordinates."""

        array = np.asarray(indices)
        if array.shape[-1] != 3:
            raise ValueError("Input array must have shape (..., 3).")
        
        return array * spacing + origin


def total_edge_pts(
    G: nx.MultiGraph,
) -> int:
    """Count the total number of points in the graph's edges."""
    return sum(len(G[u][v][key]['pts'])
               for u, v, key in G.edges(keys=True))


def smooth_edges(
    G: nx.MultiGraph,
    epsilon: float = 0.,
    dist: Any = pldist,
    algo: str = "iter",
    return_mask: bool = False,
    copy: bool = True,
) -> (Any | NDArray):
    """Smooth the edge points of a directed graph.

    Parameters
    ----------
    G : nx.MultiGraph
        The input directed graph.
    epsilon : float, optional
        The RDP simplification tolerance, by default 0.
    dist : Any, optional
        The distance function to use, by default pldist.
    algo : str, optional
        The algorithm to use for RDP, by default "iter".
    return_mask : bool, optional
        Whether to return a mask of the simplified points, by default False.
    copy : bool, optional
        Whether to return a copy of the graph, by default True.
        
    Returns
    -------
    nx.MultiGraph
        The graph with smoothed edge points.
    """
    G = G.copy() if copy else G
    for u, v, key, pts in G.edges(keys=True, data='pts'):
        if pts is None or len(pts) < 3:
            continue
        G[u][v][key]['pts'] = rdp(
            pts, epsilon=epsilon, dist=dist, algo=algo, return_mask=return_mask
        )
    return G


def remove_leaf_nodes(
    G: nx.MultiGraph
) -> nx.MultiGraph:
    """
    Remove all leaf nodes (nodes with degree 1) and their incident edges from 
    the graph.
    
    This function creates a copy of the input graph and then iteratively removes 
    any node that has degree 1. Removing a node automatically removes its incident 
    edge(s). The process repeats until no leaf nodes remain.

    Parameters
    ----------
    G : nx.MultiGraph
        The input graph.

    Returns
    -------
    nx.MultiGraph
        A new graph with all leaf nodes (and their incident edges) removed.
    """
    H = G.copy()
    while True:
        # Identify all leaf nodes (nodes with degree exactly 1)
        leaf_nodes = [node for node, degree in H.degree() if degree == 1]
        if not leaf_nodes:
            break  # Exit when there are no leaf nodes left.
        for node in leaf_nodes:
            H.remove_node(node)
    return H



# Helpers for simplify_edges

def _append_edge_pts(path, edge_pts):
    """
    Append the list edge_pts to path, but
      1) reverse edge_pts if it currently runs *into* path[-1],
      2) drop the first point if it duplicates path[-1].
    """
    if edge_pts is None or len(edge_pts) == 0:
        return

    # 1) orient the segment so it starts at path[-1]
    if np.array_equal(edge_pts[-1], path[-1]):
        pts = edge_pts[::-1]
    else:
        pts = edge_pts

    # 2) drop the overlapping endpoint
    if np.array_equal(pts[0], path[-1]):
        path.extend(pts[1:])
    else:
        # this means something’s really off: 
        # neither end of pts matches the current path end
        raise RuntimeError(
            "Edge segment doesn’t connect contiguously:\n"
            f"  current tail = {path[-1]}\n"
            f"  segment ends = ({pts[0]}, {pts[-1]})"
        )



def _edge_tag(u: int, v: int, key: int) -> Tuple[int, int, int]:
    """Return a canonical tag for a multiedge so that (u,v,key) ≡ (v,u,key)."""
    return (u, v, key) if u <= v else (v, u, key)


def _has_cycles(G: nx.MultiGraph) -> bool:
    """Quick check for *any* cycle in *G*."""
    if G.number_of_edges() == 0:
        return False
    try:
        nx.find_cycle(G)
        return True
    except nx.NetworkXNoCycle:
        return False


def _collapse_component_with_junctions(
    G: nx.MultiGraph, comp: Set[int], H: nx.MultiGraph
) -> None:
    """Collapse chains inside *comp* that contains ≥1 junction (deg>2) nodes."""

    junctions = {n for n in comp if G.degree(n) > 2}
    # 1. Copy junctions verbatim to *H*
    for j in junctions:
        H.add_node(j, **G.nodes[j])

    seen_edges: Set[Tuple[int, int, int]] = set()

    for j in junctions:
        for nbr, edict in G.adj[j].items():
            for key, ea in edict.items():
                tag = _edge_tag(j, nbr, key)
                if tag in seen_edges:
                    continue
                seen_edges.add(tag)

                # --- collect path points from j to the next junction ---
                path_pts: List[NDArray] = [G.nodes[j]["pos"]]
                _append_edge_pts(path_pts, ea.get("pts", []))

                prev, cur = j, nbr
                # Walk until we land on a junction or get stuck.
                while cur not in junctions and G.degree(cur) == 2:
                    path_pts.append(G.nodes[cur]["pos"])

                    # Step to the only neighbour different from *prev*
                    nxt_candidates = [n for n in G.neighbors(cur) if n != prev]
                    if not nxt_candidates:
                        break  # dead‑end
                    nxt = nxt_candidates[0]

                    # add (cur, nxt) edge points (first non‑visited multiedge)
                    for k2, ea2 in G[cur][nxt].items():
                        tag2 = _edge_tag(cur, nxt, k2)
                        if tag2 not in seen_edges:
                            seen_edges.add(tag2)
                            _append_edge_pts(path_pts, ea2.get("pts", []))
                            break
                    prev, cur = cur, nxt

                # close the chain on final node *cur*
                path_pts.append(G.nodes[cur]["pos"])
                if cur not in H:
                    H.add_node(cur, **G.nodes[cur])

                H.add_edge(j, cur, pts=np.asarray(path_pts))


def _collapse_cycle_component(
    G: nx.MultiGraph, comp: Set[int], H: nx.MultiGraph
) -> None:
    """Collapse a component with *no* junctions (all deg≤2) to a self‑loop."""

    # Choose a representative node (prefer a degree‑2 node if available).
    rep = next((n for n in comp if G.degree(n) == 2), None) or next(iter(comp))
    H.add_node(rep, **G.nodes[rep])

    # Nothing to collapse if isolated node.
    if G.degree(rep) == 0:
        return

    path_pts: List[NDArray] = [G.nodes[rep]["pos"]]
    seen_edges: Set[Tuple[int, int, int]] = set()

    # Initial step: pick an arbitrary outgoing edge (rep, cur)
    prev, cur = rep, next(iter(G.neighbors(rep)))

    for key, ea in G[prev][cur].items():
        tag = _edge_tag(prev, cur, key)
        if tag not in seen_edges:
            seen_edges.add(tag)
            _append_edge_pts(path_pts, ea.get("pts", []))
            break

    # Walk around the cycle until we return to *rep*
    while cur != rep:
        path_pts.append(G.nodes[cur]["pos"])
        nxt_candidates = [n for n in G.neighbors(cur) if n != prev]
        if not nxt_candidates:  # open chain – shouldn’t happen for a cycle comp
            break
        nxt = nxt_candidates[0]

        for k2, ea2 in G[cur][nxt].items():
            tag2 = _edge_tag(cur, nxt, k2)
            if tag2 not in seen_edges:
                seen_edges.add(tag2)
                _append_edge_pts(path_pts, ea2.get("pts", []))
                break
        prev, cur = cur, nxt

    # Close the loop
    path_pts.append(G.nodes[rep]["pos"])

    H.add_edge(rep, rep, pts=np.asarray(path_pts))


# Public API
def simplify_edges(G: nx.MultiGraph) -> nx.MultiGraph:
    """Collapse degree‑2 chains in *G* while preserving geometry.

    The algorithm operates *per connected component*:
    1. **Early exit** – If *G* contains **no cycles**, return a copy of the
       node‑only graph (no edges).
    2. Components that include *junction* nodes (degree > 2) have each chain
       between junctions collapsed to a *single* multiedge storing all the
       collected poly‑line points.
    3. Components with *no* junctions (i.e. simple cycles / isolated edges)
       are reduced to a single self‑loop on a representative node while
       preserving the entire path.

    Parameters
    ----------
    G : nx.MultiGraph
        Input graph. Each node should have a **pos** attribute containing its
        3‑vector coordinates. Each edge may optionally have a **pts** attribute
        (sequence of points *between* the two endpoints).

    Returns
    -------
    nx.MultiGraph
        A new graph with the same geometric information but with chains of
        degree‑2 nodes collapsed. Node labels are re‑indexed starting from 1.
    """

    G = nx.MultiGraph(G)  # work on a copy to keep the original intact.

    # (0) Early‑exit: the graph is empty or acyclic – no knot / link structure.
    if not _has_cycles(G):
        logging.info("No cycles found – returning node-only graph."
                     " May imply no knot / link structure.")
        H = nx.MultiGraph()
        for n, data in G.nodes(data=True):
            H.add_node(n, **data)
        return nx.convert_node_labels_to_integers(H, first_label=1)

    # (1) Process each connected component independently.
    H = nx.MultiGraph()
    for comp in nx.connected_components(G):
        # Any node with deg>2 marks a junction.
        if any(G.degree(n) > 2 for n in comp):
            _collapse_component_with_junctions(G, comp, H)
        else:
            _collapse_cycle_component(G, comp, H)

    # (2) Relabel nodes for compactness and deterministic order.
    return nx.convert_node_labels_to_integers(H, first_label=1, ordering="sorted")



# def get_edge_pts(G):
#     pts_list = []
#     for u, v, pts in G.edges(data='pts'):
#         pts_list.append(pts)
#     if pts_list:
#         return np.vstack(pts_list)
#     else:
#         return np.array([])
    
# def get_node_pts(G):
#     pts_list = []
#     for n, o in G.nodes(data='pos'):
#         pts_list.append(o)
#     if pts_list:
#         return np.vstack(pts_list)
#     else:
#         return np.array([])
    
# def get_all_pts(G):
#     edge_pts = get_edge_pts(G)
#     node_pts = get_node_pts(G)
#     if edge_pts.size > 0 and node_pts.size > 0:
#         return np.vstack((edge_pts, node_pts))
#     elif edge_pts.size > 0:
#         return edge_pts
#     elif node_pts.size > 0:
#         return node_pts
#     else:
#         return np.array([])