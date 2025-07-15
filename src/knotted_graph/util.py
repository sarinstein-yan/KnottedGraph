import numpy as np
import sympy as sp
import networkx as nx
from rdp import rdp, pldist
from typing import Union, Any
from numpy.typing import NDArray

# PT-operator
def PT(h: sp.Matrix) -> sp.Matrix:
    """Parity-time operator."""
    sx = sp.Matrix([[0, 1], [1, 0]])
    return sx * h.conjugate() * sx

def is_PT_symmetric(h: sp.Matrix) -> bool:
    """Check if a Hamiltonian is PT-symmetric."""
    return sp.simplify(h - PT(h)) == 0


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
            "Edge segment doesn’t connect: "
            f"path[-1]={path[-1]}, next_pts endpoints={pts[0],pts[-1]}"
        )


def remove_deg2_preserving_pts(
    G: nx.MultiGraph
) -> nx.MultiGraph:
    """
    Collapse degree-2 chains in each connected component of G, 
    retaining node and edge points along paths.
    - For components with any nodes of degree > 2: treat those 
    nodes as junctions and collapse chains between them.
    - For components without any nodes of degree > 2: collapse 
    the entire component into a single self-loop on a chosen 
    representative node, preserving the full path points.

    Parameters
    ----------
    G : nx.MultiGraph
        The input graph. Each node should have a 'pos' attribute, 
        and each edge may have a 'pts' attribute.

    Returns
    -------
    nx.MultiGraph
        A new graph with degree-2 chains collapsed. Node and edge 
        points are preserved along the collapsed paths.
    """
    G = nx.MultiGraph(G)
    H = nx.MultiGraph()

    def _tag(u, v, key):
        return (u, v, key) if u <= v else (v, u, key)

    for comp in nx.connected_components(G):
        # map degrees
        degmap = {n: G.degree(n) for n in comp}
        junctions = {n for n, d in degmap.items() if d > 2}

        if junctions:
            # standard collapse for components with junctions
            for j in junctions:
                H.add_node(j)
                H.nodes[j]['pos'] = G.nodes[j].get('pos')

            seen_edges = set()
            for j in junctions:
                for nbr, edict in G.adj[j].items():
                    for key, ea in edict.items():
                        tag = _tag(j, nbr, key)
                        if tag in seen_edges:
                            continue
                        seen_edges.add(tag)

                        # collect chain from j to next junction
                        path_pts = [G.nodes[j].get('pos')]
                        _append_edge_pts(path_pts, ea.get('pts', []))

                        prev, cur = j, nbr
                        while cur not in junctions and G.degree(cur) == 2:
                            path_pts.append(G.nodes[cur].get('pos'))
                            # step to the other neighbor
                            #nxt = [n for n in G.neighbors(cur) if n != prev][0]
                            # look for the one neighbor that isn’t `prev`
                            candidates = [n for n in G.neighbors(cur) if n != prev]
                            if not candidates:
                                # no further neighbor → stop collapsing this chain
                                break
                            nxt = candidates[0]

                            for k2, ea2 in G[cur][nxt].items():
                                t2 = _tag(cur, nxt, k2)
                                if t2 not in seen_edges:
                                    seen_edges.add(t2)
                                    _append_edge_pts(path_pts, ea2.get('pts', []))
                                    break
                            prev, cur = cur, nxt

                        # append final node
                        path_pts.append(G.nodes[cur].get('pos'))
                        if cur not in H.nodes:
                            H.add_node(cur)
                            H.nodes[cur]['pos'] = G.nodes[cur].get('pos')

                        H.add_edge(j, cur, pts=np.array(path_pts))
        else:
            # no junctions: make one self-loop
            # choose representative (prefer deg-2)
            rep = next((n for n, d in degmap.items() if d == 2), None) or next(iter(comp))
            H.add_node(rep)
            H.nodes[rep]['pos'] = G.nodes[rep].get('pos')

            # start path at rep
            path_pts = [G.nodes[rep].get('pos')]
            seen_edges = set()
            nbrs = list(G.neighbors(rep))
            if nbrs:
                prev, cur = rep, nbrs[0]
                # record first edge
                for key, ea in G[prev][cur].items():
                    tag = _tag(prev, cur, key)
                    if tag not in seen_edges:
                        seen_edges.add(tag)
                        _append_edge_pts(path_pts, ea.get('pts', []))
                        break
                # walk until back to rep
                while cur != rep:
                    path_pts.append(G.nodes[cur].get('pos'))
                    nxts = [n for n in G.neighbors(cur) if n != prev]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    for k2, ea2 in G[cur][nxt].items():
                        t2 = _tag(cur, nxt, k2)
                        if t2 not in seen_edges:
                            seen_edges.add(t2)
                            _append_edge_pts(path_pts, ea2.get('pts', []))
                            break
                    prev, cur = cur, nxt
                # close loop
                path_pts.append(G.nodes[rep].get('pos'))

            H.add_edge(rep, rep, pts=np.array(path_pts))
    
    H = nx.convert_node_labels_to_integers(H, first_label=1, ordering='sorted')
    return H


def get_edge_pts(G):
    pts_list = []
    for u, v, pts in G.edges(data='pts'):
        pts_list.append(pts)
    if pts_list:
        return np.vstack(pts_list)
    else:
        return np.array([])
    
def get_node_pts(G):
    pts_list = []
    for n, o in G.nodes(data='pos'):
        pts_list.append(o)
    if pts_list:
        return np.vstack(pts_list)
    else:
        return np.array([])
    
def get_all_pts(G):
    edge_pts = get_edge_pts(G)
    node_pts = get_node_pts(G)
    if edge_pts.size > 0 and node_pts.size > 0:
        return np.vstack((edge_pts, node_pts))
    elif edge_pts.size > 0:
        return edge_pts
    elif node_pts.size > 0:
        return node_pts
    else:
        return np.array([])