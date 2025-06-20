import numpy as np
import networkx as nx
import tensorflow as tf
from typing import Union
import networkx as nx
  
def append_edge_pts(path, edge_pts):
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
 

def collapse_deg2_exact_with_pts(G):
    """
    Collapse degree-2 chains in each connected component of G, retaining node‑ and edge‑pts along paths.

    - For components with any deg>2 nodes: treat deg>2 nodes as junctions and collapse chains between them using the standard logic.
    - For components without deg>2 nodes: collapse the entire component into a single self-loop on a chosen representative node, preserving the full path pts.
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
                H.nodes[j]['o'] = G.nodes[j].get('o')

            seen_edges = set()
            for j in junctions:
                for nbr, edict in G.adj[j].items():
                    for key, ea in edict.items():
                        tag = _tag(j, nbr, key)
                        if tag in seen_edges:
                            continue
                        seen_edges.add(tag)

                        # collect chain from j to next junction
                        path_pts = [G.nodes[j].get('o')]
                        append_edge_pts(path_pts, ea.get('pts', []))

                        prev, cur = j, nbr
                        while cur not in junctions and G.degree(cur) == 2:
                            path_pts.append(G.nodes[cur].get('o'))
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
                                    append_edge_pts(path_pts, ea2.get('pts', []))
                                    break
                            prev, cur = cur, nxt

                        # append final node
                        path_pts.append(G.nodes[cur].get('o'))
                        if cur not in H.nodes:
                            H.add_node(cur)
                            H.nodes[cur]['o'] = G.nodes[cur].get('o')

                        H.add_edge(j, cur, pts=np.array(path_pts))
        else:
            # no junctions: make one self-loop
            # choose representative (prefer deg-2)
            rep = next((n for n, d in degmap.items() if d == 2), None) or next(iter(comp))
            H.add_node(rep)
            H.nodes[rep]['o'] = G.nodes[rep].get('o')

            # start path at rep
            path_pts = [G.nodes[rep].get('o')]
            seen_edges = set()
            nbrs = list(G.neighbors(rep))
            if nbrs:
                prev, cur = rep, nbrs[0]
                # record first edge
                for key, ea in G[prev][cur].items():
                    tag = _tag(prev, cur, key)
                    if tag not in seen_edges:
                        seen_edges.add(tag)
                        append_edge_pts(path_pts, ea.get('pts', []))
                        break
                # walk until back to rep
                while cur != rep:
                    path_pts.append(G.nodes[cur].get('o'))
                    nxts = [n for n in G.neighbors(cur) if n != prev]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    for k2, ea2 in G[cur][nxt].items():
                        t2 = _tag(cur, nxt, k2)
                        if t2 not in seen_edges:
                            seen_edges.add(t2)
                            append_edge_pts(path_pts, ea2.get('pts', []))
                            break
                    prev, cur = cur, nxt
                # close loop
                path_pts.append(G.nodes[rep].get('o'))

            H.add_edge(rep, rep, pts=np.array(path_pts))
    H = nx.convert_node_labels_to_integers(H, first_label=1, ordering='sorted')
    return H

def remove_leaf_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """
    Remove all leaf nodes (nodes with degree 1) and their incident edges from the graph.
    
    This function creates a copy of the input graph and then iteratively removes any node
    that has degree 1. Removing a node automatically removes its incident edge(s).
    The process repeats until no leaf nodes remain.
    
    Parameters:
        G : nx.Graph or nx.MultiGraph
            The input graph.
            
    Returns:
        H : nx.Graph or nx.MultiGraph
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
    for n, o in G.nodes(data='o'):
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


# --- helper function: batched Kronecker product --- #
def kron_batched(a, b):
    """
    Compute the Kronecker product for each pair of matrices in batches using NumPy.
    
    Args:
        a: A numpy array of shape (..., N, N). N could be the lattice length.
        b: A numpy array of shape (..., m, m). m could be the no. of bands.
    
    Returns:
        A numpy array of shape (..., N*m, N*m) corresponding to the Kronecker product
        of the last two dimensions of a and b.
    """
    a = np.asarray(a); b = np.asarray(b)
    # Compute the batched outer product using einsum. 
    # The resulting tensor has shape (..., N, m, N, m)
    kron = np.einsum('...ij,...kl->...ikjl', a, b)
    # Determine the output shape: reshape (..., N, m, N, m) to (..., N*m, N*m)
    new_shape = a.shape[:-2] + (a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1])
    return kron.reshape(new_shape)

# from numba import njit
# @njit
# def batched_kron_nb(a, b):
#     """
#     Compute the batched Kronecker product for two rank-4 tensors.
    
#     Args:
#         a: A numpy array of shape (B, C, N, N). For instance, B and C could represent batch dimensions.
#         b: A numpy array of shape (B, C, m, m).
        
#     Returns:
#         A numpy array of shape (B, C, N*m, N*m) where each (N*m x N*m) block is the Kronecker product
#         of the corresponding (N x N) block from a and (m x m) block from b.
#     """
#     B, C, N, _ = a.shape
#     B2, C2, m, _ = b.shape
#     # Optionally, you might want to check that B == B2 and C == C2.
#     result = np.empty((B, C, N * m, N * m), dtype=a.dtype)
    
#     for batch in range(B):
#         for channel in range(C):
#             for i in range(N):
#                 for j in range(N):
#                     for k in range(m):
#                         for l in range(m):
#                             result[batch, channel, i * m + k, j * m + l] = a[batch, channel, i, j] * b[batch, channel, k, l]
#     return result

# @tf.function
# def batched_kron_tf(a, b):
#     """
#     Compute the Kronecker product for each pair of matrices in batches using TensorFlow.
    
#     Args:
#         a: A TensorFlow tensor of shape (..., N, N). N could be the lattice length.
#         b: A TensorFlow tensor of shape (..., m, m). m could be the no. of bands.
    
#     Returns:
#         A TensorFlow tensor of shape (..., N*m, N*m) corresponding to the Kronecker product
#         of the last two dimensions of a and b.
#     """
#     a = tf.convert_to_tensor(a); b = tf.convert_to_tensor(b)
#     # Compute the batched outer product using einsum. 
#     # The resulting tensor has shape (..., N, m, N, m)
#     kron = tf.einsum('...ij,...kl->...ikjl', a, b)
#     # Determine the output shape: reshape (..., N, m, N, m) to (..., N*m, N*m)
#     a_sp = tf.shape(a); b_sp = tf.shape(b)
#     new_shape = tf.concat([a_sp[:-2], [a_sp[-2] * b_sp[-2], a_sp[-1] * b_sp[-1]]], axis=0)
#     return tf.reshape(kron, new_shape)



def eig_batched(array_of_matrices, device='/CPU:0', is_hermitian=False):
    """
    Compute the eigenvalues and eigenvectors for a batch of matrices using TensorFlow.

    This function computes the eigen decomposition for a batch of matrices provided as an array.
    It supports both Hermitian matrices (using tf.linalg.eigh) and general matrices (using tf.linalg.eig).
    For general matrices, it improves numerical stability by setting near-zero entries (below a tolerance)
    to zero before computing the eigenvalues and eigenvectors.

    Parameters
    ----------
    array_of_matrices : array-like
        An array or tensor of shape (..., N, N) representing a batch of square matrices.
    device : str or tf.device
        The TensorFlow device (e.g., '/GPU:0' or '/CPU:0') on which the computation is performed.
    is_hermitian : bool, optional
        Flag indicating whether the input matrices are Hermitian. If True, uses tf.linalg.eigh.
        Otherwise, uses tf.linalg.eig with a numerical stability threshold. Default is False.

    Returns
    -------
    eigvals_np : np.ndarray
        A numpy array of eigenvalues with shape matching the batch dimensions and an extra dimension 
        for eigenvalues.
    eigvecs_np : np.ndarray
        A numpy array of eigenvectors with shape matching the batch dimensions and two extra dimensions
        for the eigenvector matrices.

    Raises
    ------
    ValueError
        If the tensor's dtype is not one of [tf.float32, tf.float64, tf.complex64, tf.complex128].

    Notes
    -----
    - For non-Hermitian matrices, the tolerance for setting near-zero values is chosen based on the data type:
      1e-14 for complex dtypes (tf.complex64, tf.complex128) and 1e-6 for float dtypes (tf.float32, tf.float64).
    - The resulting eigenvalues and eigenvectors are converted to numpy arrays, and the computation is performed
      on the specified device.
    """
    with tf.device(device):
        array_of_matrices = tf.convert_to_tensor(array_of_matrices)

        if is_hermitian:
            vals, vecs = tf.linalg.eigh(array_of_matrices)
        
        else:
            # Set near-zero entries to zero for numerical stability.
            if array_of_matrices.dtype in [tf.complex64, tf.complex128]:
                tol = 1e-14
            elif array_of_matrices.dtype in [tf.float32, tf.float64]:
                tol = 1e-6
            else: raise ValueError("Unsupported dtype. dtype must be one of "
                                "[float32, float64, complex64, complex128].")
            array_of_matrices = tf.where(tf.abs(array_of_matrices) < tol, 0., array_of_matrices)
            vals, vecs = tf.linalg.eig(array_of_matrices)
        
        # Convert to numpy array; data now on CPU.
        eigvals_np = vals.numpy(); eigvecs_np = vecs.numpy()
    
    # # Clear the TensorFlow session/graph state to release GPU memory.
    # tf.keras.backend.clear_session()

    return eigvals_np, eigvecs_np





