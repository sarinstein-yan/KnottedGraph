import numpy as np
import networkx as nx
import tensorflow as tf
from typing import Union

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