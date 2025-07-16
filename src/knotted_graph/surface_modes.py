import numpy as np
import sympy as sp
from poly2graph import (
    kron_batch, 
    eig_batch,
    shift_matrix,
    hk2hz, 
    hz2hk, 
    expand_hz_as_hop_dict,
)

__all__ = [
    "hop_dict_by_direction",
    "H_batch_from_hop_dict",
    "H_batch",
]


def hop_dict_by_direction(hamil_sympy, direction, k2, k3):
    """
    Extracts a dictionary of hopping terms from a Bloch Hamiltonian for a specified spatial direction.

    The function rotates the momentum arguments of the given Hamiltonian function to isolate 
    contributions corresponding to the desired lattice direction ('x', 'y', or 'z'). It then 
    performs a substitution to express the Hamiltonian in terms of complex variables and expands 
    it into polynomial terms.

    Parameters:
        hamil_sympy: A function that takes three momentum arguments and returns a sympy Matrix 
                     representing the Hamiltonian.
        direction (str): The spatial direction ('x', 'y', or 'z') for which to extract hopping terms.
        k2, k3: sympy symbols representing the momentum components that remain after choosing the direction.

    Returns:
        A tuple (h_dict, k2, k3) where:
            - h_dict is a dictionary mapping integer shifts (the exponent of the expansion variable)
              to sympy Matrices representing the corresponding hopping terms.
            - k2 and k3 are the momentum symbols corresponding to the remaining dimensions.
    
    Raises:
        ValueError: If the provided direction is not one of 'x', 'y', or 'z'.
    """
    # Define dummy variable k1 as a symbol for the direction of interest
    k1 = sp.symbols('k1', real=True)

    h_dict = {}
    zx, zy, zz = sp.symbols('zx zy zz', complex=True)
    if direction == 'x':
        hk = hamil_sympy(k1, k2, k3)
        hz = hk2hz(hk, k1, k2, k3, zx, None, None)
        h_dict_temp = expand_hz_as_hop_dict(hz, zx, zy, zz)
        for k, v in h_dict_temp.items():
            assert k[1] == 0 and k[2] == 0
            h_dict[k[0]] = v
    elif direction == 'y':
        hk = hamil_sympy(k3, k1, k2)
        hz = hk2hz(hk, k3, k1, k2, None, zy, None)
        h_dict_temp = expand_hz_as_hop_dict(hz, zx, zy, zz)
        for k, v in h_dict_temp.items():
            assert k[0] == 0 and k[2] == 0
            h_dict[k[1]] = v
    elif direction == 'z':
        hk = hamil_sympy(k2, k3, k1)
        hz = hk2hz(hk, k2, k3, k1, None, None, zz)
        h_dict_temp = expand_hz_as_hop_dict(hz, zx, zy, zz)
        for k, v in h_dict_temp.items():
            assert k[0] == 0 and k[1] == 0
            h_dict[k[2]] = v
    else:
        raise ValueError("direction must be one of 'x', 'y', or 'z'")

    return h_dict, k2, k3

def H_batch_from_hop_dict(hoppings, N1, k2, k2_vals, k3, k3_vals):
    """
    Constructs an array of Hamiltonian matrices for a 1D chain using both open and periodic boundary conditions.

    The function uses a dictionary of hopping terms (which may depend on momentum components k2 and k3) 
    to build the full Hamiltonian on a lattice grid. It leverages TensorFlow's linear operators and 
    constructs tensors for both open (obc) and periodic (pbc) boundary conditions.

    Parameters:
        hoppings (dict): Dictionary where keys are integer shifts and values are sympy Matrices representing
                         the hopping terms. These terms can be functions of momentum components k2 and k3.
        N1 (int): The number of sites in the 1D chain.
        k2, k3: sympy symbols representing the momentum components in the other two directions.
        k2_vals (list-like): Numerical values for the momentum component k2.
        k3_vals (list-like): Numerical values for the momentum component k3.

    Returns:
        A tuple (Hobc_arr, Hpbc_arr) where:
            - Hobc_arr is an array of Hamiltonian matrices with open boundary conditions.
            - Hpbc_arr is an array of Hamiltonian matrices with periodic boundary conditions.
    """
    Hobc_arr = None
    Hpbc_arr = None

    for shift, val in hoppings.items():
        Tobc = shift_matrix(N1, shift, pbc=False).astype(np.complex128)
        Tpbc = shift_matrix(N1, shift, pbc=True).astype(np.complex128)
        y_len, z_len = len(k2_vals), len(k3_vals)
        Tobc_arr = np.tile(Tobc, (y_len, z_len, 1, 1))
        Tpbc_arr = np.tile(Tpbc, (y_len, z_len, 1, 1))

        if val.free_symbols.isdisjoint({k2, k3}):
            hop = np.array(val.tolist(), dtype=np.complex128)
            hop_arr = np.tile(hop, (y_len, z_len, 1, 1))
        else:
            k2_arr, k3_arr = np.meshgrid(k2_vals, k3_vals, indexing='ij')
            hop_arr = np.zeros((y_len, z_len, *val.shape), dtype=np.complex128)
            for idx in np.ndindex(val.shape):
                f = sp.lambdify((k2, k3), val[idx], modules='numpy')
                hop_arr[..., *idx] = f(k2_arr, k3_arr)

        # NOTE: this is still the slowest part of this function.
        Hobc_temp = kron_batch(Tobc_arr, hop_arr)
        Hpbc_temp = kron_batch(Tpbc_arr, hop_arr)
        
        Hobc_arr = Hobc_temp if Hobc_arr is None else Hobc_arr + Hobc_temp
        Hpbc_arr = Hpbc_temp if Hpbc_arr is None else Hpbc_arr + Hpbc_temp

    return Hobc_arr, Hpbc_arr

# --- construct array of Hamiltonian setting OBC in one direction --- #
def H_batch(h_k_func, direction, N1, k2_vals, k3_vals):
    """
    Constructs Hamiltonian matrices on a arr for a 1D chain using both open and periodic boundary conditions.
    This function is a wrapper around the `H_obc_arr_from_hop_dict` function, allowing for
    the specification of a Hamiltonian function and the direction of interest.

    Parameters:
        h_k_func: A function that takes three momentum arguments and returns a sympy Matrix representing the Hamiltonian.
        direction (str): The spatial direction ('x', 'y', or 'z') for which to extract hopping terms.
        N1 (int): The number of sites in the 1D chain.
        k2_vals (list-like): Numerical values for the momentum component k2.
        k3_vals (list-like): Numerical values for the momentum component k3.
        
    Returns:
        A tuple (Hobc_arr, Hpbc_arr)
            - Hobc_arr is an array of Hamiltonian matrices with open boundary conditions.
            - Hpbc_arr is an array of Hamiltonian matrices with periodic boundary conditions.
    Raises:
        ValueError: If the provided direction is not one of 'x', 'y', or 'z'.
    """
    k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
    hoppings, k2, k3 = hop_dict_by_direction(h_k_func, direction, k2, k3)
    Hobc_arr, Hpbc_arr = H_batch_from_hop_dict(hoppings, N1, k2, k2_vals, k3, k3_vals)
    return Hobc_arr, Hpbc_arr





if __name__ == "__main__":

    # Define Pauli matrices.
    sigma_x = sp.ImmutableMatrix([[0, 1], [1, 0]])
    sigma_y = sp.ImmutableMatrix([[0, -sp.I], [sp.I, 0]])
    sigma_z = sp.ImmutableMatrix([[1, 0], [0, -1]])

    def hHopf_Herm(kx, ky, kz, t=0.):
        """
        Constructs the Hermitian Hopf Hamiltonian as a 2x2 sympy Matrix for given momentum components.
        Using Hermitian ansatz.

        Parameters:
            kx, ky, kz: sympy expressions or symbols representing the momentum components.

        Returns:
            A simplified 2x2 sympy Matrix representing the Hopf Bloch Hamiltonian.
        """
        z = sp.cos(2 * kz) + sp.Rational(1, 2) + sp.I * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - 2)
        w = sp.sin(kx) + sp.I * sp.sin(ky)
        f_val = z**2 - w**2
        # Construct the Hamiltonian.
        H = sp.re(f_val) * sigma_x + sp.im(f_val) * sigma_z
        if t != 0:
            H += t * sp.I * sigma_y
        return sp.simplify(H)

    def hHopf_nHerm(kx, ky, kz, m=2):
        """
        Constructs a non-Hermitian 2x2 Hopf Bloch Hamiltonian as a sympy Matrix.
        Using non-Hermitian ansatz.

        Parameters:
            kx, ky, kz: sympy expressions or symbols representing the momentum components.
            m (int, optional): Parameter modifying the Hamiltonian (default is 2).

        Returns:
            A simplified 2x2 sympy Matrix representing the non-Hermitian Hopf Bloch Hamiltonian.
        """
        mu = sp.sin(kz) + sp.I * (sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - m)
        w = sp.sin(kx) + sp.I * sp.sin(ky)
        # Construct the Hamiltonian.
        h = mu * sigma_x + w * sigma_y
        return sp.simplify(h)
    
    kx, ky, kz = sp.symbols('k_x k_y k_z', real=True)
    hk = hHopf_Herm(kx, ky, kz)
    sp.pprint(hk)

    zx, zy, zz = sp.symbols('z_x z_y z_z', complex=True)
    hz = hk2hz(hk, kx, ky, kz, zx, zy, zz)
    sp.pprint(hz)

    hop_dict = expand_hz_as_hop_dict(hz, zx, zy, zz)
    sp.pprint(hop_dict)

    N = 30
    nH_coeff = 0.
    kx_vals = np.linspace(-np.pi, np.pi, 100)
    ky_vals = np.linspace(-np.pi, np.pi, 100)
    kz_vals = np.linspace(0, np.pi, 100)

    h_obc_x, h_pbc_x = H_batch(hHopf_Herm, 'x', N, ky_vals, kz_vals)
    h_obc_y, h_pbc_y = H_batch(hHopf_Herm, 'y', N, kz_vals, kx_vals)
    h_obc_z, h_pbc_z = H_batch(hHopf_Herm, 'z', N, kx_vals, ky_vals)

    eigvals_obc_x, eigvecs_obc_x = eig_batch(h_obc_x)
    eigvals_obc_y, eigvecs_obc_y = eig_batch(h_obc_y)
    eigvals_obc_z, eigvecs_obc_z = eig_batch(h_obc_z)

    from knotted_graph.vis import plot_surface_modes

    fig = plot_surface_modes(
            (eigvals_obc_x, eigvals_obc_y, eigvals_obc_z),
            (kx_vals, ky_vals, kz_vals),
            (0.04, 0.07, 0.07),
            nH_coeff=nH_coeff,
        )
    fig.show()