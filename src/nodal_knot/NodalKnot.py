import numpy as np
import skimage.morphology as morph
from .utils import plot_3D_and_2D_projections
import poly2graph as p2g
import networkx as nx

class NodalKnot:
    def __init__(self, 
            k_to_zw_func, 
            zw_to_c_func
        ):
        """
        Initialize the non-Hermitian `NodalKnot` with two functions:
            1. k_to_zw_func : callable
                A vectorized function mapping (kx, ky, kz) -> (z, w).
            2. zw_to_c_func : callable
                A vectorized function mapping (z, w) -> c.

        Parameters:
        ----------
        k_to_zw_func : callable
            A vectorized function mapping (kx, ky, kz) -> (z, w).
        zw_to_c_func : callable
            A vectorized function mapping (z, w) -> c.

        """
        self.k_to_zw_func = k_to_zw_func
        self.zw_to_c_func = zw_to_c_func

        # Initialize the attributes
        self.kx_min = -np.pi; self.kx_max = np.pi
        self.ky_min = -np.pi; self.ky_max = np.pi
        self.kz_min = 0; self.kz_max = np.pi
        self.pts_per_dim = 400
        self.val = None
        self.kx_grid = None
        self.ky_grid = None
        self.kz_grid = None
        self.binarized_val = None
        self.selected_points = None
        self.skeleton = None
        self.skeleton_points = None
        self.graph = None


    def generate_region(self, 
            kx_min=-np.pi, kx_max=np.pi, 
            ky_min=-np.pi, ky_max=np.pi, 
            kz_min=0, kz_max=np.pi,
            pts_per_dim=400,
        ):
        """
        Generate values of f(z, w) for a grid of (kx, ky, kz) points.

        Parameters:
        ----------
        kx_min, kx_max : float
            Range for kx. Default is [-pi, pi].
        ky_min, ky_max : float
            Range for ky. Default is [-pi, pi].
        kz_min, kz_max : float
            Range for kz. Default is [0, pi].
        pts_per_dim : int, optional
            Number of points to sample per dimension. Default is 400.        

        Returns:
        -------
        val : np.ndarray
            Values of f(z, w).
        kx_grid, ky_grid, kz_grid : np.ndarray
            The 3D grids of kx, ky, kz values.
        """
        kx_vals = np.linspace(kx_min, kx_max, pts_per_dim)
        ky_vals = np.linspace(ky_min, ky_max, pts_per_dim)
        kz_vals = np.linspace(kz_min, kz_max, pts_per_dim)
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')

        z, w = self.k_to_zw_func(kx_grid, ky_grid, kz_grid)
        val = self.zw_to_c_func(z, w)

        self.kx_min = kx_min; self.kx_max = kx_max
        self.ky_min = ky_min; self.ky_max = ky_max
        self.kz_min = kz_min; self.kz_max = kz_max
        self.pts_per_dim = pts_per_dim
        self.val = val
        self.kx_grid = kx_grid
        self.ky_grid = ky_grid
        self.kz_grid = kz_grid

        return val, kx_grid, ky_grid, kz_grid

    def binarize_region(self,
            thickness=0.,
            epsilon=None,
            **kwargs
        ):
        """
        Binarize the region based on the thickness constant and threshold
        epsilon, i.e., abs(|f(z, w)| - thickness) < epsilon.

        Parameters:
        ----------
        thickness : float
            The thickness constant. Default is 0.
        epsilon : float, optional
            If > 0, will return the surface of the thickened knot; otherwise,
            return as a solid (fill up the interior). Default is None.
        kwargs :
            Additional keyword arguments for `generate_region`.

        Returns:
        -------
        binarized_val : np.ndarray
            3D array with 1s for zero regions and 0s otherwise.
        """
        # check if self.val is available
        if self.val is None or kwargs:
             self.generate_region(**kwargs)

        norm = np.abs(self.val)
        if epsilon is not None and epsilon > 0:
            # return the thickened knot's surface
            binarized_val = np.where(np.abs(norm - thickness) < epsilon, 1, 0)
        else:
            # return the thickened knot as a solid (filled-up surface)
            if thickness < 10/self.pts_per_dim: thickness = 10/self.pts_per_dim
            binarized_val = np.where(norm <= thickness, 1, 0)
        self.binarized_val = binarized_val

        return binarized_val

    def knot_surface_points(self,
            thickness=0.,
            epsilon=None,
            idx=None,
            **kwargs
        ):
        """
        Find the zero points in 3D space where 
            abs(|f(z, w)| - thickness) < epsilon.

        Parameters:
        ----------
        thickness : float
            The thickness constant. Default is 0.
        epsilon : float, optional
            If > 0, will return the surface of the thickened knot; otherwise,
            return as a solid (fill up the interior). Default is None.
        idx : np.ndarray, optional
            Indices of the zero points. Default is None, determined by the
            thresholding condition. If idx is provided, the zero points are
            selected based on the indices.
        kwargs :
            Additional keyword arguments for `generate_region`.

        Returns:
        -------
        selected_points : np.ndarray
            A list of points in 3D (kx, ky, kz) space that satisfy the condition.
            
        """
        # check if self.val is available
        if self.val is None or kwargs:
             self.generate_region(**kwargs)
        if idx is None:
            norm = np.abs(self.val)
            if epsilon is not None and epsilon > 0:
                # return the thickened knot's surface
                idx = np.where(np.abs(norm - thickness) < epsilon)
            else:
                # return the thickened knot as a solid (filled-up surface)
                if thickness < 10/self.pts_per_dim: thickness = 10/self.pts_per_dim
                idx = np.where(norm <= thickness)
        else:
            if idx.shape != self.val.shape:
                raise ValueError("Input `val` must have the same shape as the generated region.")
            idx = np.where(idx)

        pts = np.array([self.kx_grid[idx],
                        self.ky_grid[idx],
                        self.kz_grid[idx]]).T
        
        if idx is None: self.selected_points = pts

        return pts
    
    def knot_skeleton(self,
            thickness=0.,
            epsilon=None,
            **kwargs
        ):
        """
        Generate the skeleton of the zero region.

        Parameters:
        ----------
        thickness : float
            The thickness constant. Default is 0.
        epsilon : float, optional
            If > 0, will return the surface of the thickened knot; otherwise,
            return as a solid (fill up the interior). Default is None.
        kwargs :
            Additional keyword arguments for `generate_region`.

        Returns:
        -------
        skeleton : np.ndarray
            The skeleton of the zero region as a 3D image.
        """
        self.binarize_region(thickness, epsilon, **kwargs)
        skeleton = morph.skeletonize(self.binarized_val, method='lee')
        self.skeleton = skeleton
        return skeleton
        
    def knot_skeleton_points(self,
            thickness=0.,
            epsilon=None,
            volume_representation=False,
            **kwargs
        ):
        """
        The list of points on the skeleton of the zero region, or a volume representation thereof.

        Parameters:
        ----------
        thickness : float
            The thickness constant. Default is 0.
        epsilon : float, optional
            If > 0, will return the surface of the thickened knot; otherwise,
            return as a solid (fill up the interior). Default is None.
        volume_representation : bool
            If True, returns a volume representation of the skeleton as a 3D numpy array.
            Otherwise, returns the list of 3D coordinates.
        kwargs :
            Additional keyword arguments for `generate_region`.

        Returns:
        -------
        skeleton_points : np.ndarray
            If volume_representation is False: The list of 3D coordinates on the skeleton of the zero region.
            If volume_representation is True: A 3D numpy array (volume) with ones at the voxel positions
            corresponding to the skeleton points.
        """
        # First, generate the skeleton
        self.knot_skeleton(thickness, epsilon, **kwargs)
        # Extract the skeleton points based on the skeletonized volume
        points = self.knot_surface_points(idx=self.skeleton)
        self.skeleton_points = points

        if volume_representation:
            # Create an empty volume based on the number of points per dimension
            pts_per_dim = self.pts_per_dim
            volume = np.zeros((pts_per_dim, pts_per_dim, pts_per_dim), dtype=np.uint8)

            # Get the grid range values from the instance
            kx_min, kx_max = self.kx_min, self.kx_max
            ky_min, ky_max = self.ky_min, self.ky_max
            kz_min, kz_max = self.kz_min, self.kz_max

            # Compute indices for each coordinate component
            x_indices = np.rint((points[:, 0] - kx_min) / (kx_max - kx_min) * (pts_per_dim - 1)).astype(int)
            y_indices = np.rint((points[:, 1] - ky_min) / (ky_max - ky_min) * (pts_per_dim - 1)).astype(int)
            z_indices = np.rint((points[:, 2] - kz_min) / (kz_max - kz_min) * (pts_per_dim - 1)).astype(int)

            # Optional: Check if any indices are out of bounds
            if (x_indices.min() < 0 or x_indices.max() >= pts_per_dim or
                y_indices.min() < 0 or y_indices.max() >= pts_per_dim or
                z_indices.min() < 0 or z_indices.max() >= pts_per_dim):
                raise ValueError("Some points fall outside the specified volume range.")

            # Set the corresponding voxels to 1.
            volume[x_indices, y_indices, z_indices] = 1

            return volume

        return points

            

    @staticmethod
    def remove_leaf(G: nx.Graph) -> nx.Graph:
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

    def convert_to_graph(self, volume_points, clean=True):
        """
        Convert a volume representation of skeleton points to a graph, and optionally remove leaf nodes.
        
        Parameters:
        -----------
        volume_points : np.ndarray
            A 3D numpy array representing the volume. Voxels with value 1 are taken as skeleton points.
        clean : bool, optional
            If True, remove leaf nodes (nodes with degree 1) from the graph. Default is True.
        
        Returns:
        --------
        G : nx.Graph
            The graph constructed from the volume (optionally cleaned).
        """
        # Convert the volume to a graph using poly2graph
        G = p2g.skeleton2graph(volume_points)
        
        # Remove leaf nodes if desired
        if clean:
            G = NodalKnot.remove_leaf(G)
        
        self.graph = G
        return G
    
    
    def plot_3D(self, points, file_name=None):
        """
        Plot the zero regions in 3D space and their 2D projections.

        Parameters:
        ----------
        points : Array-like
            Points in 3D (kx, ky, kz) space to plot.
        file_name : str, optional
            File name to save the plot. If None, no saving is done.
            Default is None.

        Returns:
        -------
        fig : plotly.graph_objects.Figure
            The Plotly figure object for visualization.
        """
        fig = plot_3D_and_2D_projections(points)

        if file_name:
            fig.write_html(file_name)
        
        return fig