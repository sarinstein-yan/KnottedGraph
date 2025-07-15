import numpy as np
import networkx as nx
import skimage.morphology as morph
import minorminer
from poly2graph import skeleton2graph
import logging
from tabulate import tabulate

from knotted_graph.vis import (
    plot_3D_and_2D_projections,
    plot_3D_graph,
)
from knotted_graph.util import (
    remove_leaf_nodes,
    remove_deg2_preserving_pts,
    smooth_edges,
)

# TODO:
# - [x] initialize with sympy
# - [] solve 3D real space H, find the zero eigenvalue boundaries
# - [x] skeleton graph by skan
# - [x] graph simplification by `OSPnx.simplification`
# - [z] initialize unparametrized
# - Graph as: 
#       - [x] nodes as points
#       - edges as 
#           - [] shapely.LineString, 
#           - [x] `rdp`-simplified
#           - [x] when plotting, wrap in pv.Spline
# - [] Berry Curvature function and field plotted by pv's glyphs
# - [] May collect data so that the rotation of the knot is easier. pv also has rotate function
# - [] pd code: 
#           if there are long (> 5 pixels) segments overlapping, find a different angle
#           i.e. all linestrings' intersections containing not just points
class NodalKnot:
    
    def __init__(self, 
                 k_to_zw_func,
                 zw_to_c_func,
                 kx_min=-np.pi, kx_max=np.pi,
                 ky_min=-np.pi, ky_max=np.pi,
                 kz_min=0,      kz_max=np.pi,
                 pts_per_dim=400):
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

        # Set grid parameters with defaults if not provided
        self.kx_min = kx_min; self.kx_max = kx_max
        self.ky_min = ky_min; self.ky_max = ky_max
        self.kz_min = kz_min; self.kz_max = kz_max
        self.pts_per_dim = pts_per_dim
        
        # Initialize the attributes
        self.val = None
        self.kx_grid = None
        self.ky_grid = None
        self.kz_grid = None
        self.binarized_val = None
        self.selected_points = None
        self.skeleton = None
        self.skeleton_points = None
        self.graph = None

    @property
    def span(self):
        """
        Calculate the span of the grid in each dimension.

        Returns:
        -------
        span : tuple
            The span of the grid as (kx_span, ky_span, kz_span).
        """
        return (self.kx_max - self.kx_min,
                self.ky_max - self.ky_min,
                self.kz_max - self.kz_min)

    @property
    def spacing(self):
        """
        Calculate the spacing between points in the grid.

        Returns:
        -------
        spacing : float
            The spacing between points in the grid.
        """
        return ((self.kx_max - self.kx_min) / self.pts_per_dim,
                (self.ky_max - self.ky_min) / self.pts_per_dim,
                (self.kz_max - self.kz_min) / self.pts_per_dim)
    
    @property
    def origin(self):
        """
        Calculate the origin of the grid.

        Returns:
        -------
        origin : tuple
            The origin of the grid as (kx_min, ky_min, kz_min).
        """
        return (self.kx_min, self.ky_min, self.kz_min)

    def generate_region(self,
            kx_min=None, kx_max=None,
            ky_min=None, ky_max=None,
            kz_min=None, kz_max=None,
            pts_per_dim=None
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
        kx_vals = np.linspace(kx_min if kx_min is not None else self.kx_min,
                               kx_max if kx_max is not None else self.kx_max,
                               pts_per_dim if pts_per_dim is not None else self.pts_per_dim)
        ky_vals = np.linspace(ky_min if ky_min is not None else self.ky_min,
                               ky_max if ky_max is not None else self.ky_max,
                               pts_per_dim if pts_per_dim is not None else self.pts_per_dim)
        kz_vals = np.linspace(kz_min if kz_min is not None else self.kz_min,
                               kz_max if kz_max is not None else self.kz_max,
                               pts_per_dim if pts_per_dim is not None else self.pts_per_dim)
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')

        z, w = self.k_to_zw_func(kx_grid, ky_grid, kz_grid)
        val = self.zw_to_c_func(z, w)

        self.kx_min = self.kx_min; self.kx_max = self.kx_max
        self.ky_min = self.ky_min; self.ky_max = self.ky_max
        self.kz_min = self.kz_min; self.kz_max = self.kz_max
        self.pts_per_dim = self.pts_per_dim
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
            points_mask=None,
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
        if isinstance(thickness, (int, float)):
            logging.info("Calculating according to following condition:\n"
                 "norm(f)-c where C=constant")
        elif isinstance(thickness, list):
            logging.info("Calculating according to following condition:\n"
                 "norm(f)^2+C+R*real(f)+I*Im(f) where C=constant, R=real_coef, I=imag_coef")
            
        # check if self.val is available
        if self.val is None or kwargs:
             self.generate_region(**kwargs)
        if points_mask is None:
            norm = np.abs(self.val)

            if isinstance(thickness, (int, float)):
                if epsilon is not None and epsilon > 0:
                # return the thickened knot's surface
                    points_mask = np.where(np.abs(norm - thickness) < epsilon)
                else:
                    # return the thickened knot as a solid (filled-up surface)
                    if thickness < 10/self.pts_per_dim: thickness = 10/self.pts_per_dim
                    points_mask = np.where(norm <= thickness)
            elif isinstance(thickness, list):
                # list must be [constant, real_coef, imag_coef]
                assert len(thickness) == 3, "thickness list must be [constant, real_coef, imag_coef]"
                constant, real_coef, imag_coef = thickness
                real_part = np.real(self.val)
                imag_part = np.imag(self.val)
                thickness = constant + real_coef * real_part + imag_coef * imag_part 
                if epsilon is not None and epsilon > 0:
                # return the thickened knot's surface
                    points_mask = np.where(np.abs(norm**2+thickness) < epsilon)
                else:
                    # return the thickened knot as a solid (filled-up surface)
                    if thickness < 10/self.pts_per_dim: thickness = 10/self.pts_per_dim
                    points_mask = np.where(norm**2 <= thickness)
            else:
                raise TypeError("Thickness must be either a number corresponding to i*c*pauli_y "\
                                "or a list of three numerics corresponding to [constant(i*c*pauli_y),"\
                                " real_coef, imag_coef} deformation ")
             
        else:
            if points_mask.shape != self.val.shape:
                raise ValueError("Input `val` must have the same shape as the generated region.")
            points_mask = np.where(points_mask)

        pts = np.array([self.kx_grid[points_mask],
                        self.ky_grid[points_mask],
                        self.kz_grid[points_mask]]).T
        
        if points_mask is None: self.selected_points = pts

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
            **kwargs
        ):
        """
        The list of points on the skeleton of the zero region.

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
        skeleton_points : np.ndarray
            The list of 3D coordinates on the skeleton of the zero region.
        """
        # First, generate the skeleton
        self.knot_skeleton(thickness, epsilon, **kwargs)
        # Extract the skeleton points based on the skeletonized volume
        points = self.knot_surface_points(points_mask=self.skeleton)
        self.skeleton_points = points
        return points

    def skeleton_graph(self, 
            skeleton_3d=None,
            clean=True,
            smooth_epsilon=0.,
            **generate_region_kwargs,
        ):
        """
        Convert a volume representation of skeleton points to a graph, and optionally remove leaf nodes.
        
        Parameters:
        -----------
        skeleton_3d : np.ndarray
            A 3D numpy array representing the volume. Voxels with value 1 are taken as skeleton points.
        clean : bool, optional
            If True, remove leaf nodes (nodes with degree 1) from the graph. Default is True.
        knot_skeleton_kwargs :
            Additional keyword arguments for `knot_skeleton`. Only used if `skeleton_3d` is None.
        
        Returns:
        --------
        G : nx.Graph
            The graph constructed from the volume (optionally cleaned).
        """
        # Generate the skeleton 3d array if not provided
        if skeleton_3d is None:
            if self.skeleton is None or generate_region_kwargs:
                self.knot_skeleton(**generate_region_kwargs)
            skeleton_3d = self.skeleton
        
        # Convert the 3d skeleton image to a graph using poly2graph
        G = skeleton2graph(skeleton_3d)
        
        if clean: 
            # Remove leaf nodes
            G = remove_leaf_nodes(G)
            # Remove degree-2 nodes that does not affect the topology
            G = remove_deg2_preserving_pts(G)
            # smooth the edge points
            G = smooth_edges(G, epsilon=smooth_epsilon, copy=False)
        
        self.graph = G
        return G
    
    @staticmethod
    def plot_3D(points, file_name=None):
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

    @staticmethod
    def plot_graph(G, file_name=None):
        """
        Plot the 3D knotted graph in 3D space.

        Parameters:
        ----------
        G : nx.Graph
            The graph to plot.
        file_name : str, optional
            File name to save the plot. If None, no saving is done.
            Default is None.

        Returns:
        -------
        fig : plotly.graph_objects.Figure
            The Plotly figure object for visualization.
        """
        fig = plot_3D_graph(G)

        if file_name:
            fig.write_html(file_name)
        
        return fig
        

    @staticmethod
    def print_graph_properties(G):

        data = []
        # Basic properties
        data.append(["Number of nodes", G.number_of_nodes()])
        data.append(["Number of edges", G.number_of_edges()])

        # Degree distribution
        degree_hist = nx.degree_histogram(G)
        degree_dist = [(deg, count) for deg, count in enumerate(degree_hist) if count > 0]

        # Connectivity and component info
        if nx.is_connected(G):
            connected = "Yes"
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
            data.append(["Connected", connected])
            data.append(["Diameter", diameter])
            data.append(["Avg shortest path", avg_path])
        else:
            connected = "No"
            num_components = nx.number_connected_components(G)
            data.append(["Connected", connected])
            data.append(["# Connected components", num_components])
            # Component sizes
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            for i, comp in enumerate(components, 1):
                data.append([f"Component {i} size", len(comp)])

        print(tabulate(data, headers=["Property", "Value"], tablefmt="github"))

        # Print degree distribution as a table
        if degree_dist:
            print("\nDegree distribution:")
            print(tabulate(degree_dist, headers=["Degree", "Frequency"], tablefmt="github"))


    @staticmethod         
    def check_minor(host_graph, minor_graph):
        """
        Check whether `minor_graph` is a minor of `host_graph` using minorminer.
        
        Parameters:
            host_graph (networkx.Graph): The graph in which to search for the minor.
            minor_graph (networkx.Graph): The graph to be checked as a minor.
            
        Returns:
            dict or None: The embedding mapping if `minor_graph` is a minor of `host_graph`,
                        otherwise None.
        """
        # Attempt to find an embedding of minor_graph in host_graph.
        embedding = minorminer.find_embedding(minor_graph, host_graph)
        
        if embedding:
            print("The given graph contains the minor graph.")
            return embedding
        else:
            print("The given graph does not contain the minor graph.")
            return None