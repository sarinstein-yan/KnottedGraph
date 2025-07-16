import numpy as np
import sympy as sp
import networkx as nx
import skimage.morphology as morph
from poly2graph import skeleton2graph
from functools import cached_property, lru_cache
from tabulate import tabulate
import minorminer
import logging

import pyvista as pv

from knotted_graph.util import (
    remove_leaf_nodes,
    simplify_edges,
    smooth_edges,
    total_edge_pts,
    is_PT_symmetric,
    is_trivalent,
    idx_to_coord,
)

from typing import Tuple, Union, Optional, Any, Dict, Sequence
from numpy.typing import NDArray, ArrayLike


# TODO:
# - Graph edges as shapely.LineString for planar analysis
# - [] Berry Curvature function and field plotted by pv's glyphs
# - [] pd code:
#           if there are long (> 5 pixels) segments overlapping, find a different angle
#           i.e. all linestrings' intersections containing not just points

class NodalSkeleton:

    pauli_x = sp.ImmutableDenseMatrix([[0, 1], [1, 0]])
    pauli_y = sp.ImmutableDenseMatrix([[0, -sp.I], [sp.I, 0]])
    pauli_z = sp.ImmutableDenseMatrix([[1, 0], [0, -1]])
    pauli_vec = (pauli_x, pauli_y, pauli_z)
    
    def __init__(
        self,
        char: Union[sp.Matrix, sp.ImmutableMatrix, Sequence[sp.Expr]],
        k_symbols: Tuple[sp.Symbol, sp.Symbol, sp.Symbol] = None,
        span: Tuple[Tuple[float, float],
                    Tuple[float, float],
                    Tuple[float, float]] = ((-np.pi, np.pi),
                                            (-np.pi, np.pi),
                                            (0,      np.pi)),
        dimension: int = 200,
        # dimension_enhancement: Optional[int] = 1,
        # ^ TODO: auto span detection
    ):
        # only support two-band Hamiltonian
        if isinstance(char, (sp.Matrix,sp.ImmutableMatrix)) and char.shape==(2,2):
            self.h_k = char
            self.bloch_vec = tuple(
                sp.simplify((char * s).trace()/2)
                for s in self.pauli_vec
            )
        elif isinstance(char, Sequence) and len(char) == 3:
            self.bloch_vec = tuple(c + sp.Integer(0) for c in char)
            self.h_k = sum((h*s for h,s in zip(char, self.pauli_vec)), 
                           start=sp.zeros(2, 2))
        else:
            raise ValueError("`char` must be a 2x2 sympy Matrix or a sequence "\
                             "of three coefficients for the Pauli matrices.")
            
        if k_symbols is None:
            self.k_symbols = sorted(self.h_k.free_symbols, key=lambda s: s.name)
            self.kx_symbol, self.ky_symbol, self.kz_symbol = self.k_symbols
        elif len(k_symbols) == 3:
            self.k_symbols = k_symbols
            self.kx_symbol, self.ky_symbol, self.kz_symbol = k_symbols
        else:
            raise ValueError("`k_symbols` must be a tuple of three sympy"\
                             " symbols (kx, ky, kz).")
        
        # check Hamiltonian properties
        self.is_Hermitian = sp.simplify(self.h_k - self.h_k.H) == sp.zeros(2, 2)
        # if self.is_Hermitian:
        #     raise ValueError("The Hamiltonian must be non-Hermitian.")
        self.is_PT_symmetric = is_PT_symmetric(self.h_k)
        
        # lambda functions of the bloch vector components
        self.bloch_vec_funcs = tuple(
            sp.lambdify(self.k_symbols, b, 'numpy')
            for b in self.bloch_vec
        )
        
        # plotting helpers
        self.span = np.asarray(span)
        self.dimension = dimension
        self.spacing = np.diff(self.span, axis=1).squeeze() / (dimension-1)
        self.origin = self.span[:, 0]

        # set k-space spans and coordinates
        self.kx_span, self.ky_span, self.kz_span = span
        for axis, (mn, mx) in zip(('x', 'y', 'z'), span):
            setattr(self, f'k{axis}_min', mn)
            setattr(self, f'k{axis}_max', mx)
            setattr(self, f'k{axis}_vals', np.linspace(mn, mx, dimension))
        
        self.kx_grid, self.ky_grid, self.kz_grid = np.meshgrid(
            self.kx_vals, self.ky_vals, self.kz_vals,
            indexing='ij'
        )

        # default cache for skeleton graph
        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None
        self._pv_data_args = None


    @cached_property
    def spectrum(self) -> NDArray:
        r"""The k-space spectrum ('upper'/'positive' band).

        Returns
        -------
        NDArray
            The spectrum of the upper band, calculated as:
            \[
            + \sqrt{\lvert \vec{d} \rvert^2}
            \]
            where \(\vec{d}\) represents the Bloch vector components.
            I.e., the half of the energy band gap.
        """
        d_grid = np.asarray([
            func(self.kx_grid, self.ky_grid, self.kz_grid).astype(np.complex128)
            if expr.free_symbols
            else np.full_like(self.kx_grid, expr, dtype=np.complex128)
            for (expr, func) in zip(self.bloch_vec, self.bloch_vec_funcs)
        ])
        return np.sqrt(np.sum(d_grid**2, axis=0))
    

    @cached_property
    def band_gap(self) -> NDArray:
        r"""The k-space band gap.

        Returns
        -------
        NDArray
            The band gap, calculated as:
            \[
            \Delta = 2 \lvert \vec{d} \rvert
            \]
        """
        return 2 * np.abs(self.spectrum)
    

    @property
    def _interior_mask(self) -> NDArray:
        """The filled interior of the exceptional surface as a binary mask.
        I.e. the region of pure imaginary energy."""
        # return self.spectrum.imag != 0
        return self.spectrum.real == 0
    

    @cached_property
    def _skeleton_image(self) -> NDArray:
        """The skeleton (medial axis) of the exceptional surface."""
        return morph.skeletonize(self._interior_mask, method='lee')
    

    @cached_property
    def skeleton_coords(self) -> NDArray:
        """The k-space coordinates of the exceptional surface skeleton points."""
        point_mask = np.where(self._skeleton_image)
        return np.asarray([self.kx_grid[point_mask],
                           self.ky_grid[point_mask],
                           self.kz_grid[point_mask]]).T
    

    def skeleton_graph(
        self, 
        simplify: bool = True,
        smooth_epsilon: int = 4,
        *,
        skeleton_image: Optional[NDArray] = None
    ) -> nx.MultiGraph:
        """The skeleton graph of the exceptional surface."""
        
        # Check if the arguments match the cached ones
        args = (smooth_epsilon, simplify, id(skeleton_image))
        if self.skeleton_graph_cache is not None and \
           self.skeleton_graph_cache_args == args:
            return self.skeleton_graph_cache
        
        # Compute the graph
        G = skeleton2graph(self._skeleton_image) \
            if skeleton_image is None else skeleton_image
        if simplify:
            G = remove_leaf_nodes(G)
            G = simplify_edges(G)
        G = smooth_edges(G, epsilon=smooth_epsilon, copy=False)
        G.graph['is_trivalent'] = is_trivalent(G)

        # cache the result
        self.skeleton_graph_cache = G
        self.skeleton_graph_cache_args = args
        return G
    

    @property
    def total_edge_pts(self) -> int:
        """The total number of edge points in the skeleton graph."""
        return total_edge_pts(self.skeleton_graph_cache or self.skeleton_graph())


    def clear_cache(self):
        """Clear the cached skeleton graph."""
        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None


    @cached_property
    def spectrum_volume_pv(self) -> pv.PolyData:
        """The real part of the spectrum as a PyVista PolyData object."""
        engy = self.spectrum
        volume = pv.ImageData(
            dimensions=engy.shape,
            spacing=self.spacing,
            origin=self.origin
        )
        volume.point_data['real'] = engy.real.ravel(order='F')
        volume.point_data['imag'] = engy.imag.ravel(order='F')
        volume.point_data['gap'] = self.band_gap.ravel(order='F')
        helper = np.abs(engy.real) - np.abs(engy.imag)
        volume.point_data['ES_helper'] = helper.ravel(order='F')
        return volume


    @cached_property
    def exceptional_surface_pv(self) -> pv.PolyData:
        """The exceptional surface ad a PyVista PolyData object."""
        return self.spectrum_volume_pv.contour(
            isosurfaces=[0.], scalars='ES_helper'
        )
    

    def plot_exceptional_surface(
        self,
        plotter: Optional[pv.Plotter] = None,
        surf_color: Any = "#12a47f",
        surf_opacity: float = 0.9,
        surf_decimation: float = 0.2,
        surf_kwargs: Dict = {},
        add_silhouettes: bool = False,
        silh_color: Any = 'gray',
        silh_opacity: float = 0.2,
        silh_origins: Optional[ArrayLike | str] = None,
        silh_decimation: float = 0.2,
        silh_kwargs: Dict = {},
    ) -> pv.Plotter:
        """Plot the exceptional surface."""
        if plotter is None:
            plotter = pv.Plotter()
        
        ES = self.exceptional_surface_pv

        if add_silhouettes:
            if isinstance(silh_origins, str):
                silh_origins = self.origin
            elif silh_origins is None:
                silh_origins = (None, None, None)

            silh_kwargs = {
                "color": silh_color,
                "opacity": silh_opacity,
                "silhouette": True,
                **silh_kwargs
            }

            for (n, o) in zip(np.eye(3), silh_origins):
                proj = ES.project_points_to_plane(normal=n, origin=o)
                proj = proj.decimate_pro(silh_decimation)
                plotter.add_mesh(proj, **silh_kwargs)
        
        ES_deci = ES.decimate_pro(
            surf_decimation, preserve_topology=True
        )
        surf_kwargs = {
            "color": surf_color,
            "opacity": surf_opacity,
            "smooth_shading": True,
            "specular": 0.5,
            "specular_power": 20,
            "metallic": 1.,
            **surf_kwargs
        }
        plotter.add_mesh(ES_deci, **surf_kwargs)

        return plotter
    

    def _idx_to_coord(self, indices: ArrayLike) -> NDArray:
        """Convert indices to coordinates in the k-space."""
        return idx_to_coord(indices, self.spacing, self.origin)


    def _pyvista_graph_data(
            self,
            node_radius: float = 0.08,
            tube_radius: float = 0.04,
        ) -> Tuple[pv.MultiBlock, pv.PolyData]:
        """Prepare the spatial graph data for plotting."""
        args = (tube_radius, node_radius)
        if self._pv_data_args == args:
            return self.node_glyphs_pv, self.edge_tubes_pv

        G = self.skeleton_graph_cache or self.skeleton_graph()

        nodes_pos = self._idx_to_coord(
            np.asarray([n['pos'] for n in G.nodes.values()])
        )
        node_data = pv.PolyData(nodes_pos)
        node_glyphs = node_data.glyph(
            orient=False,
            scale=False,
            geom=pv.Sphere(radius=node_radius),
        )

        edges_pts = [self._idx_to_coord(e['pts']) for e in G.edges.values()]
        edge_data = [pv.Spline(e, 2*len(e)) for e in edges_pts]
        edge_tubes = pv.MultiBlock()
        for e in edge_data:
            edge_tubes.append(e.tube(radius=tube_radius))
        
        self.node_data_pv = node_data
        self.node_glyphs_pv = node_glyphs
        self.edge_data_pv = edge_data
        self.edge_tubes_pv = edge_tubes
        # Update the cache key
        self._pv_data_args = args
        return node_glyphs, edge_tubes


    def plot_skeleton_graph(
            self,
            plotter: Optional[pv.Plotter] = None,
            add_nodes: bool = True,
            add_edges: bool = True,
            node_radius: float = 0.08,
            tube_radius: float = 0.04,
            node_color: Any = '#A60628',
            edge_color: Any = '#348ABD',
            node_kwargs: Dict = {},
            edge_kwargs: Dict = {},
            add_silhouettes: bool = False,
            silh_color: Any = 'gray',
            silh_origins: Optional[ArrayLike | str] = None,
            silh_kwargs: Dict = {},
        ) -> pv.Plotter:
        """Plot the skeleton graph."""
        if plotter is None:
            plotter = pv.Plotter()

        comm = {
            "opacity": 1.,
            "smooth_shading": True,
            "specular": 0.5,
            "specular_power": 20,
            "metallic": 1.,
        }
        
        node_glyphs, edge_tubes = \
            self._pyvista_graph_data(node_radius, tube_radius)
            
        if add_edges:
            edge_kwargs = {'color': edge_color, **comm, **edge_kwargs}
            plotter.add_mesh(edge_tubes, **edge_kwargs)

        if add_nodes:
            node_kwargs = {'color': node_color, **comm, **node_kwargs}
            plotter.add_mesh(node_glyphs, **node_kwargs)
        
        if add_silhouettes:
            if isinstance(silh_origins, str):
                silh_origins = self.origin
            elif silh_origins is None:
                silh_origins = (None, None, None)
            
            silh_kwargs = {
                'color': silh_color,
                # 'silhouette': True,
                **silh_kwargs
            }
            def _add_silhouette(poly, origins, **kwargs):
                for (n, o) in zip(np.eye(3), origins):
                    proj = poly.project_points_to_plane(normal=n, origin=o)
                    plotter.add_mesh(proj, **kwargs)
            
            for e in edge_tubes:
                _add_silhouette(e, silh_origins, opacity=.2, **silh_kwargs)
            
            _add_silhouette(node_glyphs, silh_origins, opacity=1., **silh_kwargs)
        
        return plotter


    def graph_summary(
            self, 
            G: Optional[nx.Graph | nx.MultiGraph] = None
        ) -> None:        
        if G is None: G = self.skeleton_graph_cache
        # Basic properties
        data = []
        data.append(["Number of nodes", G.number_of_nodes()])
        data.append(["Number of edges", G.number_of_edges()])
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
            components = sorted(nx.connected_components(G), 
                                key=len, 
                                reverse=True)
            for i, comp in enumerate(components, 1):
                data.append([f"Component {i} size", len(comp)])
        print(tabulate(data, 
                       headers=["Property", "Value"], 
                       tablefmt="github"))

        # Degree distribution
        degree_hist = nx.degree_histogram(G)
        degree_dist = [(deg, count) for deg, count in 
                       enumerate(degree_hist) if count > 0]
        if degree_dist:
            print("\nDegree distribution:")
            print(tabulate(degree_dist, 
                           headers=["Degree", "Frequency"], 
                           tablefmt="github"))


    @staticmethod
    def check_minor(
        host_graph: nx.MultiGraph | nx.Graph,
        minor_graph: nx.Graph
    ) -> Any:
        """Check whether `minor_graph` is a minor of `host_graph`.

        Parameters
        ----------
        host_graph : nx.MultiGraph | nx.Graph
            The graph in which to search for the minor.
        minor_graph : nx.Graph
            The graph to be checked as a minor.

        Returns
        -------
        Any
            The embedding mapping if `minor_graph` is a minor of `host_graph`,
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



if __name__ == "__main__":

    ### Example usage

    # Hamiltonian of a hoft-link metal
    kx, ky, kz = sp.symbols('k_x k_y k_z', real=True)
    z = sp.cos(2*kz) + sp.Rational(1, 2) \
        + sp.I*(sp.cos(kx) + sp.cos(ky) + sp.cos(kz) - 2)
    w = sp.sin(kx) + sp.I*sp.sin(ky)
    f = z**2 - w**2 
    cx = sp.simplify(sp.re(f))
    cz = sp.simplify(sp.im(f))


    nHerm = .2
    char = (cx, nHerm*sp.I, cz)
    # Create a NodalSkeleton instance
    ske = NodalSkeleton(char)


    # Check properties
    print(f"self.h_k: {ske.h_k}")
    print(f"self.bloch_vec: {ske.bloch_vec}")
    print(f"Is Hermitian: {ske.is_Hermitian}")
    print(f"Is PT-symmetric: {ske.is_PT_symmetric}")
    print(f"self.span: {ske.span}")
    print(f"self.dimension: {ske.dimension}")
    print(f"self.kx_span: {ske.kx_span}")
    print(f"self.ky_span: {ske.ky_span}")
    print(f"self.kz_span: {ske.kz_span}")
    print(f"self.kx_min: {ske.kx_min}")
    print(f"self.kx_max: {ske.kx_max}")
    print(f"self.ky_min: {ske.ky_min}")
    print(f"self.ky_max: {ske.ky_max}")
    print(f"self.kz_min: {ske.kz_min}")
    print(f"self.kz_max: {ske.kz_max}")
    print(f"self.spacing: {ske.spacing}")
    print(f"self.origin: {ske.origin}")
    print(f"self.spectrum: {ske.spectrum.shape} {ske.spectrum.dtype}")
    print(f"self._skeleton_image: {ske._skeleton_image.shape} {ske._skeleton_image.dtype}")
    print(f"Total edge points: {ske.total_edge_pts}")
    ske.graph_summary()
    print(f"Check minor: {ske.check_minor(
        ske.skeleton_graph_cache, 
        ske.skeleton_graph_cache.subgraph([0])
    )}")

    # Check plotting
    pl = pv.Plotter(shape=(1, 2), window_size=(1200, 500), off_screen=True)
    pl.link_views()
    pl.subplot(0, 0)
    ske.plot_exceptional_surface(plotter=pl)
    pl.subplot(0, 1)
    ske.plot_skeleton_graph(plotter=pl)
    pl.close()
    print("Plotting done. Use `pl.show()` to display the plot in a notebook.")