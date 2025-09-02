import numpy as np
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
import skimage.morphology as morph
from poly2graph import skeleton2graph
from functools import cached_property, lru_cache
from tabulate import tabulate
import minorminer
import logging

import pyvista as pv

from knotted_graph.util import (
    compute_yamada_safely,
    remove_leaf_nodes,
    simplify_edges,
    smooth_edges,
    get_all_edge_pts,
    total_edge_pts,
    is_PT_symmetric,
    is_trivalent,
    idx_to_coord,
)
from knotted_graph.yamada import PDCode, compute_yamada_polynomial

from typing import Tuple, Union, Optional, Any, Dict, Sequence
from numpy.typing import NDArray, ArrayLike


# TODO:
# - [] Orthogonal slices of the spectrum.imag + edge_points
# - pd code: 
    # - [] from_graph_to_pd_code, from_pd_code_to_yamada, from_graph_to_yamada, 
    # - [] self.pd_code, self.yamada, self.graph_diagram_pv (parallel projection, and if possible color the undercrossing in transparent)


class NodalSkeleton:
    r"""
    Analyzes and visualizes the nodal structures of 2-band non-Hermitian Hamiltonians.

    This class computes the exceptional surface, its skeleton (medial axis),
    and represents the skeleton as a spatial graph. It provides tools for
    analyzing the topology of this graph and visualizing both the surface and
    the graph in 3D k-space.

    Parameters
    ----------
    char : Union[sp.Matrix, sp.ImmutableMatrix, Sequence[sp.Expr]]
        The characterization of the Hamiltonian. This can be a 2x2 SymPy matrix
        representing the Hamiltonian $H(k)$, or a sequence of three SymPy
        expressions representing the components of the Bloch vector $\vec{d}(k)$
        such that $H(k) = \vec{d}(k) \cdot \vec{\sigma}$, where $\vec{\sigma}$
        are the Pauli matrices.
    k_symbols : Tuple[sp.Symbol, sp.Symbol, sp.Symbol], optional
        A tuple of three SymPy symbols for the momentum components (kx, ky, kz).
        If None, they are inferred from the free symbols in `char`.
        Defaults to None.
    span : Tuple[(float, float),
                 (float, float),
                 (float, float)], optional
        The plotting range for (kx, ky, kz) as ((kx_min, kx_max),
        (ky_min, ky_max), (kz_min, kz_max)). Defaults to
        ((-np.pi, np.pi), (-np.pi, np.pi), (0, np.pi)).
    dimension : int, optional
        The number of points to use for each dimension of the k-space grid.
        Defaults to 200.

    Attributes
    ----------
    h_k : sp.Matrix
        The 2x2 SymPy matrix of the Hamiltonian.
    bloch_vec : tuple[sp.Expr, sp.Expr, sp.Expr]
        The components of the Bloch vector (dx, dy, dz).
    spectrum_expr : sp.Expr
        The expression for the k-space spectrum, defined as
        \( + \sqrt{ \vec{d} \cdot \vec{d} } \).
    band_gap_expr : sp.Expr
        The expression for the k-space band gap, defined as
        \( \Delta = 2 \lVert \vec{d} \rVert \).
    dispersion_expr : tuple[sp.Expr, sp.Expr, sp.Expr]
        The expressions for the k-space dispersion relation
        \( \frac{dE}{dk} \) for each component (dx, dy, dz).
    berry_curvature_expr : tuple[sp.Expr, sp.Expr, sp.Expr]
        The expressions for the Berry curvature vector field
    k_symbols : tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        The symbols for the momentum components (kx, ky, kz).
    span : np.ndarray
        The plotting range for (kx, ky, kz).
    dimension : int
        The resolution of the k-space grid.
    kx_grid, ky_grid, kz_grid : np.ndarray
        The meshgrid arrays for the k-space coordinates.
    kx_span, ky_span, kz_span : tuple[float, float]
        The spans for kx, ky, and kz.
    kx_vals, ky_vals, kz_vals : np.ndarray
        The values for kx, ky, and kz at the grid points.
    is_Hermitian : bool
        Whether the Hamiltonian is Hermitian.
    is_PT_symmetric : bool
        Whether the Hamiltonian is PT-symmetric.
    bloch_vec_funcs : tuple[callable, callable, callable]
        The lambdified functions for the Bloch vector components.
    spectrum : (N, N, N) complex
        The k-space spectrum (upper band) calculated from the Bloch vector.
    band_gap : (N, N, N) float
        The k-space band gap, defined as \( \Delta = 2 \lVert \vec{d} \rVert \).
    dispersion : (N, N, N, 3) complex
        The k-space dispersion relation \( \frac{dE}{dk} \).
    berry_curvature : (N, N, N, 3) float
        The Berry curvature vector field on the k-space grid.
    interior_mask : (N, N, N) bool
        A binary mask of the filled interior of the exceptional surface,
        where the energy is purely imaginary.
    skeleton_coords : (M, 3) float
        The k-space coordinates of the points on the skeleton of the
        exceptional surface, where M is the number of skeleton points.
    skeleton_graph_cache : Optional[nx.MultiGraph]
        Cached graph representation of the skeleton of the exceptional surface.
    total_edge_pts : int
        The total number of points constituting the edges of the skeleton graph.
    fields_pv : pv.ImageData
        A PyVista grid object containing the k-space field data
    """

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
    def _bloch_vec_grid(self) -> np.ndarray:
        """The Bloch vector components evaluated on the k-space grid."""
        k_grids = (self.kx_grid, self.ky_grid, self.kz_grid)
        return np.asarray([
            func(*k_grids).astype(np.complex128)
            if expr.free_symbols
            else np.full_like(self.kx_grid, complex(expr), dtype=np.complex128)
            for expr, func in zip(self.bloch_vec, self.bloch_vec_funcs)
        ])

    @cached_property
    def spectrum(self) -> NDArray:
        r"""The k-space spectrum ('upper'/'positive' band).

        Returns
        -------
        NDArray
            The spectrum of the upper band, calculated as:
            \[
            + \sqrt{ \vec{d} \cdot \vec{d} }
            \]
            where \(\vec{d}\) represents the Bloch vector components.
            I.e., the half of the energy band gap.
        """
        return np.sqrt(np.sum(self._bloch_vec_grid**2, axis=0))


    @cached_property
    def band_gap(self) -> NDArray:
        r"""The k-space band gap \( \Delta = 2 \lVert \vec{d} \rVert \).

        Returns
        -------
        NDArray
            The band gap, calculated as:
            \[
            \Delta = 2 \lVert \vec{d} \rVert
            \]
        """
        return 2 * np.abs(self.spectrum)


    @cached_property
    def dispersion(self) -> NDArray:
        r"""The k-space dispersion relation \( \frac{dE}{dk} \).

        Returns
        -------
        NDArray
            The dispersion relation, \( \frac{dE}{dk} \)
        """
        grad = np.gradient(self.spectrum, *self.spacing)
        return np.stack(grad, axis=-1)


    @staticmethod
    def check_berry_curvature_prerequisites(
        bloch_vec: Sequence[sp.Expr],
    ) -> Dict[str, Any]:
        """Helper to check validity and get indices for Berry curvature."""
        real_indices = []
        imag_info = {}
        for i, d_i in enumerate(bloch_vec):
            if d_i.is_real:
                real_indices.append(i)
            elif d_i.is_imaginary and not d_i.free_symbols:
                if 'idx' in imag_info: return {'valid': False}
                imag_info = {'idx': i, 'gamma': sp.im(d_i)}

        if len(real_indices) == 2 and 'idx' in imag_info:
            return {
                'valid': True,
                'gamma': imag_info['gamma'],
                'gamma_idx': imag_info['idx'],
                'd1_idx': real_indices[0],
                'd2_idx': real_indices[1],
            }
        return {'valid': False}

    @cached_property
    def _berry_prerequisites(self) -> Dict[str, Any]:
        """Checks prerequisites for Berry curvature calculation."""
        return NodalSkeleton.check_berry_curvature_prerequisites(self.bloch_vec)

    @cached_property
    def berry_curvature(self) -> NDArray:
        """The Berry curvature vector field on the k-space grid."""
        prereqs = self._berry_prerequisites
        if not prereqs['valid']:
            raise NotImplementedError("Berry curvature is only defined for Bloch vectors "
                                      "with two real and one imaginary component.")

        gamma = float(prereqs['gamma'])
        d1_grid = self._bloch_vec_grid[prereqs['d1_idx']].real
        d2_grid = self._bloch_vec_grid[prereqs['d2_idx']].real
        p, q, r = prereqs['d1_idx'], prereqs['d2_idx'], prereqs['gamma_idx']
        perm_sign = int(sp.LeviCivita(p, q, r))

        grad_d1 = np.gradient(d1_grid, *self.spacing)
        grad_d2 = np.gradient(d2_grid, *self.spacing)

        curl = np.stack([
            grad_d2[1]*grad_d1[2] - grad_d2[2]*grad_d1[1],
            grad_d2[2]*grad_d1[0] - grad_d2[0]*grad_d1[2],
            grad_d2[0]*grad_d1[1] - grad_d2[1]*grad_d1[0]
        ], axis=0)

        eps_sq = gamma**2 - d1_grid**2 - d2_grid**2
        # Mask for regions outside ES or on the singularity
        PT_mask = eps_sq <= 0

        # Use np.divide to handle division by zero gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = 2 * eps_sq**1.5
            prefactor = gamma / denominator

        F = np.zeros_like(curl)
        F[:, ~PT_mask] = perm_sign * prefactor[None, ~PT_mask] * curl[:, ~PT_mask]
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        return np.stack(F, axis=-1)


    @property
    def _interior_mask(self) -> NDArray:
        """The filled interior of the exceptional surface as a binary mask.
        I.e. the region of *pure* imaginary energy."""
        # return self.spectrum.imag != 0
        return self.spectrum.real == 0


    @cached_property
    def _skeleton_image(self) -> NDArray:
        """A binary image of the skeleton (medial axis) of the exceptional
        surface."""
        image = morph.skeletonize(self._interior_mask, method='lee')
        if np.sum(image) == 0:
            raise ValueError(
                "The skeleton image is empty. "
                "Ensure the Hamiltonian has a non-empty exceptional surface."
            )
        return image

    @cached_property
    def skeleton_coords(self) -> NDArray:
        """
        The k-space coordinates of the points on the skeleton.

        Returns
        -------
        NDArray
            An array of shape (N, 3) where N is the number of points in the
            skeleton, and each row is the (kx, ky, kz) coordinate of a point.
        """
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
        """
        Computes the graph representation of the exceptional surface's skeleton.

        This method converts the skeleton image into a `networkx.MultiGraph`, where
        nodes represent endpoints or junctions and edges represent the paths
        connecting them. The graph can be simplified and smoothed.

        Parameters
        ----------
        simplify : bool, optional
            If True, removes small leaf nodes (pruning) and simplifies edges
            by removing redundant intermediate points. Defaults to True.
        smooth_epsilon : int, optional
            The tolerance for edge smoothing. A larger value results in
            smoother, less detailed edge paths. Defaults to 4.
        skeleton_image : Optional[NDArray], optional
            An external skeleton image to use instead of the one computed
            internally. If None, the internally computed skeleton is used.
            Defaults to None.

        Returns
        -------
        nx.MultiGraph
            The graph representation of the skeleton. Node attributes include
            'pos' (index coordinates), and edge attributes include 'pts'
            (a list of index coordinates along the edge).
        """
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
        self.is_graph_trivalent = G.graph['is_trivalent']

        # cache the result
        self.skeleton_graph_cache = G
        self.skeleton_graph_cache_args = args
        return G


    @property
    def total_edge_pts(self) -> int:
        """
        The total number of points constituting the edges of the skeleton graph.

        Returns
        -------
        int
            The sum of the number of points in all edges of the cached
            skeleton graph.
        """
        return total_edge_pts(self.skeleton_graph_cache or self.skeleton_graph())


    def clear_cache(self):
        """Clears the cached skeleton graph and related PyVista data."""
        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None
        self._pv_data_args = None


    @cached_property
    def fields_pv(self) -> pv.PolyData:
        """
        A PyVista grid object containing the k-space field data.

        This grid stores the real and imaginary parts of the energy spectrum,
        the band gap, and a helper scalar field for isosurfacing the
        exceptional surface.

        Returns
        -------
        pv.PolyData
            A PyVista `ImageData` object with point data for 'real', 'imag',
            'gap', and 'ES_helper'.
        """
        engy = self.spectrum
        vol = pv.ImageData(
            dimensions=engy.shape,
            spacing=self.spacing,
            origin=self.origin
        )

        # scaler fields
        vol.point_data['imag'] = engy.imag.ravel(order='F')
        vol.point_data['real'] = engy.real.ravel(order='F')
        vol.point_data['gap'] = self.band_gap.ravel(order='F')
        helper = np.abs(engy.real) - np.abs(engy.imag)
        vol.point_data['ES_helper'] = helper.ravel(order='F')

        # vector fields
        im_disp = np.stack(
            np.gradient(engy.imag, *self.spacing, edge_order=2), axis=-1
        )
        im_disp[~self._interior_mask] = 0.
        im_disp = im_disp.reshape(-1, 3, order='F')
        im_disp_norm = np.linalg.norm(im_disp, axis=-1)
        vol.point_data['im_disp'] = im_disp
        vol.point_data['|im_disp|'] = im_disp_norm
        vol.point_data['log10(|im_disp|+1)'] = np.log10(im_disp_norm + 1)

        berry = self.berry_curvature.copy()
        berry = berry.reshape(-1, 3, order='F')
        berry_norm = np.linalg.norm(berry, axis=-1)
        vol.point_data['berry'] = berry
        vol.point_data['|berry|'] = berry_norm
        vol.point_data['log10(|berry|+1)'] = np.log10(berry_norm + 1)

        return vol


    @cached_property
    def exceptional_surface_pv(self) -> pv.PolyData:
        """
        The exceptional surface as a PyVista mesh object.

        This is computed by taking an isosurface of the spectrum volume where
        the real and imaginary parts of the energy are equal.

        Returns
        -------
        pv.PolyData
            A PyVista mesh (`PolyData`) representing the exceptional surface.
        """
        return self.fields_pv.contour(
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
        """
        Plots the exceptional surface in 3D k-space using PyVista.

        Parameters
        ----------
        plotter : pv.Plotter, optional
            An existing PyVista plotter object to add the mesh to. If None, a new
            plotter is created. Defaults to None.
        surf_color : Any, optional
            Color of the surface. Defaults to "#12a47f".
        surf_opacity : float, optional
            Opacity of the surface. Defaults to 0.9.
        surf_decimation : float, optional
            Fraction of polygons to keep for the surface mesh to improve
            rendering performance. Value between 0.0 and 1.0. Defaults to 0.2.
        surf_kwargs : Dict, optional
            Additional keyword arguments passed to `plotter.add_mesh` for the surface.
        add_silhouettes : bool, optional
            If True, adds 2D projections (silhouettes) of the surface onto the
            boundary planes. Defaults to False.
        silh_color : Any, optional
            Color of the silhouettes. Defaults to 'gray'.
        silh_opacity : float, optional
            Opacity of the silhouettes. Defaults to 0.2.
        silh_origins : Optional[ArrayLike | str], optional
            Origins for the projection planes. Can be a list of three origins,
            'auto' to use the k-space origin, or None. Defaults to None.
        silh_decimation : float, optional
            Decimation factor for the silhouette meshes. Defaults to 0.2.
        silh_kwargs : Dict, optional
            Additional keyword arguments for the silhouette meshes.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object with the surface mesh added.
        """
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
            "label": "Exceptional Surface",
            "opacity": surf_opacity,
            "smooth_shading": True,
            "specular": 0.5,
            "specular_power": 20,
            "metallic": 1.,
            **surf_kwargs
        }
        plotter.add_mesh(ES_deci, name='exceptional_surface', **surf_kwargs)

        return plotter


    def _idx_to_coord(self, indices: ArrayLike) -> NDArray:
        """Converts grid indices to k-space coordinates."""
        return idx_to_coord(indices, self.spacing, self.origin)


    def _pyvista_graph_data(
            self,
            node_radius: float = 0.08,
            tube_radius: float = 0.04,
            add_edge_field: bool = False,
            *,
            orient: str | bool = None,
            scale: str | bool = None,
            glyph_factor: float = 0.1,
            glyph_tolerance: float = 0.01,
            glyph_interval: int = 1,
            glyph_geom: Optional[pv.PolyData] = None,
    ) -> Tuple[pv.MultiBlock, pv.PolyData, Optional[pv.PolyData]]:
        """
        Prepares the spatial graph and optional edge vector field as PyVista objects.

        This internal method generates the visual components of the skeleton graph:
        spheres for nodes, tubes for edges, and optionally glyphs for a vector
        field sampled along the edges.

        Parameters
        ----------
        node_radius : float
            Radius of the spheres for nodes.
        tube_radius : float
            Radius of the tubes for edges.
        add_edge_field : bool
            If True, computes glyphs for the edge vector field.
            Defaults to False.
        orient : str | bool, optional
            Name of the vector field to orient glyphs on edges. Defaults to None.
        scale : str | bool, optional
            Name of the scalar/vector field to scale glyphs on edges. Defaults to None.
        glyph_factor : float, optional
            Scaling factor for the edge glyphs. Defaults to 0.1.
        glyph_tolerance : float, optional
            Tolerance for the glyphs. Defaults to 0.01.
        glyph_interval : int, optional
            The sampling stride for placing glyphs on edge points. A smaller
            number means denser glyphs. Defaults to 1.
        glyph_geom : Optional[pv.PolyData], optional
            The geometry to use for the glyphs (e.g., `pv.Arrow()`).

        Returns
        -------
        Tuple[pv.MultiBlock, pv.PolyData, Optional[pv.PolyData]]
            A tuple containing:
            - node_glyphs: PyVista object for the graph nodes.
            - edge_tubes: PyVista object for the graph edges.
            - edge_field_glyphs: PyVista object for the edge vector field, or
              None if not computed.
        """
        args = (node_radius, tube_radius, add_edge_field, orient, scale, 
                glyph_interval, glyph_factor, id(glyph_geom))

        if self._pv_data_args == args:
            return (self.node_glyphs_pv, self.edge_tubes_pv, 
                    getattr(self, 'edge_field_glyphs_pv', None))

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
        edge_tubes = pv.MultiBlock([
            e.tube(radius=tube_radius) for e in edge_data
        ])

        # Generate edge field glyphs if requested
        edge_field_glyphs = None
        if add_edge_field:
            all_edge_points = []
            for spline in edge_data:
                points = spline.points
                if len(points) > 0:
                    # Sample points from the spline using the interval stride
                    all_edge_points.append(points[::glyph_interval])

            if all_edge_points:
                sampled_points = np.vstack(all_edge_points)
                sampled_poly = pv.PolyData(sampled_points)
                # sampled_poly = sampled_poly.interpolate(self.fields_pv)
                sampled_poly = sampled_poly.sample(self.fields_pv)
                if orient is False and glyph_geom is None:
                    glyph_geom = pv.Sphere()
                edge_field_glyphs = sampled_poly.glyph(
                    orient=orient,
                    scale=scale,
                    factor=glyph_factor,
                    tolerance=glyph_tolerance,
                    geom=glyph_geom,
                )
        
        self.node_data_pv = node_data
        self.node_glyphs_pv = node_glyphs
        self.edge_data_pv = edge_data
        self.edge_tubes_pv = edge_tubes
        self.edge_field_glyphs_pv = edge_field_glyphs
        self._pv_data_args = args # Update the cache key

        return node_glyphs, edge_tubes, edge_field_glyphs


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
            # --- Field plotting args ---
            add_edge_field: bool = False,
            orient: str | bool = None,
            scale: str | bool = None,
            glyph_factor: float = 0.1,
            glyph_tolerance: float = 0.01,
            glyph_interval: int = 1,
            glyph_geom: Optional[pv.PolyData] = None,
            field_cmap: str = 'viridis',
            field_kwargs: Dict = {},
            # --- Silhouette args ---
            add_silhouettes: bool = False,
            silh_color: Any = 'gray',
            silh_origins: Optional[ArrayLike | str] = None,
            silh_kwargs: Dict = {},
        ) -> pv.Plotter:
        """
        Plots the skeleton graph in 3D k-space using PyVista.

        Nodes are represented as spheres and edges as tubes.

        Parameters
        ----------
        plotter : pv.Plotter, optional
            An existing PyVista plotter. If None, a new one is created.
        add_nodes : bool, optional
            Whether to plot the graph nodes. Defaults to True.
        add_edges : bool, optional
            Whether to plot the graph edges. Defaults to True.
        node_radius : float, optional
            Radius of the spheres representing nodes. Defaults to 0.08.
        tube_radius : float, optional
            Radius of the tubes representing edges. Defaults to 0.04.
        node_color : Any, optional
            Color of the nodes. Defaults to '#A60628'.
        edge_color : Any, optional
            Color of the edges. Defaults to '#348ABD'.
        node_kwargs : Dict, optional
            Additional keyword arguments for plotting nodes.
        edge_kwargs : Dict, optional
            Additional keyword arguments for plotting edges.
        add_silhouettes : bool, optional
            If True, adds 2D projections of the graph. Defaults to False.
        silh_color : Any, optional
            Color of the silhouettes. Defaults to 'gray'.
        silh_origins : Optional[ArrayLike | str], optional
            Origins for the projection planes. Defaults to None.
        silh_kwargs : Dict, optional
            Additional keyword arguments for the silhouette meshes.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object with the graph added.
        """
        if plotter is None:
            plotter = pv.Plotter()

        node_glyphs, edge_tubes, edge_field_glyphs = self._pyvista_graph_data(
            node_radius, tube_radius, add_edge_field,
            orient=orient, scale=scale,
            glyph_factor=glyph_factor,
            glyph_tolerance=glyph_tolerance,
            glyph_interval=glyph_interval,
            glyph_geom=glyph_geom,
        )

        comm = {
            "opacity": 1.,
            "smooth_shading": True,
            "specular": 0.5,
            "specular_power": 20,
            "metallic": 1.,
        }

        if add_edge_field and edge_field_glyphs:
            edge_kwargs = {'opacity': .1, **edge_kwargs}
            field_kwargs = {
                'scalars': scale,
                'cmap': field_cmap,
                'name': 'edge_field',
                'label': f"Edge Field ({str(scale or orient)})",
                'show_scalar_bar': bool(scale), 
                **field_kwargs
            }
            plotter.add_mesh(edge_field_glyphs, **field_kwargs)
        
        if add_edges:
            edge_kwargs = {'color': edge_color,
                           'name': 'edge',
                           'label': 'Graph Edge',
                           **comm, **edge_kwargs}
            plotter.add_mesh(edge_tubes, **edge_kwargs)

        if add_nodes:
            node_kwargs = {'color': node_color,
                           'name': 'node',
                           'label': 'Graph Node',
                           **comm, **node_kwargs}
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

            if add_edges:
                for e in edge_tubes:
                    _add_silhouette(e, silh_origins, opacity=.2, **silh_kwargs)
            if add_nodes:
                _add_silhouette(node_glyphs, silh_origins, opacity=1., **silh_kwargs)

        return plotter


    def plot_vector_field(
            self,
            plotter: Optional[pv.Plotter] = None,
            orient: str | bool = None,
            scale: str | bool = None,
            cmap: str = 'coolwarm',
            glyph_factor: float = 0.1,
            glyph_tolerance: float = 0.01,
            glyph_geom: Optional[pv.PolyData] = None,
            glyph_kwargs: Dict = {},
            orient_data: Optional[NDArray] = None,
            scale_data: Optional[NDArray] = None,
            show_surf: bool = True,
            surf_color: Any = 'gray',
            surf_opacity: float = 0.05,
            surf_decimation: float = 0.2,
            surf_kwargs: Dict = {},
        ) -> pv.Plotter:
        """
        Plots a vector field within the exceptional surface using glyphs.

        This is a general-purpose method for visualizing vector data (e.g.,
        Berry curvature, energy dispersion) at points inside the exceptional
        surface. Glyphs (like arrows or spheres) are used to represent the
        vector at each point.

        Parameters
        ----------
        plotter : pv.Plotter, optional
            An existing PyVista plotter. If None, a new one is created.
        orient : str | bool, optional
            The name of the vector field in `fields_pv.point_data` to use for
            orienting the glyphs. If False, glyphs are not oriented.
            Defaults to None.
        scale : str | bool, optional
            The name of the scalar or vector field in `fields_pv.point_data` to
            use for scaling the glyphs. If a vector field is provided, its
            magnitude is used. If False, glyphs are not scaled. Defaults to None.
        cmap : str, optional
            The colormap for the glyphs. Defaults to 'coolwarm'.
        glyph_factor : float, optional
            A scaling factor for the glyphs. Defaults to 0.1.
        glyph_tolerance : float, optional
            Controls the density of the glyphs. A smaller value means more
            glyphs. Defaults to 0.01.
        glyph_geom : pv.PolyData, optional
            The geometry to use for the glyphs (e.g., `pv.Arrow()`). If None,
            the default PyVista glyph is used.
        glyph_kwargs : Dict, optional
            Additional keyword arguments passed to `plotter.add_mesh` for the
            glyphs.
        orient_data : NDArray, optional
            Custom array of vectors for glyph orientation. If provided, it's added
            to the plotter data with the name specified by `orient`.
        scale_data : NDArray, optional
            Custom array of scalars for glyph scaling. If provided, it's added
            to the plotter data with the name specified by `scale`.
        show_surf : bool, optional
            If True, the exceptional surface is plotted as a translucent
            background. Defaults to True.
        surf_color : Any, optional
            Color of the exceptional surface. Defaults to 'gray'.
        surf_opacity : float, optional
            Opacity of the exceptional surface. Defaults to 0.05.
        surf_decimation : float, optional
            Decimation factor for the surface mesh to improve performance.
            Defaults to 0.2.
        surf_kwargs : Dict, optional
            Additional keyword arguments for the surface mesh.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object with the vector field added.
        """
        if plotter is None:
            plotter = pv.Plotter()

        vol = self.fields_pv.copy()
        mask = np.where(vol.point_data['imag'] != 0)[0]
        if orient_data:
            if not isinstance(orient, str): orient = 'orient'
            vol.point_data[orient] = orient_data
        if scale_data:
            if not isinstance(scale, str): scale = 'scale'
            vol.point_data[scale] = scale_data

        interior = vol.extract_points(mask)
        glyph = interior.glyph(
            orient=orient,
            scale=scale,
            factor=glyph_factor,
            tolerance=glyph_tolerance,
            geom=glyph_geom
        )

        glyph_kwargs = {"cmap": cmap,
                        "show_scalar_bar": True,
                        **glyph_kwargs}
        plotter.add_mesh(glyph, name='field', **glyph_kwargs)

        if show_surf:
            ES = self.exceptional_surface_pv
            ES_deci = ES.decimate_pro(
                surf_decimation, preserve_topology=True
            )
            surf_kwargs = {
                "color": surf_color,
                "opacity": surf_opacity,
                "label": "Exceptional Surface",
                **surf_kwargs
            }
            plotter.add_mesh(ES_deci, name='exceptional_surface', **surf_kwargs)

        return plotter


    def plot_scalar_field(
            self,
            plotter: Optional[pv.Plotter] = None,
            scale: str | bool = None,
            cmap: str = 'coolwarm',
            glyph_factor: float = 0.01,
            glyph_tolerance: float = 0.01,
            glyph_geom: Optional[pv.PolyData] = pv.Sphere(),
            glyph_kwargs: Dict = {},
            scale_data: Optional[NDArray] = None,
            show_surf: bool = True,
            surf_color: Any = 'gray',
            surf_opacity: float = 0.05,
            surf_decimation: float = 0.2,
            surf_kwargs: Dict = {},
        ) -> pv.Plotter:
        """
        Plots a scalar field within the exceptional surface using glyphs.

        This method visualizes a scalar field by scaling glyphs (e.g., spheres)
        at points inside the exceptional surface. It's a convenience wrapper
        around `plot_vector_field` with orientation disabled.

        Parameters
        ----------
        plotter : pv.Plotter, optional
            An existing PyVista plotter. If None, a new one is created.
        scale : str | bool, optional
            The name of the scalar field in `fields_pv.point_data` to use for
            scaling the glyphs. Defaults to None.
        cmap : str, optional
            The colormap for the glyphs. Defaults to 'coolwarm'.
        glyph_factor : float, optional
            A scaling factor for the glyphs. Defaults to 0.01.
        glyph_tolerance : float, optional
            Controls the density of the glyphs. Defaults to 0.01.
        glyph_geom : pv.PolyData, optional
            The geometry to use for the glyphs. Defaults to `pv.Sphere()`.
        glyph_kwargs : Dict, optional
            Additional keyword arguments for the glyphs.
        scale_data : NDArray, optional
            Custom array of scalars for glyph scaling.
        show_surf : bool, optional
            If True, displays the translucent exceptional surface. Defaults to True.
        surf_color : Any, optional
            Color of the exceptional surface. Defaults to 'gray'.
        surf_opacity : float, optional
            Opacity of the exceptional surface. Defaults to 0.05.
        surf_decimation : float, optional
            Decimation factor for the surface mesh. Defaults to 0.2.
        surf_kwargs : Dict, optional
            Additional keyword arguments for the surface mesh.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object with the scalar field added.
        """
        return self.plot_vector_field(
            plotter=plotter,orient=False,scale=scale,scale_data=scale_data,
            cmap=cmap,glyph_factor=glyph_factor,glyph_tolerance=glyph_tolerance,
            glyph_geom=glyph_geom,glyph_kwargs=glyph_kwargs,show_surf=show_surf,
            surf_color=surf_color,surf_opacity=surf_opacity,
            surf_decimation=surf_decimation,surf_kwargs=surf_kwargs,
        )


    def plot_berry_curvature(
            self,
            plotter: Optional[pv.Plotter] = None,
            cmap: str = 'coolwarm',
            glyph_factor: float = 0.1,
            glyph_tolerance: float = 0.01,
            glyph_geom: Optional[pv.PolyData] = None,
            glyph_kwargs: Dict = {},
            show_surf: bool = True,
            surf_color: Any = 'gray',
            surf_opacity: float = 0.05,
            surf_decimation: float = 0.2,
            surf_kwargs: Dict = {},
        ) -> pv.Plotter:
        """
        Plots the Berry curvature vector field in 3D k-space.

        This is a specialized plotting method that visualizes the Berry
        curvature inside the exceptional surface. It calls `plot_vector_field`
        with appropriate defaults for orientation ('berry') and scaling
        ('log10(|berry|+1)').

        Parameters
        ----------
        plotter : pv.Plotter, optional
            An existing PyVista plotter. If None, a new one is created.
        cmap : str, optional
            The colormap for the glyphs. Defaults to 'coolwarm'.
        glyph_factor : float, optional
            A scaling factor for the glyphs. Defaults to 0.1.
        glyph_tolerance : float, optional
            Controls the density of the glyphs. Defaults to 0.025.
        glyph_geom : pv.PolyData, optional
            The geometry to use for the glyphs. If None, PyVista's default is used.
        glyph_kwargs : Dict, optional
            Additional keyword arguments for the glyphs.
        show_surf : bool, optional
            If True, displays the translucent exceptional surface. Defaults to True.
        surf_color : Any, optional
            Color of the exceptional surface. Defaults to 'gray'.
        surf_opacity : float, optional
            Opacity of the exceptional surface. Defaults to 0.05.
        surf_decimation : float, optional
            Decimation factor for the surface mesh. Defaults to 0.2.
        surf_kwargs : Dict, optional
            Additional keyword arguments for the surface mesh.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object with the Berry curvature field added.
        """
        return self.plot_vector_field(
            plotter=plotter,orient='berry',scale='log10(|berry|+1)',
            cmap=cmap,glyph_factor=glyph_factor,glyph_tolerance=glyph_tolerance,
            glyph_geom=glyph_geom,glyph_kwargs=glyph_kwargs,show_surf=show_surf,
            surf_color=surf_color,surf_opacity=surf_opacity,
            surf_decimation=surf_decimation,surf_kwargs=surf_kwargs,
        )


    def plot_interior_dispersion(
            self,
            plotter: Optional[pv.Plotter] = None,
            cmap: str = 'coolwarm',
            glyph_factor: float = 0.1,
            glyph_tolerance: float = 0.01,
            glyph_geom: Optional[pv.PolyData] = None,
            glyph_kwargs: Dict = {},
            show_surf: bool = True,
            surf_color: Any = 'gray',
            surf_opacity: float = 0.05,
            surf_decimation: float = 0.2,
            surf_kwargs: Dict = {},
        ) -> pv.Plotter:
        """
        Plots the energy dispersion field within the exceptional surface.

        This method visualizes the gradient of the imaginary part of the energy
        (dispersion) inside the exceptional surface. It calls `plot_vector_field`
        with orientation set to 'im_disp' and scaling to
        'log10(|im_disp|+1)'.

        Parameters
        ----------
        plotter : pv.Plotter, optional
            An existing PyVista plotter. If None, a new one is created.
        cmap : str, optional
            The colormap for the glyphs. Defaults to 'coolwarm'.
        glyph_factor : float, optional
            A scaling factor for the glyphs. Defaults to 0.05.
        glyph_tolerance : float, optional
            Controls the density of the glyphs. Defaults to 0.025.
        glyph_geom : pv.PolyData, optional
            The geometry to use for the glyphs. If None, PyVista's default is used.
        glyph_kwargs : Dict, optional
            Additional keyword arguments for the glyphs.
        show_surf : bool, optional
            If True, displays the translucent exceptional surface. Defaults to True.
        surf_color : Any, optional
            Color of the exceptional surface. Defaults to 'gray'.
        surf_opacity : float, optional
            Opacity of the exceptional surface. Defaults to 0.05.
        surf_decimation : float, optional
            Decimation factor for the surface mesh. Defaults to 0.2.
        surf_kwargs : Dict, optional
            Additional keyword arguments for the surface mesh.

        Returns
        -------
        pv.Plotter
            The PyVista plotter object with the dispersion field added.
        """
        return self.plot_vector_field(
            plotter=plotter,orient='im_disp',scale='log10(|im_disp|+1)',
            cmap=cmap,glyph_factor=glyph_factor,glyph_tolerance=glyph_tolerance,
            glyph_geom=glyph_geom,glyph_kwargs=glyph_kwargs,show_surf=show_surf,
            surf_color=surf_color,surf_opacity=surf_opacity,
            surf_decimation=surf_decimation,surf_kwargs=surf_kwargs,
        )


    @cached_property
    def PDCode(self) -> PDCode:
        """Construct a PDCode object from the skeleton graph.

        Returns
        -------
        PDCode
            The Planar Diagram as a `PDCode` object.
        """        
        if not self.skeleton_graph_cache:
            self.skeleton_graph()
        return PDCode(self.skeleton_graph_cache)


    def plot_planar_diagram(self,
            rotation_angles: Optional[tuple[float]] = None,
            rotation_order: str = 'ZYX',
            ax: Optional[plt.Axes] = None,
            edge_kwargs: Dict = {},
            vertex_kwargs: Dict = {},
            undercrossing_offset: float = 5.,
            mark_crossings: bool = False,
            crossing_kwargs: Dict = {},
        ) -> plt.Axes:
        """Plot the planar diagram.

        Parameters
        ----------
        rotation_angles : tuple[float], optional
            The angles for the rotations in radians. Defaults to (0., 0., 0.).
            See `NodalSkeleton.util.get_rotation_matrix` for details.
        rotation_order : str, optional
            The order of rotations to apply. Defaults to 'ZYX'.
            See `NodalSkeleton.util.get_rotation_matrix` for details.
        ax : plt.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        edge_kwargs : Dict, optional
            Additional keyword arguments for the edge glyphs.
        vertex_kwargs : Dict, optional
            Additional keyword arguments for the vertex glyphs.
        undercrossing_offset : float, optional
            The offset for undercrossing arcs. Unit in pixels. Defaults to 5.
        mark_crossings : bool, optional
            Whether to mark the crossings. Defaults to False.
        crossing_kwargs : Dict, optional
            Additional keyword arguments for the crossing glyphs.

        Returns
        -------
        plt.Axes
            The axes with the planar diagram plotted.
        """
        from collections import defaultdict
        from shapely.ops import substring

        pd = self.PDCode
        pd.compute(rotation_angles, rotation_order)
        if ax is None:
            fig, ax = plt.subplots(figsize=(3,3))
        
        under_arcs = defaultdict(lambda: [False, False])
        for x in pd.crossings.values():
            for uid in [x.ccw_ordered_arcs[i] for i in (0,2)]:
                arc = pd.arcs[uid]
                if arc.start_type == 'x' and arc.start_id == x.id:
                    under_arcs[uid][0] = True
                if arc.end_type == 'x' and arc.end_id == x.id:
                    under_arcs[uid][1] = True

        for arc in pd.arcs.values():
            line = arc.line
            t = undercrossing_offset
            L = line.length
            if under_arcs[arc.id][0] and under_arcs[arc.id][1]:
                line = substring(line, t, L - t)
            elif under_arcs[arc.id][0]:
                line = substring(line, t, L)
            elif under_arcs[arc.id][1]:
                line = substring(line, 0., L - t)
            edge_kwargs = {'color': 'tab:blue', 'zorder': -1, 
                           **edge_kwargs}
            ax.plot(*line.xy, **edge_kwargs)
        
        for v in pd.vertices.values():
            vertex_kwargs = {'s': 15, 'marker': 'o', 'color': 'tab:red', 
                             **vertex_kwargs}
            ax.scatter(*v.point.xy, **vertex_kwargs)

        if mark_crossings:
            for x in pd.crossings.values():
                crossing_kwargs = {'s': 30, 'marker': 'x', 'color': 'k', 
                                   **crossing_kwargs}
            ax.scatter(*x.point.xy, **crossing_kwargs)
        
        return ax


    def yamada_polynomial(
        self,
        variable: sp.Symbol,
        normalize: bool = True,
        n_jobs: int = -1,
        *,
        num_rotations: int = 10,
        rotation_angles: tuple[float] = (0.,0.,0.),
        rotation_order: str = 'ZYX'
    ) -> sp.Expr:
        """
        Computes the Yamada polynomial for the skeleton graph.

        Parameters
        ----------
        variable : sp.Symbol
            The variable to use in the polynomial.
        normalize : bool, optional
            Whether to normalize the polynomial. Defaults to True.
        n_jobs : int, optional
            The number of jobs to run in parallel. Defaults to -1.
        num_rotations: int, optional
            ONLY if the skeleton graph is trivalent.
            The number of different rotations to sample. Defaults to 10.
        rotation_angles : tuple[float], optional
            ONLY if the skeleton graph is NOT trivalent.
            The angles for the rotations in radians. Defaults to (0., 0., 0.).
            See `NodalSkeleton.util.get_rotation_matrix` for details.
        rotation_order : str, optional
            ONLY if the skeleton graph is NOT trivalent.
            The order of rotations to apply. Defaults to 'ZYX'.
            See `NodalSkeleton.util.get_rotation_matrix` for details.

        Returns
        -------
        sp.Expr
            The Yamada polynomial.
        """
        if not self.skeleton_graph_cache:
            self.skeleton_graph()

        if self.is_graph_trivalent:
            logging.info(
                f"The skeleton graph is trivalent, running {num_rotations} different"
                " rotations to compute the Yamada polynomial safely."
            )
            return compute_yamada_safely(
                self.skeleton_graph_cache, variable,
                normalize=normalize, n_jobs=n_jobs,
                num_rotations=num_rotations
            )
        else:
            logging.warning(
                "The skeleton graph is not trivalent. "
                f"Calculating the rotation angles {rotation_angles} "
                f"in {rotation_order} order."
            )
            return compute_yamada_polynomial(
                self.skeleton_graph_cache, variable,
                normalize=normalize, n_jobs=n_jobs,
                rotation_angles=rotation_angles,
                rotation_order=rotation_order
            )


    def graph_summary(
            self,
            G: Optional[nx.Graph | nx.MultiGraph] = None
        ) -> None:
        """
        Prints a summary of the skeleton graph's properties.

        Displays information such as the number of nodes and edges, connectivity,
        and the degree distribution.

        Parameters
        ----------
        G : nx.Graph or nx.MultiGraph, optional
            The graph to summarize. If None, the cached skeleton graph is used.
            Defaults to None.
        """
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


    def check_minor(
        self,
        minor_graph: nx.Graph,
        host_graph: Optional[nx.MultiGraph | nx.Graph] = None,
    ) -> Any:
        """
        Checks whether `minor_graph` is a minor of `host_graph`.

        This function uses the `minorminer` library to find an embedding of
        `minor_graph` within `host_graph`.

        Parameters
        ----------
        minor_graph : nx.Graph
            The graph to be checked as a minor.
        host_graph : nx.MultiGraph or nx.Graph, optional
            The graph in which to search for the minor.
            Defaults to None, looking for the self.skeleton_graph_cache.

        Returns
        -------
        Any
            The embedding mapping (a dictionary) if `minor_graph` is a minor
            of `host_graph`; otherwise, returns None.
        """
        if host_graph is None:
            host_graph = self.skeleton_graph_cache

        # Attempt to find an embedding of minor_graph in host_graph.
        embedding = minorminer.find_embedding(minor_graph, host_graph)

        if embedding:
            print("The given graph contains the minor graph.")
            return embedding
        else:
            print("The given graph DOES NOT contain the minor graph.")
            return None

    # --- symbolic expressions properties (for theoretical inspection) ---
    @cached_property
    def spectrum_expr(self) -> sp.Expr:
        """Symbolic expression for the complex energy spectrum."""
        return sp.sqrt(sum(b**2 for b in self.bloch_vec))

    @cached_property
    def band_gap_expr(self) -> sp.Expr:
        """Symbolic expression for the band gap."""
        return 2 * sp.Abs(self.spectrum_expr)

    @cached_property
    def dispersion_expr(self) -> sp.Matrix:
        """Symbolic expression for the group velocity vector."""
        # Group velocity is the gradient of the real part of the energy
        re_E = sp.re(self.spectrum_expr)
        return sp.Matrix([sp.diff(re_E, k) for k in self.k_symbols])

    @cached_property
    def berry_curvature_expr(self) -> sp.Matrix:
        """Symbolic expression for the Berry curvature vector."""
        prereqs = self._berry_prerequisites
        if not prereqs['valid']:
            raise NotImplementedError("Berry curvature is only defined for Bloch vectors "
                                      "with two real and one imaginary component.")

        gamma = prereqs['gamma']
        p, q, r = prereqs['d1_idx'], prereqs['d2_idx'], prereqs['gamma_idx']
        perm_sign = sp.LeviCivita(p, q, r)

        d1_expr = self.bloch_vec[p]
        d2_expr = self.bloch_vec[q]

        grad_d1 = sp.Matrix([sp.diff(d1_expr, k) for k in self.k_symbols])
        grad_d2 = sp.Matrix([sp.diff(d2_expr, k) for k in self.k_symbols])

        numerator_vec = perm_sign * grad_d2.cross(grad_d1)
        eps_sq = gamma**2 - d1_expr**2 - d2_expr**2
        denominator = 2 * eps_sq**sp.Rational(3, 2)

        return (gamma/denominator) * numerator_vec



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
    print(f"""Check minor: {ske.check_minor(
        ske.skeleton_graph_cache.subgraph([1]),
        ske.skeleton_graph_cache, 
    )}""")

    # Check plotting
    pl = pv.Plotter(shape=(1, 2), window_size=(1200, 500), off_screen=True)
    pl.link_views()
    pl.subplot(0, 0)
    ske.plot_exceptional_surface(plotter=pl)
    pl.subplot(0, 1)
    ske.plot_skeleton_graph(plotter=pl)
    pl.close()
    print("Plotting done. Use `pl.show()` to display the plot in a notebook.")