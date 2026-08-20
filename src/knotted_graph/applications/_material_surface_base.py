"""Hermitian multiband material-surface workflow.

This module promotes the former ``NodalSkeletonMultiBand`` research helper into
the public applications layer without changing the existing two-band
``NodalSkeleton`` workflow.

The numerical definition is intentionally preserved:

    selected pairwise band gap <= gap_tol

defines a thickened nodal region.  Its boundary is exposed as a PyVista surface,
and its morphological skeleton is converted to the standard embedded
``networkx.MultiGraph`` contract used elsewhere in KnottedGraph.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pyvista as pv
import sympy as sp
from numpy.typing import NDArray

from knotted_graph.applications.nodal.skeleton import NodalSkeleton
from knotted_graph.applications.nodal.symmetry import is_PT_symmetric
from knotted_graph.core import (
    is_trivalent,
    remove_leaf_nodes,
    simplify_edges,
    smooth_edges,
    total_edge_pts,
)
from knotted_graph.extraction import skeleton_image_to_graph, skeletonize_volume


class MaterialFermiSurface(NodalSkeleton):
    """Analyze a Hermitian multiband material Hamiltonian.

    This is the public applications-layer version of the former
    ``NodalSkeletonMultiBand`` helper.

    Parameters
    ----------
    char
        Square Hermitian SymPy Hamiltonian ``H(k)``.
    k_symbols
        Three momentum symbols ``(kx, ky, kz)``.  If omitted, they are inferred
        from the Hamiltonian.
    span
        Sampling bounds for each momentum direction.
    dimension
        Number of grid samples along each axis.
    axis_scale
        Geometric scale applied to the three coordinate axes.
    band_pair
        Pair of band indices ``(i, j)`` used to define the gap field.
    gap_tol
        Thickness threshold.  The interior mask is
        ``band_gap(i, j) <= gap_tol``.
    sort_by
        Preserved for compatibility with the former multiband helper.  Hermitian
        eigenvalues are evaluated with ``numpy.linalg.eigvalsh`` and are already
        sorted.
    gap_mode
        ``"abs"``, ``"real"``, or ``"imag"`` for the selected pairwise gap.
        For the Hermitian workflow, ``"abs"`` is the normal choice.
    compute_berry
        Preserved only as an explicit guard.  Berry curvature is not computed by
        this Hermitian multiband class.
    chunk_size
        Number of momentum points evaluated per eigenvalue batch.
    force_small_edge_contraction
        Whether to contract short graph edges after skeleton simplification.
    small_edge_limit
        Short-edge threshold measured in k-space coordinates.
    previous_n_edgepoint
        Number of nearby edge-polyline samples used when smoothly relocating an
        endpoint during a contraction.
    """

    def __init__(
        self,
        char: sp.Matrix,
        *,
        k_symbols=None,
        span=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
        dimension: int = 120,
        axis_scale=(1.0, 1.0, 1.0),
        band_pair=(0, 1),
        gap_tol: float = 1e-2,
        sort_by: str = "real_imag",
        gap_mode: str = "abs",
        compute_berry: bool = False,
        chunk_size: int = 50_000,
        force_small_edge_contraction: bool = False,
        small_edge_limit: float = 0.0,
        previous_n_edgepoint: int = 20,
    ):
        material_hamiltonian = sp.Matrix(char)
        if material_hamiltonian.rows != material_hamiltonian.cols:
            raise ValueError("`char` must be a square sympy Matrix (NxN).")

        self.n_bands = int(material_hamiltonian.rows)

        if k_symbols is None:
            syms = sorted(material_hamiltonian.free_symbols, key=lambda s: s.name)
            if len(syms) != 3:
                raise ValueError(
                    "Could not infer exactly 3 k-symbols from H(k). "
                    "Pass k_symbols=(kx, ky, kz) and substitute all other parameters."
                )
            resolved_k_symbols = tuple(syms)
        else:
            if len(k_symbols) != 3:
                raise ValueError(
                    "`k_symbols` must be a tuple of three sympy symbols (kx, ky, kz)."
                )
            resolved_k_symbols = tuple(k_symbols)

        extra = set(material_hamiltonian.free_symbols) - set(resolved_k_symbols)
        if extra:
            raise ValueError(
                "H(k) contains free symbols other than kx,ky,kz. "
                "Substitute them to numbers before constructing this class. "
                f"Extra: {sorted(s.name for s in extra)}"
            )

        band_pair = tuple(band_pair)
        if len(band_pair) != 2 or band_pair[0] == band_pair[1]:
            raise ValueError("band_pair must be a tuple (i,j) with i != j.")

        self.band_pair = band_pair
        self.gap_tol = float(gap_tol)
        self.sort_by = str(sort_by)
        self.gap_mode = str(gap_mode)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        self.force_small_edge_contraction = bool(force_small_edge_contraction)
        self.small_edge_limit = float(small_edge_limit)
        if self.small_edge_limit < 0:
            raise ValueError("small_edge_limit must be >= 0.")

        self.previous_n_edgepoint = int(previous_n_edgepoint)
        if self.previous_n_edgepoint < 0:
            raise ValueError("previous_n_edgepoint must be >= 0.")

        hermitian = (
            sp.simplify(material_hamiltonian - material_hamiltonian.H)
            == sp.zeros(self.n_bands, self.n_bands)
        )
        if not hermitian:
            raise ValueError(
                "MaterialFermiSurface requires a Hermitian H(k)=H(k)†. "
                "Use the non-Hermitian NodalSkeleton workflow for exceptional-surface physics."
            )

        try:
            pt_symmetric = is_PT_symmetric(material_hamiltonian)
        except Exception:
            pt_symmetric = False

        if compute_berry:
            raise ValueError(
                "Berry curvature is not part of this Hermitian multiband workflow."
            )
        self.compute_berry = False

        # Lambdify before initializing the two-band base class.  The base class is
        # used only for the common sampling grid and visualization/graph helpers.
        self._h_elem = [[None] * self.n_bands for _ in range(self.n_bands)]
        for i in range(self.n_bands):
            for j in range(self.n_bands):
                expr = material_hamiltonian[i, j]
                if expr.free_symbols:
                    self._h_elem[i][j] = sp.lambdify(
                        resolved_k_symbols, expr, "numpy"
                    )
                else:
                    self._h_elem[i][j] = complex(expr)

        dummy = sp.Matrix([[0, 0], [0, 0]])
        super().__init__(
            dummy,
            k_symbols=resolved_k_symbols,
            span=span,
            dimension=dimension,
            axis_scale=axis_scale,
        )

        # Restore the actual multiband metadata after the base class has created
        # the common momentum grid.
        self.h_k = material_hamiltonian
        self.k_symbols = resolved_k_symbols
        self.kx_symbol, self.ky_symbol, self.kz_symbol = resolved_k_symbols
        self.is_Hermitian = True
        self.is_PT_symmetric = pt_symmetric

        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None
        self._pv_data_args = None
        self.is_graph_trivalent = None

    @classmethod
    def from_hamiltonian(cls, hamiltonian: sp.Matrix, **kwargs):
        """Construct from a material Hamiltonian.

        This convenience constructor does not introduce new physics or filtering;
        it is exactly equivalent to ``MaterialFermiSurface(hamiltonian, **kwargs)``.
        """
        return cls(hamiltonian, **kwargs)

    def _pair_indices(self) -> tuple[int, int]:
        i, j = self.band_pair
        if not (0 <= i < self.n_bands and 0 <= j < self.n_bands):
            raise ValueError(
                f"band_pair indices must be in [0,{self.n_bands - 1}]. "
                f"Got {self.band_pair}."
            )
        return int(i), int(j)

    def _eval_H_chunk(
        self,
        kx: NDArray,
        ky: NDArray,
        kz: NDArray,
    ) -> NDArray:
        """Evaluate ``H(k)`` for one vectorized chunk."""
        n_points = kx.size
        H = np.empty(
            (n_points, self.n_bands, self.n_bands),
            dtype=np.complex128,
        )

        for i in range(self.n_bands):
            for j in range(self.n_bands):
                entry = self._h_elem[i][j]
                if callable(entry):
                    H[:, i, j] = entry(kx, ky, kz)
                else:
                    H[:, i, j] = entry
        return H

    @cached_property
    def eigvals_sorted(self) -> NDArray:
        """Hermitian eigenvalues on the sampled grid, sorted ascending."""
        nx_, ny_, nz_ = self.kx_grid.shape
        n_points = nx_ * ny_ * nz_

        kx = self.kx_grid.ravel(order="F")
        ky = self.ky_grid.ravel(order="F")
        kz = self.kz_grid.ravel(order="F")

        out = np.empty((n_points, self.n_bands), dtype=np.float64)

        for start in range(0, n_points, self.chunk_size):
            end = min(n_points, start + self.chunk_size)
            H = self._eval_H_chunk(kx[start:end], ky[start:end], kz[start:end])
            out[start:end, :] = np.linalg.eigvalsh(H)

        return out.reshape(
            (nx_, ny_, nz_, self.n_bands),
            order="F",
        )

    @cached_property
    def spectrum(self) -> NDArray:
        """Half of the selected pairwise band separation."""
        i, j = self._pair_indices()
        Ei = self.eigvals_sorted[..., i]
        Ej = self.eigvals_sorted[..., j]
        return 0.5 * (Ej - Ei)

    @cached_property
    def band_gap(self) -> NDArray:
        """Selected pairwise band gap."""
        dE = 2.0 * self.spectrum
        if self.gap_mode == "abs":
            return np.abs(dE)
        if self.gap_mode == "real":
            return np.abs(np.real(dE))
        if self.gap_mode == "imag":
            return np.abs(np.imag(dE))
        raise ValueError("gap_mode must be one of: abs, real, imag")

    @property
    def _interior_mask(self) -> NDArray:
        """Thickened nodal region: ``band_gap <= gap_tol``."""
        return self.band_gap <= self.gap_tol

    @cached_property
    def _skeleton_image(self) -> NDArray:
        """Optimized Lee skeleton of the thickened nodal region."""
        try:
            return skeletonize_volume(self._interior_mask)
        except ValueError as exc:
            if "does not contain any True voxels" in str(exc):
                raise ValueError(
                    "The skeleton image is empty. "
                    "Try increasing gap_tol, checking band_pair, or enlarging the k-span."
                ) from exc
            raise

    @property
    def berry_curvature(self):
        """Berry curvature is intentionally not defined in this workflow."""
        raise NotImplementedError(
            "MaterialFermiSurface is the Hermitian multiband gap-skeleton workflow. "
            "Use NodalSkeleton for the non-Hermitian Berry-curvature workflow."
        )

    @cached_property
    def fields_pv(self) -> pv.PolyData:
        """Return sampled multiband gap and gradient fields as PyVista data."""
        engy = self.spectrum
        vol = pv.ImageData(
            dimensions=engy.shape,
            spacing=self.spacing * self.axis_scale,
            origin=self.origin,
        )

        gap = self.band_gap

        vol.point_data["real"] = np.asarray(
            engy, dtype=np.float64
        ).ravel(order="F")
        vol.point_data["imag"] = np.zeros_like(
            gap, dtype=np.float64
        ).ravel(order="F")
        vol.point_data["gap"] = gap.ravel(order="F")
        vol.point_data["ES_helper"] = (
            gap - self.gap_tol
        ).ravel(order="F")

        disp = np.stack(
            np.gradient(gap, *self.spacing, edge_order=2),
            axis=-1,
        )
        disp[~self._interior_mask] = 0.0
        disp = disp.reshape(-1, 3, order="F")
        disp_norm = np.linalg.norm(disp, axis=-1)

        vol.point_data["im_disp"] = disp
        vol.point_data["|im_disp|"] = disp_norm
        vol.point_data["log10(|im_disp|+1)"] = np.log10(
            disp_norm + 1
        )

        return vol

    @cached_property
    def exceptional_surface_pv(self) -> pv.PolyData:
        """Boundary isosurface ``band_gap == gap_tol``."""
        return self.fields_pv.contour(
            isosurfaces=[0.0],
            scalars="ES_helper",
        )

    def _node_coord(
        self,
        graph: nx.MultiGraph,
        node: Any,
    ) -> NDArray:
        """Convert one graph-node image coordinate to k-space."""
        pos = np.asarray(graph.nodes[node]["pos"], dtype=float)
        return self._idx_to_coord(pos.reshape(1, 3))[0]

    @staticmethod
    def _hermite(
        P0: NDArray,
        P1: NDArray,
        m0: NDArray,
        m1: NDArray,
        n: int,
    ) -> NDArray:
        t = np.linspace(0.0, 1.0, int(n), dtype=float)
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return (
            h00[:, None] * P0
            + h10[:, None] * m0
            + h01[:, None] * P1
            + h11[:, None] * m1
        )

    @staticmethod
    def _dedup_consecutive(
        arr: NDArray,
        tol: float = 1e-9,
    ) -> NDArray:
        if arr.shape[0] <= 1:
            return arr
        keep = [0]
        for i in range(1, arr.shape[0]):
            if np.linalg.norm(arr[i] - arr[keep[-1]]) > tol:
                keep.append(i)
        return arr[keep]

    def _smooth_move_endpoint_in_pts(
        self,
        pts,
        old_pos: NDArray,
        new_pos: NDArray,
        n_prev: int,
    ):
        """Smoothly move one polyline endpoint during edge contraction."""
        arr = np.asarray(pts, dtype=float)
        if (
            arr.ndim != 2
            or arr.shape[1] != 3
            or arr.shape[0] == 0
        ):
            return pts

        m = arr.shape[0]
        if m < 3:
            d0 = np.linalg.norm(arr[0] - old_pos)
            d1 = np.linalg.norm(arr[-1] - old_pos)
            arr[0 if d0 <= d1 else -1] = new_pos
            return arr.tolist()

        d0 = np.linalg.norm(arr[0] - old_pos)
        d1 = np.linalg.norm(arr[-1] - old_pos)
        move_start = d0 <= d1

        n_prev = int(max(0, n_prev))
        alpha = 0.5

        if move_start:
            join_idx = min(n_prev, m - 2)
            Pj = arr[join_idx]
            Pn = arr[join_idx + 1]
            vj = Pn - Pj
            if np.linalg.norm(vj) < 1e-12:
                vj = Pj - new_pos

            dist = float(np.linalg.norm(Pj - new_pos))
            if dist < 1e-12:
                arr[0] = new_pos
                return arr.tolist()

            m0 = (
                (Pj - new_pos)
                / (dist + 1e-12)
                * (alpha * dist)
            )
            m1 = (
                vj
                / (np.linalg.norm(vj) + 1e-12)
                * (alpha * dist)
            )

            seg = self._hermite(
                new_pos,
                Pj,
                m0,
                m1,
                n=join_idx + 1,
            )
            seg[0] = new_pos
            seg[-1] = Pj
            out = np.vstack([seg, arr[join_idx + 1 :]])
            return self._dedup_consecutive(out).tolist()

        join_idx = max(1, (m - 1) - min(n_prev, m - 2))
        Pj = arr[join_idx]
        Pp = arr[join_idx - 1]
        vj = Pj - Pp
        if np.linalg.norm(vj) < 1e-12:
            vj = new_pos - Pj

        dist = float(np.linalg.norm(new_pos - Pj))
        if dist < 1e-12:
            arr[-1] = new_pos
            return arr.tolist()

        m0 = (
            vj
            / (np.linalg.norm(vj) + 1e-12)
            * (alpha * dist)
        )
        m1 = (
            (new_pos - Pj)
            / (dist + 1e-12)
            * (alpha * dist)
        )

        seg = self._hermite(
            Pj,
            new_pos,
            m0,
            m1,
            n=(m - join_idx),
        )
        seg[0] = Pj
        seg[-1] = new_pos
        out = np.vstack([arr[:join_idx], seg])
        return self._dedup_consecutive(out).tolist()

    def _contract_one_edge(
        self,
        graph: nx.MultiGraph,
        u: Any,
        v: Any,
        *,
        previous_n_edgepoint: int = 20,
    ) -> bool:
        """Contract one edge using the legacy multiband endpoint treatment."""
        if u not in graph or v not in graph or u == v:
            return False

        du = graph.degree[u]
        dv = graph.degree[v]

        if du > dv:
            keep, kill = u, v
        elif dv > du:
            keep, kill = v, u
        else:
            keep, kill = (
                (u, v)
                if str(u) <= str(v)
                else (v, u)
            )

        keep_pos = np.asarray(
            graph.nodes[keep]["pos"],
            dtype=float,
        )
        kill_pos = np.asarray(
            graph.nodes[kill]["pos"],
            dtype=float,
        )

        incident = list(
            graph.edges(kill, keys=True, data=True)
        )

        for a, b, key, data in incident:
            other = b if a == kill else a

            if graph.has_edge(a, b, key):
                graph.remove_edge(a, b, key)

            if other == keep:
                continue

            new_data = dict(data) if data is not None else {}

            if new_data.get("pts") is not None:
                new_pts = self._smooth_move_endpoint_in_pts(
                    new_data["pts"],
                    old_pos=kill_pos,
                    new_pos=keep_pos,
                    n_prev=previous_n_edgepoint,
                )

                arr = np.asarray(new_pts, dtype=float)
                if (
                    arr.ndim == 2
                    and arr.shape[0] > 0
                    and arr.shape[1] == 3
                ):
                    other_pos = np.asarray(
                        graph.nodes[other]["pos"],
                        dtype=float,
                    )
                    if (
                        np.linalg.norm(arr[0] - keep_pos)
                        <= np.linalg.norm(arr[-1] - keep_pos)
                    ):
                        arr[0] = keep_pos
                        arr[-1] = other_pos
                    else:
                        arr[-1] = keep_pos
                        arr[0] = other_pos
                    new_data["pts"] = arr
                else:
                    new_data["pts"] = new_pts

            graph.add_edge(keep, other, **new_data)

        if kill in graph:
            graph.remove_node(kill)

        return True

    def _contract_small_edges(
        self,
        graph: nx.MultiGraph,
        small_edge_limit: float,
        *,
        previous_n_edgepoint: Optional[int] = None,
    ) -> nx.MultiGraph:
        """Iteratively contract short edges measured in k-space."""
        if small_edge_limit <= 0:
            return graph

        nprev = (
            self.previous_n_edgepoint
            if previous_n_edgepoint is None
            else int(previous_n_edgepoint)
        )
        if nprev < 0:
            raise ValueError("previous_n_edgepoint must be >= 0.")

        while True:
            short_edges: list[tuple[float, Any, Any]] = []
            for u, v in graph.edges():
                if u == v:
                    continue
                try:
                    pu = self._node_coord(graph, u)
                    pv_ = self._node_coord(graph, v)
                except Exception:
                    continue

                length = float(np.linalg.norm(pu - pv_))
                if length < small_edge_limit:
                    short_edges.append((length, u, v))

            if not short_edges:
                break

            short_edges.sort(key=lambda item: item[0])
            _, u, v = short_edges[0]
            changed = self._contract_one_edge(
                graph,
                u,
                v,
                previous_n_edgepoint=nprev,
            )
            if not changed:
                try:
                    graph.remove_edge(u, v)
                except Exception:
                    break

        return graph

    def skeleton_graph(
        self,
        simplify: bool = True,
        smooth_epsilon: int = 4,
        *,
        skeleton_image: Optional[
            Union[NDArray, nx.Graph, nx.MultiGraph]
        ] = None,
        force_small_edge_contraction: Optional[bool] = None,
        small_edge_limit: Optional[float] = None,
        previous_n_edgepoint: Optional[int] = None,
    ) -> nx.MultiGraph:
        """Convert the multiband mask skeleton to an embedded graph."""
        force = (
            self.force_small_edge_contraction
            if force_small_edge_contraction is None
            else bool(force_small_edge_contraction)
        )
        limit = (
            self.small_edge_limit
            if small_edge_limit is None
            else float(small_edge_limit)
        )
        nprev = (
            self.previous_n_edgepoint
            if previous_n_edgepoint is None
            else int(previous_n_edgepoint)
        )
        if nprev < 0:
            raise ValueError("previous_n_edgepoint must be >= 0.")

        args = (
            smooth_epsilon,
            simplify,
            id(skeleton_image),
            force,
            limit,
            nprev,
        )
        if (
            self.skeleton_graph_cache is not None
            and self.skeleton_graph_cache_args == args
        ):
            return self.skeleton_graph_cache

        if skeleton_image is None:
            graph = skeleton_image_to_graph(self._skeleton_image)
        elif isinstance(
            skeleton_image,
            (nx.Graph, nx.MultiGraph),
        ):
            graph = (
                skeleton_image
                if isinstance(skeleton_image, nx.MultiGraph)
                else nx.MultiGraph(skeleton_image)
            )
        else:
            graph = skeleton_image_to_graph(skeleton_image)

        if simplify:
            graph = remove_leaf_nodes(graph)
            graph = simplify_edges(graph)

        if force and limit > 0:
            graph = self._contract_small_edges(
                graph,
                limit,
                previous_n_edgepoint=nprev,
            )

        graph = smooth_edges(
            graph,
            epsilon=smooth_epsilon,
            copy=False,
        )
        graph.graph["is_trivalent"] = is_trivalent(graph)
        self.is_graph_trivalent = graph.graph["is_trivalent"]

        self.skeleton_graph_cache = graph
        self.skeleton_graph_cache_args = args
        return graph

    @property
    def total_edge_pts(self) -> int:
        return total_edge_pts(
            self.skeleton_graph_cache
            or self.skeleton_graph()
        )

    def clear_cache(self):
        """Clear multiband graph, field, and eigenvalue caches."""
        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None
        self._pv_data_args = None

        for name in (
            "_skeleton_image",
            "eigvals_sorted",
            "spectrum",
            "band_gap",
            "fields_pv",
            "exceptional_surface_pv",
        ):
            self.__dict__.pop(name, None)
