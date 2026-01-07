from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional, Sequence, Tuple, Union,Sequence

import logging
import numpy as np
import networkx as nx
import pyvista as pv
import skimage.morphology as morph
import sympy as sp
from numpy.typing import NDArray

from poly2graph import skeleton2graph

from knotted_graph.NodalSkeleton import NodalSkeleton
from knotted_graph.util import (
    remove_leaf_nodes,
    simplify_edges,
    smooth_edges,
    total_edge_pts,
    is_PT_symmetric,
    is_trivalent,
)

log = logging.getLogger(__name__)


class NodalSkeletonMultiBand(NodalSkeleton):
    """
    Hermitian multiband extension of NodalSkeleton for an NxN sympy Hamiltonian H(k).

    Defines a "thickened nodal region" via a selected band pair (i,j):
        band_gap(i,j) <= gap_tol

    Then skeletonizes that 3D region and converts the skeleton to a graph.

    Additions in this version:
      - skeleton_pad removed (no padding before skeletonization).
      - optional small-edge contraction in skeleton_graph:
            force_small_edge_contraction (default False)
            small_edge_limit (k-space distance threshold)
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
        sort_by: str = "real_imag",   # kept for API compatibility (unused for Hermitian eigvalsh)
        gap_mode: str = "abs",        # "abs" uses |Ei - Ej|
        compute_berry: bool = False,  # forced OFF for Hermitian-only pipeline
        chunk_size: int = 50_000,
        force_small_edge_contraction: bool = False,
        small_edge_limit: float = 0.0,
        previous_n_edgepoint: int = 20
    ):
        # --- Hamiltonian validation ---
        self.h_k = sp.Matrix(char)
        if self.h_k.rows != self.h_k.cols:
            raise ValueError("`char` must be a square sympy Matrix (NxN).")
        self.n_bands = int(self.h_k.rows)

        # --- k symbols ---
        if k_symbols is None:
            syms = sorted(self.h_k.free_symbols, key=lambda s: s.name)
            if len(syms) != 3:
                raise ValueError(
                    "Could not infer exactly 3 k-symbols from H(k). "
                    "Pass k_symbols=(kx,ky,kz) and substitute all other parameters."
                )
            self.k_symbols = tuple(syms)
        else:
            if len(k_symbols) != 3:
                raise ValueError("`k_symbols` must be a tuple of three sympy symbols (kx, ky, kz).")
            self.k_symbols = tuple(k_symbols)

        self.kx_symbol, self.ky_symbol, self.kz_symbol = self.k_symbols

        extra = set(self.h_k.free_symbols) - set(self.k_symbols)
        if extra:
            raise ValueError(
                "H(k) contains free symbols other than kx,ky,kz. "
                f"Substitute them to numbers before constructing this class. Extra: {sorted([s.name for s in extra])}"
            )

        # --- options ---
        self.band_pair = tuple(band_pair)
        if len(self.band_pair) != 2 or self.band_pair[0] == self.band_pair[1]:
            raise ValueError("band_pair must be a tuple (i,j) with i != j.")
        self.gap_tol = float(gap_tol)
        self.sort_by = str(sort_by)  # unused for Hermitian, preserved
        self.gap_mode = str(gap_mode)
        self.chunk_size = int(chunk_size)

        self.force_small_edge_contraction = bool(force_small_edge_contraction)
        self.small_edge_limit = float(small_edge_limit)
        if self.small_edge_limit < 0:
            raise ValueError("small_edge_limit must be >= 0.")
        self.previous_n_edgepoint = int(previous_n_edgepoint)
        if self.previous_n_edgepoint < 0:
            raise ValueError("previous_n_edgepoint must be >= 0.")
        # --- Hermitian check ---
        self.is_Hermitian = sp.simplify(self.h_k - self.h_k.H) == sp.zeros(self.n_bands, self.n_bands)
        if not self.is_Hermitian:
            raise ValueError(
                "This Hermitian-only NodalSkeletonMultiBand requires H(k)=H(k)†. "
                "Use your non-Hermitian pipeline/class for exceptional-surface physics."
            )

        # PT check (metadata)
        try:
            self.is_PT_symmetric = is_PT_symmetric(self.h_k)
        except Exception:
            self.is_PT_symmetric = False

        # Berry disabled
        if compute_berry:
            raise ValueError(
                "Berry curvature support has been removed/disabled in this Hermitian-only class."
            )
        self.compute_berry = False

        # --- lambdify each matrix element ---
        self._h_elem = [[None] * self.n_bands for _ in range(self.n_bands)]
        for i in range(self.n_bands):
            for j in range(self.n_bands):
                expr = self.h_k[i, j]
                if expr.free_symbols:
                    self._h_elem[i][j] = sp.lambdify(self.k_symbols, expr, "numpy")
                else:
                    self._h_elem[i][j] = complex(expr)

        # --- initialize the base NodalSkeleton grids/geometry ---
        dummy = sp.Matrix([[0, 0], [0, 0]])
        super().__init__(
            dummy,
            k_symbols=self.k_symbols,
            span=span,
            dimension=dimension,
            axis_scale=axis_scale,
        )

        # caches
        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None
        self._pv_data_args = None
        self.is_graph_trivalent = None

    # -------------------------
    # helpers
    # -------------------------
    def _pair_indices(self) -> tuple[int, int]:
        i, j = self.band_pair
        if not (0 <= i < self.n_bands and 0 <= j < self.n_bands):
            raise ValueError(f"band_pair indices must be in [0,{self.n_bands-1}]. Got {self.band_pair}.")
        return (int(i), int(j))

    def _eval_H_chunk(self, kx: NDArray, ky: NDArray, kz: NDArray) -> NDArray:
        """
        Evaluate H(k) for a vectorized chunk of points.
        Returns shape (M, n, n).
        """
        M = kx.size
        H = np.empty((M, self.n_bands, self.n_bands), dtype=np.complex128)

        for i in range(self.n_bands):
            for j in range(self.n_bands):
                f = self._h_elem[i][j]
                if callable(f):
                    H[:, i, j] = f(kx, ky, kz)
                else:
                    H[:, i, j] = f
        return H

    @cached_property
    def eigvals_sorted(self) -> NDArray:
        """
        Hermitian eigenvalues on the full grid, sorted ascending per k (eigvalsh).
        Shape: (Nx, Ny, Nz, n_bands)
        """
        Nx, Ny, Nz = self.kx_grid.shape
        M = Nx * Ny * Nz

        kx = self.kx_grid.ravel(order="F")
        ky = self.ky_grid.ravel(order="F")
        kz = self.kz_grid.ravel(order="F")

        out = np.empty((M, self.n_bands), dtype=np.float64)

        cs = self.chunk_size
        for start in range(0, M, cs):
            end = min(M, start + cs)
            H = self._eval_H_chunk(kx[start:end], ky[start:end], kz[start:end])
            # Hermitian => real eigvals
            w = np.linalg.eigvalsh(H)
            out[start:end, :] = w

        return out.reshape((Nx, Ny, Nz, self.n_bands), order="F")

    @cached_property
    def spectrum(self) -> NDArray:
        """
        For compatibility with NodalSkeleton:
            spectrum := (E_j - E_i)/2
        Shape: (Nx, Ny, Nz)
        """
        i, j = self._pair_indices()
        Ei = self.eigvals_sorted[..., i]
        Ej = self.eigvals_sorted[..., j]
        return 0.5 * (Ej - Ei)

    @cached_property
    def band_gap(self) -> NDArray:
        """
        Pairwise band gap for the selected bands. Default: |E_j - E_i|.
        Shape: (Nx, Ny, Nz)
        """
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
        """Thickened nodal region: band_gap <= gap_tol."""
        return self.band_gap <= self.gap_tol

    # -------------------------
    # skeleton image (NO padding)
    # -------------------------
    @cached_property
    def _skeleton_image(self) -> NDArray:
        """
        Skeleton of the interior mask (no padding).
        """
        skel = morph.skeletonize(self._interior_mask, method="lee")
        if np.sum(skel) == 0:
            raise ValueError(
                "The skeleton image is empty. "
                "Try increasing gap_tol, checking band_pair, or enlarging the k-span."
            )
        return skel

    # -------------------------
    # PyVista fields / surface
    # -------------------------
    @cached_property
    def fields_pv(self) -> pv.PolyData:
        """
        Like base fields_pv, but:
          - 'gap' is the pairwise gap
          - 'ES_helper' := (gap - gap_tol) so contour at 0 gives gap == gap_tol isosurface
          - 'im_disp' is repurposed as grad(gap) (keeps key for plot_vector_field compatibility)
        """
        engy = self.spectrum
        vol = pv.ImageData(
            dimensions=engy.shape,
            spacing=self.spacing * self.axis_scale,
            origin=self.origin,
        )

        gap = self.band_gap

        vol.point_data["real"] = np.asarray(engy, dtype=np.float64).ravel(order="F")
        vol.point_data["imag"] = np.zeros_like(gap, dtype=np.float64).ravel(order="F")
        vol.point_data["gap"] = gap.ravel(order="F")
        vol.point_data["ES_helper"] = (gap - self.gap_tol).ravel(order="F")

        disp = np.stack(np.gradient(gap, *self.spacing, edge_order=2), axis=-1)
        disp[~self._interior_mask] = 0.0
        disp = disp.reshape(-1, 3, order="F")
        disp_norm = np.linalg.norm(disp, axis=-1)

        vol.point_data["im_disp"] = disp
        vol.point_data["|im_disp|"] = disp_norm
        vol.point_data["log10(|im_disp|+1)"] = np.log10(disp_norm + 1)

        return vol

    @cached_property
    def exceptional_surface_pv(self) -> pv.PolyData:
        """Isosurface of ES_helper==0, i.e. band_gap == gap_tol."""
        return self.fields_pv.contour(isosurfaces=[0.0], scalars="ES_helper")

    # -------------------------
    # Small-edge contraction
    # -------------------------
    def _node_coord(self, G: nx.MultiGraph, n: Any) -> NDArray:
        """Convert a node 'pos' (index coords) to k-space coordinate."""
        pos = np.asarray(G.nodes[n]["pos"], dtype=float)
        # base helper expects (N,3); keep it robust
        return self._idx_to_coord(pos.reshape(1, 3))[0]

    @staticmethod
    def _replace_endpoint_in_pts(
        pts: Sequence[Sequence[float]],
        old_pos: NDArray,
        new_pos: NDArray,
    ) -> list[list[float]]:
        """
        Replace whichever endpoint of pts is closer to old_pos with new_pos.
        pts are in index-coordinate space.
        """
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            # nothing to do / unexpected format
            return [list(p) for p in pts]

        d0 = np.linalg.norm(arr[0] - old_pos)
        d1 = np.linalg.norm(arr[-1] - old_pos)
        if d0 <= d1:
            arr[0] = new_pos
        else:
            arr[-1] = new_pos
        return arr.tolist()

    def _contract_one_edge(
        self,
        G: nx.MultiGraph,
        u: Any,
        v: Any,
        *,
        previous_n_edgepoint: int = 20,
    ) -> bool:
        """
        Contract u--v by merging the lower-degree node into the higher-degree node.

        Key behavior (smooth contraction of edge polylines):
        - For every edge incident to the node being removed ("kill"), we rewire it to "keep".
        - We update that edge's 'pts' by preserving the polyline except for the last/first
            `previous_n_edgepoint` samples near the moved endpoint, which are replaced by a
            smooth (C1-like) cubic Hermite segment that lands on the contracted node.

        Returns True if contraction happened.
        """
        import numpy as np

        # -----------------------
        # local helpers (self-contained)
        # -----------------------
        def _hermite(P0, P1, m0, m1, n: int) -> np.ndarray:
            t = np.linspace(0.0, 1.0, int(n), dtype=float)
            t2 = t * t
            t3 = t2 * t
            h00 =  2.0 * t3 - 3.0 * t2 + 1.0
            h10 =        t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 =        t3 -       t2
            return (h00[:, None] * P0 +
                    h10[:, None] * m0 +
                    h01[:, None] * P1 +
                    h11[:, None] * m1)

        def _dedup_consecutive(arr: np.ndarray, tol: float = 1e-9) -> np.ndarray:
            if arr.shape[0] <= 1:
                return arr
            keep_idx = [0]
            for i in range(1, arr.shape[0]):
                if np.linalg.norm(arr[i] - arr[keep_idx[-1]]) > tol:
                    keep_idx.append(i)
            return arr[keep_idx]

        def _smooth_move_endpoint_in_pts(
            pts,
            old_pos: np.ndarray,
            new_pos: np.ndarray,
            n_prev: int,
        ):
            """
            Move whichever endpoint is closer to old_pos onto new_pos,
            but do it smoothly by replacing the last/first `n_prev` points
            with a Hermite segment that matches the existing tangent at the join.
            """
            arr = np.asarray(pts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
                return pts

            m = arr.shape[0]
            if m < 3:
                # Too short for shaping; just hard snap the closer endpoint.
                d0 = np.linalg.norm(arr[0] - old_pos)
                d1 = np.linalg.norm(arr[-1] - old_pos)
                arr[0 if d0 <= d1 else -1] = new_pos
                return arr.tolist()

            d0 = np.linalg.norm(arr[0] - old_pos)
            d1 = np.linalg.norm(arr[-1] - old_pos)
            move_start = (d0 <= d1)

            n_prev = int(max(0, n_prev))
            # clamp so we always have a neighbor to compute tangent
            if move_start:
                join_idx = min(n_prev, m - 2)  # need join_idx+1
                Pj = arr[join_idx]
                Pn = arr[join_idx + 1]
                vj = Pn - Pj
                if np.linalg.norm(vj) < 1e-12:
                    vj = Pj - new_pos

                dist = float(np.linalg.norm(Pj - new_pos))
                if dist < 1e-12:
                    arr[0] = new_pos
                    return arr.tolist()

                # Tangent magnitudes (tunable): scale with distance to avoid sharp kinks
                alpha = 0.5
                m0 = (Pj - new_pos) / (dist + 1e-12) * (alpha * dist)   # at new_pos
                m1 = vj / (np.linalg.norm(vj) + 1e-12) * (alpha * dist) # at join

                seg = _hermite(new_pos, Pj, m0, m1, n=join_idx + 1)
                seg[0] = new_pos
                seg[-1] = Pj
                out = np.vstack([seg, arr[join_idx + 1:]])
                out = _dedup_consecutive(out)
                return out.tolist()

            else:
                join_idx = max(1, (m - 1) - min(n_prev, m - 2))  # need join_idx-1
                Pj = arr[join_idx]
                Pp = arr[join_idx - 1]
                vj = Pj - Pp
                if np.linalg.norm(vj) < 1e-12:
                    vj = new_pos - Pj

                dist = float(np.linalg.norm(new_pos - Pj))
                if dist < 1e-12:
                    arr[-1] = new_pos
                    return arr.tolist()

                alpha = 0.5
                m0 = vj / (np.linalg.norm(vj) + 1e-12) * (alpha * dist)          # at join
                m1 = (new_pos - Pj) / (dist + 1e-12) * (alpha * dist)            # at new_pos

                seg = _hermite(Pj, new_pos, m0, m1, n=(m - join_idx))
                seg[0] = Pj
                seg[-1] = new_pos
                out = np.vstack([arr[:join_idx], seg])
                out = _dedup_consecutive(out)
                return out.tolist()

        # -----------------------
        # contraction proper
        # -----------------------
        if u not in G or v not in G or u == v:
            return False

        du = G.degree[u]
        dv = G.degree[v]

        # keep = higher-degree node (tie-break by stable ordering)
        if du > dv:
            keep, kill = u, v
        elif dv > du:
            keep, kill = v, u
        else:
            keep, kill = (u, v) if str(u) <= str(v) else (v, u)

        keep_pos = np.asarray(G.nodes[keep]["pos"], dtype=float)
        kill_pos = np.asarray(G.nodes[kill]["pos"], dtype=float)

        # collect all incident edges to 'kill' first (includes keep--kill edge(s))
        incident = list(G.edges(kill, keys=True, data=True))

        # remove edges kill-keep (the contracted edge(s)) and rewire the rest
        for a, b, k, data in incident:
            other = b if a == kill else a

            # remove original edge
            if G.has_edge(a, b, k):
                G.remove_edge(a, b, k)

            if other == keep:
                continue  # contracted edge(s)

            new_data = dict(data) if data is not None else {}

            # Smoothly move the endpoint that was at/near kill_pos onto keep_pos
            if new_data.get("pts") is not None:
                new_pts = _smooth_move_endpoint_in_pts(
                    new_data["pts"],
                    old_pos=kill_pos,
                    new_pos=keep_pos,
                    n_prev=previous_n_edgepoint,
                )

                # Hard-snap endpoints to node positions (robust against orientation)
                arr = np.asarray(new_pts, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3:
                    other_pos = np.asarray(G.nodes[other]["pos"], dtype=float)
                    # decide which end is keep, which is other
                    if np.linalg.norm(arr[0] - keep_pos) <= np.linalg.norm(arr[-1] - keep_pos):
                        arr[0] = keep_pos
                        arr[-1] = other_pos
                    else:
                        arr[-1] = keep_pos
                        arr[0] = other_pos
                    new_data["pts"] = arr.tolist()
                else:
                    new_data["pts"] = new_pts

            G.add_edge(keep, other, **new_data)

        # finally remove node
        if kill in G:
            G.remove_node(kill)

        return True



    def _contract_small_edges(
        self,
        G: nx.MultiGraph,
        small_edge_limit: float,
        *,
        previous_n_edgepoint: Optional[int] = None,
    ) -> nx.MultiGraph:
        """
        Iteratively contract edges shorter than small_edge_limit (measured in k-space).
        """
        if small_edge_limit <= 0:
            return G

        nprev = self.previous_n_edgepoint if previous_n_edgepoint is None else int(previous_n_edgepoint)
        if nprev < 0:
            raise ValueError("previous_n_edgepoint must be >= 0.")

        # Loop until no more short edges exist
        while True:
            short_edges: list[tuple[float, Any, Any]] = []
            for u, v, data in G.edges(data=True):
                if u == v:
                    continue
                try:
                    pu = self._node_coord(G, u)
                    pv = self._node_coord(G, v)
                except Exception:
                    continue
                L = float(np.linalg.norm(pu - pv))
                if L < small_edge_limit:
                    short_edges.append((L, u, v))

            if not short_edges:
                break

            short_edges.sort(key=lambda t: t[0])  # smallest first
            _, u, v = short_edges[0]
            changed = self._contract_one_edge(G, u, v, previous_n_edgepoint=nprev)
            if not changed:
                # Safety: if something prevented contraction, drop that edge and continue.
                # (Should be rare.)
                try:
                    G.remove_edge(u, v)
                except Exception:
                    break

        return G

    # -------------------------
    # skeleton graph (with contraction option)
    # -------------------------
    def skeleton_graph(
        self,
        simplify: bool = True,
        smooth_epsilon: int = 4,
        *,
        skeleton_image: Optional[Union[NDArray, nx.MultiGraph]] = None,
        force_small_edge_contraction: Optional[bool] = None,
        small_edge_limit: Optional[float] = None,
        previous_n_edgepoint: Optional[int] = None,
    ) -> nx.MultiGraph:
        """
        Convert skeleton image to a graph, optionally simplify/smooth, and (optionally)
        contract short edges.

        Parameters
        ----------
        force_small_edge_contraction:
            If True, iteratively contract edges with length < small_edge_limit.
            If None, uses self.force_small_edge_contraction.

        small_edge_limit:
            k-space length threshold for contraction. If None, uses self.small_edge_limit.

        previous_n_edgepoint:
            Number of edge polyline points near the moved endpoint to replace with a smooth
            segment when contracting (if contraction is enabled). If None, uses self.previous_n_edgepoint.
        """
        force = self.force_small_edge_contraction if force_small_edge_contraction is None else bool(force_small_edge_contraction)
        limit = self.small_edge_limit if small_edge_limit is None else float(small_edge_limit)

        nprev = self.previous_n_edgepoint if previous_n_edgepoint is None else int(previous_n_edgepoint)
        if nprev < 0:
            raise ValueError("previous_n_edgepoint must be >= 0.")

        args = (smooth_epsilon, simplify, id(skeleton_image), force, limit, nprev)
        if self.skeleton_graph_cache is not None and self.skeleton_graph_cache_args == args:
            return self.skeleton_graph_cache

        # pick skeleton source
        if skeleton_image is None:
            skel = self._skeleton_image
            G = skeleton2graph(skel)
        elif isinstance(skeleton_image, (nx.Graph, nx.MultiGraph)):
            G = skeleton_image
        else:
            # assume ndarray mask
            G = skeleton2graph(skeleton_image)

        if simplify:
            G = remove_leaf_nodes(G)
            G = simplify_edges(G)

        if force and limit > 0:
            G = self._contract_small_edges(G, limit, previous_n_edgepoint=nprev)

        G = smooth_edges(G, epsilon=smooth_epsilon, copy=False)
        G.graph["is_trivalent"] = is_trivalent(G)
        self.is_graph_trivalent = G.graph["is_trivalent"]

        self.skeleton_graph_cache = G
        self.skeleton_graph_cache_args = args
        return G

    @property
    def total_edge_pts(self) -> int:
        return total_edge_pts(self.skeleton_graph_cache or self.skeleton_graph())

    def clear_cache(self):
        """Clears graph + PyVista caches (does not modify the Hamiltonian or grids)."""
        self.skeleton_graph_cache = None
        self.skeleton_graph_cache_args = None
        self._pv_data_args = None

        # clear cached_property values if they were computed
        for name in (
            "_skeleton_image",
            "eigvals_sorted",
            "spectrum",
            "band_gap",
            "fields_pv",
            "exceptional_surface_pv",
        ):
            self.__dict__.pop(name, None)
