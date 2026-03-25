"""Planar diagram refinement and plotting utilities.

This module provides a :class:`PlanarDiagram` helper that refines the layout of
an already-projected :class:`~knotted_graph.yamada.pd_code.PDCode`. The class
performs a lightweight ambient-isotopic massage of the diagram using a
force-directed relaxation while validating that the cyclic ordering of arcs at
each node is preserved and that no spurious crossings are introduced. The
resulting geometry is rendered with smooth spline-like curves.

Compared to the basic version, this implementation also samples a configurable
number of *edge control points* per arc. These interior particles are connected
by springs and interact via a discrete tangent–point repulsion energy so that
entire edge segments, not only endpoints, participate in the relaxation. This
significantly reduces the chance that the subsequent Bezier smoothing
introduces spurious intersections that would break isotopy.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString
from shapely.ops import substring

from .geom import Arc
from .pd_code import PDCode

__all__ = ["PlanarDiagram"]


@dataclass
class _NodeRef:
    """Lightweight reference to a PDCode node (vertex or crossing).

    kind
        'v' for a vertex, 'x' for a crossing.
    idx
        Integer ID on the PDCode side.
    """

    kind: str  # 'v' or 'x'
    idx: int

    @property
    def key(self) -> str:
        return f"{self.kind}{self.idx}"


class PlanarDiagram:
    """Refine and draw planar diagrams derived from :class:`PDCode`.

    Parameters
    ----------
    pd : PDCode
        Pre-computed planar diagram.
    spring_constant : float, optional
        Attraction strength along arcs during relaxation. Defaults to 0.08.
    nodes_repulsion : float, optional
        Isotropic repulsion strength between all particles (nodes and edge
        control points). Defaults to 0.12.
    control_points_per_edge : int, optional
        Number of interior control points sampled along each arc's initial
        :class:`shapely.LineString` using
        ``line.interpolate(..., normalized=True)``. These points are treated as
        additional particles connected by springs to the endpoints, so each arc
        becomes a polyline chain. Set to 0 to disable interior control points.
        Defaults to 3.
    tangent_repulsion : float, optional
        Strength of the discrete tangent–point repulsion between edge control
        points belonging to different arcs. Larger values push parallel and
        nearly-touching segments apart more aggressively. Defaults to 0.08.
    """

    def __init__(
        self,
        pd: PDCode,
        *,
        spring_constant: float = 0.08,
        nodes_repulsion: float = 0.12,
        control_points_per_edge: int = 3,
        tangent_repulsion: float = 0.08,
    ) -> None:
        self.pd = pd
        self.spring_constant = spring_constant
        self.repulsion = nodes_repulsion
        self.control_points_per_edge = int(control_points_per_edge)
        self.tangent_repulsion = float(tangent_repulsion)

        # Positions for all particles (vertices, crossings, and edge controls).
        self._positions: Dict[str, np.ndarray] = self._initial_positions()
        # Cyclic ordering of incident arcs at each vertex / crossing.
        self._orders: Dict[str, List[int]] = self._capture_cyclic_orders()
        # Map PDCode arc IDs to their endpoint node refs.
        self._arc_nodes: Dict[int, Tuple[_NodeRef, _NodeRef]] = self._map_arcs_to_nodes()
        # Typical arc length scale (used e.g. as a fallback).
        self._ideal_length: float = self._estimate_ideal_length()
        # Edge control points + per-arc polyline node chains.
        (
            self._edge_points,
            self._arc_poly_nodes,
        ) = self._init_edge_control_points()
        # Rest length for each spring segment along each arc polyline.
        self._segment_rest_lengths: Dict[Tuple[int, int], float] = (
            self._estimate_segment_rest_lengths()
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _initial_positions(self) -> Dict[str, np.ndarray]:
        """Initial positions for PDCode vertices and crossings."""

        positions: Dict[str, np.ndarray] = {}
        for v in self.pd.vertices.values():
            positions[_NodeRef("v", v.id).key] = np.array(
                v.point.coords[0][:2], dtype=float
            )
        for x in self.pd.crossings.values():
            positions[_NodeRef("x", x.id).key] = np.array(
                x.point.coords[0][:2], dtype=float
            )
        return positions

    def _capture_cyclic_orders(self) -> Dict[str, List[int]]:
        """Capture target cyclic ordering of incident arcs at each node."""

        orders: Dict[str, List[int]] = {}
        for v in self.pd.vertices.values():
            orders[_NodeRef("v", v.id).key] = list(v.ccw_ordered_arcs)
        for x in self.pd.crossings.values():
            raw = list(getattr(x, "_raw_ccw_ordered_arcs", []))
            orders[_NodeRef("x", x.id).key] = raw if raw else list(x.ccw_ordered_arcs)
        return orders

    def _map_arcs_to_nodes(self) -> Dict[int, Tuple[_NodeRef, _NodeRef]]:
        """Map each arc to its start / end PDCode nodes."""

        mapping: Dict[int, Tuple[_NodeRef, _NodeRef]] = {}
        for arc in self.pd.arcs.values():
            start = _NodeRef(arc.start_type, arc.start_id)
            end = _NodeRef(arc.end_type, arc.end_id)
            mapping[arc.id] = (start, end)
        return mapping

    def _estimate_ideal_length(self) -> float:
        """Median arc length as a global length scale."""

        if not self.pd.arcs:
            return 1.0
        lengths = [a.line.length for a in self.pd.arcs.values()]
        return float(np.median(lengths))

    # ------------------------------------------------------------------
    # Edge control points & per-arc polylines
    # ------------------------------------------------------------------
    def _init_edge_control_points(
        self,
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
        """Create interior control points along each arc.

        Returns
        -------
        edge_points
            Mapping ``arc_id -> [key_e0, key_e1, ...]`` of control point
            identifiers for that arc.
        arc_poly_nodes
            Mapping ``arc_id -> [node_key0, node_key1, ...]`` of the full
            polyline node chain for that arc, including endpoints and interior
            control points.
        """

        edge_points: Dict[int, List[str]] = {}
        arc_poly_nodes: Dict[int, List[str]] = {}

        # If disabled, each arc is just a straight segment between endpoints.
        if self.control_points_per_edge <= 0:
            for arc_id, (s, t) in self._arc_nodes.items():
                edge_points[arc_id] = []
                arc_poly_nodes[arc_id] = [s.key, t.key]
            return edge_points, arc_poly_nodes

        for arc in self.pd.arcs.values():
            start_ref, end_ref = self._arc_nodes[arc.id]
            poly_nodes: List[str] = [start_ref.key]
            ctrl_keys: List[str] = []

            line = arc.line  # shapely.LineString in the initial embedding
            n_ctrl = self.control_points_per_edge
            for i in range(n_ctrl):
                # Equidistant interior points along the original edge geometry
                # (excluding end points) using a normalized arc-length
                # parameterization.
                frac = (i + 1) / (n_ctrl + 1)
                p = line.interpolate(frac, normalized=True)
                key = f"e{arc.id}_{i}"
                self._positions[key] = np.array(p.coords[0][:2], dtype=float)
                ctrl_keys.append(key)
                poly_nodes.append(key)

            poly_nodes.append(end_ref.key)
            edge_points[arc.id] = ctrl_keys
            arc_poly_nodes[arc.id] = poly_nodes

        return edge_points, arc_poly_nodes

    def _estimate_segment_rest_lengths(self) -> Dict[Tuple[int, int], float]:
        """Rest length for each polyline segment along each arc.

        We distribute the original arc length equally over its polyline
        segments so that the relaxed configuration roughly preserves the
        original edge lengths while allowing some re-distribution of curvature.
        """

        rest: Dict[Tuple[int, int], float] = {}
        global_scale = self._ideal_length if self._ideal_length > 0 else 1.0

        for arc in self.pd.arcs.values():
            keys = self._arc_poly_nodes[arc.id]
            seg_count = max(len(keys) - 1, 1)
            base_len = float(arc.line.length)
            if base_len <= 0:
                # Fallback to global scale if the original geometry is
                # degenerate.
                base_len = global_scale
            seg_rest = base_len / seg_count
            for j in range(seg_count):
                rest[(arc.id, j)] = seg_rest
        return rest

    # ------------------------------------------------------------------
    # Local ordering & intersection checks
    # ------------------------------------------------------------------
    def _order_from_positions(
        self, node: _NodeRef, positions: Dict[str, np.ndarray]
    ) -> List[int]:
        """Compute the ccw ordering of incident arcs at ``node``.

        This uses the *endpoint* positions only and ignores interior control
        points, which is sufficient because the PDCode connectivity lives only
        on vertices and crossings.
        """

        incident: List[int] = []
        for arc_id, (s, t) in self._arc_nodes.items():
            if s.key == node.key or t.key == node.key:
                incident.append(arc_id)

        center = positions[node.key]
        angles: List[Tuple[int, float]] = []
        for arc_id in incident:
            s, t = self._arc_nodes[arc_id]
            other = t if s.key == node.key else s
            vec = positions[other.key] - center
            angle = math.atan2(vec[1], vec[0])
            angles.append((arc_id, angle))

        return [aid for aid, _ in sorted(angles, key=lambda x: x[1])]

    # BUG: check against the /yamada module. What constitutes equivalence?
    @staticmethod
    def _cyclically_equivalent(target: List[int], current: List[int]) -> bool:
        """Return True if ``current`` is a cyclic rotation of ``target``."""

        if len(target) != len(current):
            return False
        if not target:
            return True
        n = len(target)
        doubled = current * 2
        for i in range(n):
            if doubled[i : i + n] == target:
                return True
        return False

    def _preserves_orders(self, positions: Dict[str, np.ndarray]) -> bool:
        """Check whether the cyclic ordering at each node is unchanged."""

        for node_key, target in self._orders.items():
            if not target:
                continue
            node = _NodeRef(node_key[0], int(node_key[1:]))
            current = self._order_from_positions(node, positions)
            if not self._cyclically_equivalent(target, current):
                return False
        return True

    def _edges_cross(self, positions: Dict[str, np.ndarray]) -> bool:
        """Detect spurious edge–edge intersections.

        We approximate each arc by the polyline chain defined by its endpoints
        and interior control points, then test for intersections between
        segments that do not share endpoints. Segments that meet at a vertex or
        designed crossing share a node key and are therefore ignored by the
        intersection test.
        """

        segments: List[Tuple[Tuple[int, int], LineString, set[str]]] = []
        for arc in self.pd.arcs.values():
            keys = self._arc_poly_nodes[arc.id]
            for j in range(len(keys) - 1):
                u, v = keys[j], keys[j + 1]
                p0, p1 = positions[u], positions[v]
                seg = LineString([p0, p1])
                segments.append(((arc.id, j), seg, {u, v}))

        for i, (seg_id_a, seg_a, nodes_a) in enumerate(segments):
            for seg_id_b, seg_b, nodes_b in segments[i + 1 :]:
                # Ignore segments that share an endpoint (either within the
                # same arc or at a designed vertex / crossing).
                if nodes_a & nodes_b:
                    continue
                if seg_a.crosses(seg_b):
                    return True
        return False

    # ------------------------------------------------------------------
    # Tangent–point style repulsion between edge segments
    # ------------------------------------------------------------------
    def _tangent_samples(
        self, positions: Dict[str, np.ndarray]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Compute positions and tangents at edge control points for each arc.

        Returns
        -------
        samples
            Mapping ``arc_id -> {"keys", "pos", "tan"}`` where

            * ``keys`` is a list of control point keys for that arc
            * ``pos`` is an ``(M, 2)`` array of their positions
            * ``tan`` is an ``(M, 2)`` array of unit tangent vectors
        """

        samples: Dict[int, Dict[str, np.ndarray]] = {}
        for arc in self.pd.arcs.values():
            ctrl_keys = self._edge_points.get(arc.id, [])
            if not ctrl_keys:
                continue

            poly_keys = self._arc_poly_nodes[arc.id]
            # Map control keys to indices along the full polyline.
            idx_map = {k: poly_keys.index(k) for k in ctrl_keys}

            pos_list: List[np.ndarray] = []
            tan_list: List[np.ndarray] = []

            for key in ctrl_keys:
                idx = idx_map[key]
                # Finite-difference tangent based on neighbours along the
                # polyline.
                if 0 < idx < len(poly_keys) - 1:
                    prev_key = poly_keys[idx - 1]
                    next_key = poly_keys[idx + 1]
                    t_vec = positions[next_key] - positions[prev_key]
                elif idx == 0 and len(poly_keys) > 1:
                    t_vec = positions[poly_keys[1]] - positions[poly_keys[0]]
                else:  # idx == len(poly_keys) - 1 or degenerate
                    t_vec = positions[poly_keys[-1]] - positions[poly_keys[-2]]

                p = positions[key]
                pos_list.append(p)

                norm = float(np.linalg.norm(t_vec))
                if norm > 0:
                    tan_list.append(t_vec / norm)
                else:
                    tan_list.append(np.array([1.0, 0.0], dtype=float))

            samples[arc.id] = {
                "keys": np.array(ctrl_keys, dtype=object),
                "pos": np.stack(pos_list, axis=0),
                "tan": np.stack(tan_list, axis=0),
            }
        return samples

    def _apply_tangent_repulsion(
        self,
        positions: Dict[str, np.ndarray],
        forces: Dict[str, np.ndarray],
    ) -> None:
        """Apply a discrete tangent–point repulsion between edge segments.

        The implementation is a simplified discrete analogue of the
        tangent–point energies used in repulsive curve flows. For each pair of
        edge control points ``(p_i, p_j)`` on *different* arcs, we compute the
        distance from ``p_j`` to the tangent line at ``p_i`` and apply a
        repulsive force on both control points along the perpendicular
        direction. This helps to push nearly-parallel, nearly-touching segments
        apart while leaving distant segments essentially unaffected.
        """

        if self.tangent_repulsion <= 0.0 or self.control_points_per_edge <= 0:
            return

        samples = self._tangent_samples(positions)
        arc_ids = list(samples.keys())
        if len(arc_ids) < 2:
            return

        eps = 1e-12

        for ia, arc_id_a in enumerate(arc_ids):
            data_a = samples[arc_id_a]
            keys_a = data_a["keys"]
            pos_a = data_a["pos"]
            tan_a = data_a["tan"]

            for arc_id_b in arc_ids[ia + 1 :]:
                data_b = samples[arc_id_b]
                keys_b = data_b["keys"]
                pos_b = data_b["pos"]

                for idx_a, key_a in enumerate(keys_a):
                    p_a = pos_a[idx_a]
                    t_a = tan_a[idx_a]

                    for idx_b, key_b in enumerate(keys_b):
                        p_b = pos_b[idx_b]
                        d = p_b - p_a
                        dist2 = float(np.dot(d, d)) + eps
                        if dist2 <= eps:
                            continue

                        # Component of d perpendicular to the tangent at p_a.
                        proj = float(np.dot(d, t_a)) * t_a
                        perp = d - proj
                        perp2 = float(np.dot(perp, perp)) + eps
                        perp_norm = math.sqrt(perp2)

                        # Tangent–point style force magnitude. The scaling
                        # ``1 / (perp2 * dist2)`` is a simple, rapidly-decaying
                        # surrogate for the more complicated analytic
                        # expressions in the continuous theory.
                        f_mag = self.tangent_repulsion / (perp2 * dist2)
                        force_vec = f_mag * (perp / perp_norm)

                        # Equal and opposite forces on the two control points.
                        forces[key_a] -= force_vec
                        forces[key_b] += force_vec

    # ------------------------------------------------------------------
    # Relaxation
    # ------------------------------------------------------------------
    def _relax_once(self, step: float) -> Tuple[Dict[str, np.ndarray], float]:
        """Single relaxation step.

        The particle set consists of:

        * one node per PDCode vertex / crossing, and
        * ``control_points_per_edge`` interior nodes per arc.

        Forces are the sum of:

        * isotropic repulsion between all particles,
        * logarithmic springs along every arc-polyline segment, and
        * a discrete tangent–point repulsion between edge control points on
          different arcs.
        """

        positions = self._positions
        nodes = list(positions.keys())
        forces: Dict[str, np.ndarray] = {n: np.zeros(2, dtype=float) for n in nodes}

        # Repulsive forces between all particles (O(N^2), but N is modest for
        # typical diagrams).
        for i, n1 in enumerate(nodes):
            p1 = positions[n1]
            for n2 in nodes[i + 1 :]:
                p2 = positions[n2]
                delta = p1 - p2
                dist_sq = float(np.dot(delta, delta)) + 1e-6
                dist = math.sqrt(dist_sq)
                if dist <= 0.0:
                    continue
                force_mag = self.repulsion / dist_sq
                direction = delta / dist
                f_vec = force_mag * direction
                forces[n1] += f_vec
                forces[n2] -= f_vec

        # Attractive forces along each arc polyline (springs).
        for arc in self.pd.arcs.values():
            keys = self._arc_poly_nodes[arc.id]
            if len(keys) < 2:
                continue
            for j in range(len(keys) - 1):
                u, v = keys[j], keys[j + 1]
                delta = positions[v] - positions[u]
                dist = float(np.linalg.norm(delta)) + 1e-6
                direction = delta / dist
                target = self._segment_rest_lengths.get(
                    (arc.id, j), self._ideal_length / max(len(keys) - 1, 1)
                )
                # Logarithmic spring (acts like Hooke's law near equilibrium
                # but remains well-behaved for large distortions).
                force_mag = self.spring_constant * math.log(dist / target)
                f_vec = force_mag * direction
                forces[u] += f_vec
                forces[v] -= f_vec

        # Tangent–point repulsion between edge segments.
        self._apply_tangent_repulsion(positions, forces)

        # Euler step.
        new_positions = {n: positions[n] + step * forces[n] for n in nodes}
        loss = float(sum(np.linalg.norm(forces[n]) for n in nodes))
        return new_positions, loss

    def layout(
        self,
        *,
        max_iter: int = 200,
        step: float = 1.0,
        min_step: float = 1e-4,
        preserve_orders: bool = False,
    ) -> None:
        """Iteratively relax node positions while maintaining isotopy.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of relaxation max_iter. Defaults to 200.
        step : float, optional
            Initial step size for the explicit Euler integrator. The step is
            adaptively halved whenever a candidate update would introduce edge
            crossings (or, optionally, violate local cyclic orders). Defaults
            to 1.0.
        min_step : float, optional
            Minimum allowed step size; once the step is reduced below this
            threshold the relaxation terminates. Defaults to 1e-4.
        preserve_orders : bool, optional
            If True, enforce the original cyclic ordering of arcs at each
            vertex / crossing. This is conservative but somewhat more
            expensive; by default we only enforce the topological constraint
            via edge–edge intersection checks. Defaults to False.
        """

        logging.info("[PlanarDiagram] Starting layout refinement...")
        for it in range(max_iter):
            logging.info(
                f"[PlanarDiagram]  iteration {it + 1}/{max_iter} starts...",
                end="\r",
            )
            candidate, loss = self._relax_once(step)

            # Isotopy safeguard: reject steps that introduce crossings or, if
            # requested, change the local cyclic ordering at any node.
            violates_orders = preserve_orders and (not self._preserves_orders(candidate))
            if violates_orders or self._edges_cross(candidate):
                step *= 0.5
                if step < min_step:
                    break
                continue

            self._positions = candidate
            if loss < 1e-6:
                break

        logging.info(f"[PlanarDiagram]  iteration {it + 1}/{max_iter} done.          ")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _tangents_from_arc(self, arc: Arc) -> Tuple[np.ndarray, np.ndarray]:
        """Tangent directions at the start and end of an arc.

        We use the *current* polyline geometry defined by the particle
        positions so that the subsequent Bezier smoothing follows the relaxed
        layout, not the original PDCode line geometry.
        """

        keys = self._arc_poly_nodes[arc.id]
        coords = np.array([self._positions[k] for k in keys], dtype=float)
        if len(coords) < 2:
            # Degenerate; fall back to a dummy direction.
            return np.array([1.0, 0.0], dtype=float), np.array([-1.0, 0.0], dtype=float)
        if len(coords) == 2:
            start_dir = coords[1] - coords[0]
            end_dir = coords[0] - coords[1]
        else:
            start_dir = coords[1] - coords[0]
            end_dir = coords[-2] - coords[-1]
        return start_dir, end_dir

    def _bezier_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        start_dir: np.ndarray,
        end_dir: np.ndarray,
        samples: int = 60,
    ) -> np.ndarray:
        """Cubic Bezier interpolation between two endpoints + tangents."""

        base = float(np.linalg.norm(end - start))
        control_scale = 0.35 * base

        if np.linalg.norm(start_dir) > 0:
            start_ctrl = start + control_scale * (
                start_dir / np.linalg.norm(start_dir)
            )
        else:
            start_ctrl = start

        if np.linalg.norm(end_dir) > 0:
            end_ctrl = end + control_scale * (end_dir / np.linalg.norm(end_dir))
        else:
            end_ctrl = end

        t = np.linspace(0.0, 1.0, samples)
        one_minus = 1.0 - t
        points = (
            (one_minus**3)[:, None] * start
            + 3.0 * (one_minus**2)[:, None] * t[:, None] * start_ctrl
            + 3.0 * one_minus[:, None] * (t**2)[:, None] * end_ctrl
            + (t**3)[:, None] * end
        )
        return points

    def _curve_for_arc(self, arc: Arc, bezier_samples: int = 60) -> LineString:
        """Smooth Bezier curve for an arc in its current layout."""

        start_node, end_node = self._arc_nodes[arc.id]
        start = self._positions[start_node.key]
        end = self._positions[end_node.key]
        start_dir, end_dir = self._tangents_from_arc(arc)
        pts = self._bezier_points(start, end, start_dir, end_dir, bezier_samples)
        return LineString(pts)

    def _undercrossing_flags(self) -> Dict[int, Tuple[bool, bool]]:
        """Return flags indicating which arc endpoints are under-crossings."""

        flags: Dict[int, Tuple[bool, bool]] = defaultdict(lambda: [False, False])
        for x in self.pd.crossings.values():
            if not x.ccw_ordered_arcs:
                continue
            for uid in [x.ccw_ordered_arcs[i] for i in (1, 3)]:
                arc = self.pd.arcs[uid]
                if arc.start_type == "x" and arc.start_id == x.id:
                    flags[uid][0] = True
                if arc.end_type == "x" and arc.end_id == x.id:
                    flags[uid][1] = True
        return {k: (v[0], v[1]) for k, v in flags.items()}

    # ------------------------------------------------------------------
    # Public plotting API
    # ------------------------------------------------------------------
    def plot(
        self,
        ax: plt.Axes | None = None,
        edge_kwargs: Dict | None = None,
        vertex_kwargs: Dict | None = None,
        crossing_kwargs: Dict | None = None,
        undercrossing_offset: float = 5.0,
        mark_crossings: bool = False,
        bezier_samples: int = 50,
    ) -> plt.Axes:
        """Plot the refined planar diagram with spline-smoothed arcs.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        edge_kwargs : dict, optional
            Keyword arguments forwarded to :meth:`Axes.plot` for edges.
        vertex_kwargs : dict, optional
            Keyword arguments forwarded to :meth:`Axes.scatter` for vertices.
        crossing_kwargs : dict, optional
            Keyword arguments forwarded to :meth:`Axes.scatter` for crossings
            when ``mark_crossings`` is True.
        undercrossing_offset : float, optional
            Geodesic offset (in data units) to trim under-crossing segments
            back from the crossing point to create the standard under/over
            visual. Defaults to 5.0.
        mark_crossings : bool, optional
            If True, draw markers at crossing positions. Defaults to False.
        bezier_samples : int, optional
            Number of samples used per Bezier curve when rendering edges.
            Defaults to 60.
        """

        edge_kwargs = {"color": "tab:blue", "linewidth": 1.6, "zorder": -1, **(edge_kwargs or {})}
        vertex_kwargs = {"s": 20, "marker": "o", "color": "tab:red", **(vertex_kwargs or {})}
        crossing_kwargs = {"s": 35, "marker": "x", "color": "k", **(crossing_kwargs or {})}

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        under_flags = self._undercrossing_flags()
        curves: Dict[int, LineString] = {
            arc.id: self._curve_for_arc(arc, bezier_samples)
            for arc in self.pd.arcs.values()
        }

        # Draw edges with small trims at under-crossings.
        for arc_id, curve in curves.items():
            start_under, end_under = under_flags.get(arc_id, (False, False))
            t = undercrossing_offset
            L = curve.length
            line = curve
            if start_under and end_under and L > 2 * t:
                line = substring(curve, t, L - t)
            elif start_under and L > t:
                line = substring(curve, t, L)
            elif end_under and L > t:
                line = substring(curve, 0.0, L - t)
            ax.plot(*line.xy, **edge_kwargs)

        # Draw vertices.
        for v in self.pd.vertices.values():
            pos = self._positions[_NodeRef("v", v.id).key]
            ax.scatter(pos[0], pos[1], **vertex_kwargs)

        # Optionally mark crossings.
        if mark_crossings:
            for x in self.pd.crossings.values():
                pos = self._positions[_NodeRef("x", x.id).key]
                ax.scatter(pos[0], pos[1], **crossing_kwargs)

        # ax.set_aspect("equal", adjustable="datalim")
        # ax.axis("off")
        return ax