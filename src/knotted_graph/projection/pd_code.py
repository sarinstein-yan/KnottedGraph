from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import shapely
import sympy as sp
from shapely import LineString, MultiLineString, MultiPoint, Point
from shapely.affinity import affine_transform
from shapely.ops import substring
from shapely.strtree import STRtree

from knotted_graph.core.embedding import ensure_embedding
from knotted_graph.invariants.yamada.polynomial import Yamada

from .geom import Arc, Crossing, Vertex
from .rotations import generate_isotopy_angles, get_rotation_matrix


__all__ = [
    "PDCode",
    "ProjectionResult",
    "YamadaComputationResult",
    "explode_to_segments",
    "find_all_crossings",
    "project_crossings_on_edge",
    "compute_pd_code",
    "compute_yamada_polynomial",
    "sample_projections",
    "select_projection",
]


@dataclass
class ProjectionResult:
    """A computed projection of a spatial graph into a planar diagram."""

    processor: "PDCode" = field(repr=False, compare=False)
    rotation_angles: tuple[float, float, float] | None
    rotation_order: str
    pd_code: str
    num_crossings: int

    @property
    def vertices(self) -> list[Vertex]:
        return list(self.processor.vertices.values())

    @property
    def crossings(self) -> list[Crossing]:
        return list(self.processor.crossings.values())

    @property
    def arcs(self) -> list[Arc]:
        return list(self.processor.arcs.values())


@dataclass
class YamadaComputationResult:
    """Yamada polynomial plus the projection used to compute it."""

    polynomial: sp.Expr
    projection: ProjectionResult


class PDCode:
    """Process an embedded spatial graph into a planar-diagram representation."""

    def __init__(self, skeleton_graph: nx.MultiGraph, tolerance: float = 1e-8):
        self.skeleton_graph = ensure_embedding(skeleton_graph, copy=True, normalize=True)
        self.tolerance = float(tolerance)
        self.vertices: Dict[int, Vertex] = {}
        self.crossings: Dict[int, Crossing] = {}
        self.arcs: Dict[int, Arc] = {}
        self.node_key_to_vertex_id = {}
        self.edge_key_to_index = {}
        self._cache = {}

    def compute(
        self,
        rotation_angles: Optional[tuple[float, float, float]] = None,
        rotation_order: str = "ZYX",
    ) -> str:
        """Compute a PD representation after an optional rigid rotation."""
        args = (tuple(rotation_angles or (0.0, 0.0, 0.0)), rotation_order)

        self.vertices = {}
        self.crossings = {}
        self.arcs = {}
        self.node_key_to_vertex_id = {}
        self.edge_key_to_index = {}
        Arc.reset_counter()

        node_points = MultiPoint(
            [Point(node_data["pos"]) for node_data in self.skeleton_graph.nodes.values()]
        )
        edge_lines = MultiLineString(
            [LineString(edge_data["pts"]) for edge_data in self.skeleton_graph.edges.values()]
        )

        if rotation_angles is not None:
            matrix = get_rotation_matrix(rotation_angles, rotation_order)
            rotation = matrix.ravel().tolist() + [0, 0, 0]
            node_points = affine_transform(node_points, rotation)
            edge_lines = affine_transform(edge_lines, rotation)

        self._initialize_vertices(node_points)
        crossing_points = self._find_all_crossings(
            edge_lines,
            tolerance=self.tolerance,
        )
        self._initialize_crossings(crossing_points)
        self._process_edges(edge_lines)
        self._determine_crossing_types()

        self._cache[args] = self._generate_pd_code()
        return self._cache[args]

    def _initialize_vertices(self, node_points: MultiPoint) -> None:
        for i, (node_key, _node_data) in enumerate(self.skeleton_graph.nodes.items()):
            vertex = Vertex(id=i, key=node_key, point=node_points.geoms[i])
            self.vertices[i] = vertex
            self.node_key_to_vertex_id[node_key] = i

    def _initialize_crossings(self, crossing_points: List[Point]) -> None:
        for i, point in enumerate(crossing_points):
            self.crossings[i] = Crossing(id=i, point=point)

    @staticmethod
    def _segment_z_at_xy(segment: LineString, point: Point) -> float:
        """Interpolate a 3-D segment's z coordinate at a projected XY point."""
        coords = np.asarray(segment.coords, dtype=float)
        p0 = coords[0]
        p1 = coords[-1]
        dxy = p1[:2] - p0[:2]
        denom = float(np.dot(dxy, dxy))
        if denom <= np.finfo(float).eps:
            raise ValueError("Projection contains a segment with zero XY extent")
        target = np.asarray((point.x, point.y), dtype=float)
        t = float(np.dot(target - p0[:2], dxy) / denom)
        t = min(1.0, max(0.0, t))
        return float(p0[2] + t * (p1[2] - p0[2]))

    @staticmethod
    def _is_true_spatial_contact(
        seg_a: LineString,
        seg_b: LineString,
        point: Point,
        *,
        tolerance: float,
    ) -> bool:
        """Return True when two projected strands actually meet in 3-D.

        This is the crucial distinction missing from the old endpoint-touch
        rejection.  Two segments may ``touch`` in XY because a genuine crossing
        lands exactly on an internal polyline sample point; if their interpolated
        z coordinates differ, that point is still a real diagram crossing.
        """
        z_a = PDCode._segment_z_at_xy(seg_a, point)
        z_b = PDCode._segment_z_at_xy(seg_b, point)
        return abs(z_a - z_b) <= tolerance

    @staticmethod
    def _find_all_crossings(
        multilines: MultiLineString,
        tolerance: float = 1e-8,
    ) -> List[Point]:
        """Find all proper projected crossings, including self-crossings.

        Segment endpoint contacts are no longer discarded wholesale.  A contact
        is ignored only when the two source segments coincide at the same 3-D
        point (ordinary polyline continuity or a true graph vertex).  If two
        strands share XY but have different heights, the point is a genuine
        over/under crossing even when it lies at an artificial sampling vertex.
        """
        segment_list = PDCode._explode_to_segments(multilines)
        if len(segment_list) < 2:
            return []

        segments = np.asarray(segment_list, dtype=object)
        tree = STRtree(segments)
        pairs = tree.query(segments)
        left = pairs[0].astype(np.intp, copy=False)
        right = pairs[1].astype(np.intp, copy=False)

        pair_mask = right > left
        left = left[pair_mask]
        right = right[pair_mask]
        if not len(left):
            return []

        seg_a = segments[left]
        seg_b = segments[right]
        intersections = shapely.intersection(seg_a, seg_b)

        seen: Set[Tuple[float, float]] = set()
        for a, b, inter in zip(seg_a, seg_b, intersections):
            if inter.is_empty:
                continue
            gtype = inter.geom_type
            if gtype.startswith("Line") or gtype == "GeometryCollection":
                raise ValueError("Found overlapping (colinear) projected segments")

            points: list[Point]
            if gtype == "Point":
                points = [inter]
            elif gtype == "MultiPoint":
                points = list(inter.geoms)
            else:
                continue

            for point in points:
                spatial_contact = PDCode._is_true_spatial_contact(
                    a,
                    b,
                    point,
                    tolerance=tolerance,
                )
                if spatial_contact:
                    # Same 3-D point: either neighbouring polyline pieces or a
                    # graph vertex.  It is not an over/under crossing.
                    continue
                seen.add((float(point.x), float(point.y)))

        return [Point(xy) for xy in sorted(seen)]

    @staticmethod
    def _explode_to_segments(lines: MultiLineString | LineString) -> list[LineString]:
        """Break every LineString into individual two-point 3-D segments."""
        line_geoms = [lines] if isinstance(lines, LineString) else list(lines.geoms)
        return [
            LineString([line.coords[i], line.coords[i + 1]])
            for line in line_geoms
            for i in range(len(line.coords) - 1)
        ]

    @staticmethod
    def _project_crossings_on_edge(
        edge: LineString,
        crossings: List[Point],
        tolerance: float = 1e-8,
    ) -> List[Tuple[float, int]]:
        """Return ``(distance_along_edge, crossing_id)`` incidences."""
        intersections = []
        coords = list(edge.coords)
        segment_start_dist = 0.0
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            for crossing_id, crossing_pt in enumerate(crossings):
                if segment.distance(crossing_pt) < tolerance:
                    dist_local = segment.project(crossing_pt)
                    intersections.append((segment_start_dist + dist_local, crossing_id))
            segment_start_dist += segment.length
        return PDCode._deduplicate_crossing_distances(intersections, tolerance=tolerance)

    @staticmethod
    def _deduplicate_crossing_distances(
        intersections: list[tuple[float, int]],
        *,
        tolerance: float,
    ) -> list[tuple[float, int]]:
        """Merge duplicate incidences created at internal polyline sample points."""
        if not intersections:
            return []
        ordered = sorted(intersections, key=lambda item: (item[1], item[0]))
        unique: list[tuple[float, int]] = []
        for distance, crossing_id in ordered:
            if unique:
                previous_distance, previous_id = unique[-1]
                if (
                    crossing_id == previous_id
                    and abs(distance - previous_distance) <= tolerance
                ):
                    unique[-1] = (
                        0.5 * (previous_distance + distance),
                        crossing_id,
                    )
                    continue
            unique.append((distance, crossing_id))
        return sorted(unique)

    @staticmethod
    def _project_crossings_on_edge_indexed(
        edge: LineString,
        crossings: List[Point],
        crossing_tree: STRtree,
        tolerance: float = 1e-8,
    ) -> List[Tuple[float, int]]:
        """Indexed equivalent of :meth:`_project_crossings_on_edge`."""
        intersections = []
        coords = list(edge.coords)
        segment_start_dist = 0.0
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            query_geometry = segment.buffer(tolerance)
            candidate_ids = sorted(int(idx) for idx in crossing_tree.query(query_geometry))
            for crossing_id in candidate_ids:
                crossing_pt = crossings[crossing_id]
                if segment.distance(crossing_pt) < tolerance:
                    dist_local = segment.project(crossing_pt)
                    intersections.append((segment_start_dist + dist_local, crossing_id))
            segment_start_dist += segment.length
        return PDCode._deduplicate_crossing_distances(
            intersections,
            tolerance=tolerance,
        )

    def _process_edges(self, edge_lines: MultiLineString) -> None:
        edge_keys = list(self.skeleton_graph.edges.keys())
        crossing_points = [crossing.point for crossing in self.crossings.values()]
        crossing_tree = STRtree(crossing_points) if crossing_points else None

        for i, (edge_line, edge_key) in enumerate(zip(edge_lines.geoms, edge_keys)):
            self.edge_key_to_index[edge_key] = i
            u, v, _key = edge_key
            start_vertex_id = self.node_key_to_vertex_id[u]
            end_vertex_id = self.node_key_to_vertex_id[v]

            if crossing_tree is None:
                intersections = []
            else:
                intersections = self._project_crossings_on_edge_indexed(
                    edge_line,
                    crossing_points,
                    crossing_tree,
                    tolerance=self.tolerance,
                )

            if not intersections:
                arc = Arc(
                    edge_key=edge_key,
                    line=edge_line,
                    start_type="v",
                    start_id=start_vertex_id,
                    end_type="v",
                    end_id=end_vertex_id,
                )
                self.arcs[arc.id] = arc
                self._update_incidences(arc)
                continue

            self._split_edge_at_crossings(
                edge_line,
                edge_key,
                start_vertex_id,
                end_vertex_id,
                intersections,
            )

    def _split_edge_at_crossings(
        self,
        edge: LineString,
        edge_key,
        start_vertex_id: int,
        end_vertex_id: int,
        intersections: List[Tuple[float, int]],
    ) -> None:
        cut_points = [0.0] + [dist for dist, _ in intersections] + [edge.length]
        crossing_ids = [None] + [cid for _, cid in intersections] + [None]

        for i in range(len(cut_points) - 1):
            start_dist = cut_points[i]
            end_dist = cut_points[i + 1]
            if end_dist - start_dist < self.tolerance:
                continue

            arc_line = substring(edge, start_dist, end_dist)
            if i == 0:
                start_type, start_id = "v", start_vertex_id
            else:
                start_type, start_id = "x", crossing_ids[i]

            if i == len(cut_points) - 2:
                end_type, end_id = "v", end_vertex_id
            else:
                end_type, end_id = "x", crossing_ids[i + 1]

            arc = Arc(
                edge_key=edge_key,
                line=arc_line,
                start_type=start_type,
                start_id=start_id,
                end_type=end_type,
                end_id=end_id,
            )
            self.arcs[arc.id] = arc
            self._update_incidences(arc)

    def _update_incidences(self, arc: Arc) -> None:
        def angle_from(base_point, other_coords):
            dx = other_coords[0] - base_point.x
            dy = other_coords[1] - base_point.y
            return float(np.arctan2(dy, dx))

        if arc.start_type == "v":
            vertex_pt = self.vertices[arc.start_id].point
            self.vertices[arc.start_id].add_incident_arc(
                arc.id,
                angle_from(vertex_pt, arc.line.coords[1]),
            )
        else:
            crossing_pt = self.crossings[arc.start_id].point
            self.crossings[arc.start_id].add_incident_arc(
                arc.id,
                angle_from(crossing_pt, arc.line.coords[1]),
            )

        if arc.end_type == "v":
            vertex_pt = self.vertices[arc.end_id].point
            self.vertices[arc.end_id].add_incident_arc(
                arc.id,
                angle_from(vertex_pt, arc.line.coords[-2]),
            )
        else:
            crossing_pt = self.crossings[arc.end_id].point
            self.crossings[arc.end_id].add_incident_arc(
                arc.id,
                angle_from(crossing_pt, arc.line.coords[-2]),
            )

    @staticmethod
    def _angular_distance(a: float, b: float) -> float:
        return abs(float(np.arctan2(np.sin(a - b), np.cos(a - b))))

    def _z_for_crossing_incidence(
        self,
        arc: Arc,
        xid: int,
        incidence_angle: float,
    ) -> float:
        """Return z for a specific half-edge incidence at a crossing.

        For an arc that starts and ends at the same crossing, the arc ID alone
        is insufficient; the incidence angle identifies which endpoint is meant.
        """
        crossing_pt = self.crossings[xid].point
        candidates: list[tuple[float, float]] = []

        if arc.start_type == "x" and arc.start_id == xid:
            coords = arc.line.coords
            angle = float(
                np.arctan2(
                    coords[1][1] - crossing_pt.y,
                    coords[1][0] - crossing_pt.x,
                )
            )
            candidates.append((angle, float(coords[0][2])))

        if arc.end_type == "x" and arc.end_id == xid:
            coords = arc.line.coords
            angle = float(
                np.arctan2(
                    coords[-2][1] - crossing_pt.y,
                    coords[-2][0] - crossing_pt.x,
                )
            )
            candidates.append((angle, float(coords[-1][2])))

        if not candidates:
            raise RuntimeError(f"Arc {arc.id} is not incident to crossing {xid}.")
        if len(candidates) == 1:
            return candidates[0][1]
        return min(
            candidates,
            key=lambda item: self._angular_distance(item[0], incidence_angle),
        )[1]

    def _determine_crossing_types(self) -> None:
        """Determine over/under strand from the four actual half-edge incidences."""
        for xid, crossing in self.crossings.items():
            assert len(crossing.incident_arcs) == 4, \
                "Crossing must have exactly 4 incidences."
            ordered = sorted(crossing.incident_arcs, key=lambda item: item[1])
            (arc_a_id, angle_a), (arc_b_id, angle_b), _, _ = ordered
            z_a = self._z_for_crossing_incidence(self.arcs[arc_a_id], xid, angle_a)
            z_b = self._z_for_crossing_incidence(self.arcs[arc_b_id], xid, angle_b)
            if abs(z_a - z_b) <= self.tolerance:
                raise ValueError(
                    "Nongeneric projection: crossing strands have indistinguishable "
                    f"heights at crossing {xid}."
                )
            crossing._correctly_overstrand = z_a > z_b

    def _generate_pd_code(self) -> str:
        v_parts = [vertex.pd_code for vertex in self.vertices.values() if vertex.pd_code]
        x_parts = [crossing.pd_code for crossing in self.crossings.values() if crossing.pd_code]
        if not v_parts and not x_parts:
            return ""
        return ";".join(v_parts + x_parts)

    @property
    def vertex_coords(self) -> List[Tuple[float, float]]:
        return [vertex.point.coords[0][:2] for vertex in self.vertices.values()]

    @property
    def crossing_coords(self) -> List[Tuple[float, float]]:
        return [crossing.point.coords[0][:2] for crossing in self.crossings.values()]

    @property
    def vertex_xy(self) -> Tuple[List[float], List[float]]:
        pts = np.array(self.vertex_coords)
        return pts[:, 0].tolist(), pts[:, 1].tolist()

    @property
    def crossing_xy(self) -> Tuple[List[float], List[float]]:
        pts = np.array(self.crossing_coords)
        return pts[:, 0].tolist(), pts[:, 1].tolist()

    def compute_yamada(
        self,
        variable: sp.Symbol,
        normalize: bool = True,
        n_jobs: int = -1,
        method: str = "negami",
    ) -> sp.Expr:
        if not self._cache:
            raise ValueError("PD code must be computed before Yamada polynomial.")
        computer = Yamada(
            vertices=list(self.vertices.values()),
            crossings=list(self.crossings.values()),
            arcs=list(self.arcs.values()),
        )
        return computer.compute(
            variable,
            normalize=normalize,
            n_jobs=n_jobs,
            method=method,
        )


explode_to_segments = PDCode._explode_to_segments
find_all_crossings = PDCode._find_all_crossings
project_crossings_on_edge = PDCode._project_crossings_on_edge


def compute_pd_code(
    skeleton_graph: nx.MultiGraph,
    rotation_angles: Optional[list[float]] = None,
    rotation_order: str = "ZYX",
) -> str:
    generator = PDCode(skeleton_graph)
    return generator.compute(
        rotation_angles=rotation_angles,
        rotation_order=rotation_order,
    )


def _normalize_rotation_angles(
    rotation_angles: Optional[list[float] | tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    if rotation_angles is None:
        return None
    if len(rotation_angles) != 3:
        raise ValueError("rotation_angles must contain exactly three values.")
    return tuple(float(angle) for angle in rotation_angles)


def _compute_projection(
    skeleton_graph: nx.MultiGraph,
    rotation_angles: tuple[float, float, float] | None,
    rotation_order: str,
) -> ProjectionResult:
    processor = PDCode(skeleton_graph)
    pd_code = processor.compute(
        rotation_angles=rotation_angles,
        rotation_order=rotation_order,
    )
    return ProjectionResult(
        processor=processor,
        rotation_angles=rotation_angles,
        rotation_order=rotation_order,
        pd_code=pd_code,
        num_crossings=len(processor.crossings),
    )


def sample_projections(
    skeleton_graph: nx.MultiGraph,
    *,
    num_rotation_samples: int = 10,
    rotation_order: str = "ZYX",
) -> list[ProjectionResult]:
    """Sample generic projections and return valid results in sample order."""
    if num_rotation_samples <= 0:
        raise ValueError("num_rotation_samples must be a positive integer.")
    skeleton_graph = ensure_embedding(skeleton_graph, copy=True, normalize=True)

    errors: list[str] = []
    projections: list[ProjectionResult] = []
    for sample_index, angles in enumerate(
        generate_isotopy_angles(num_rotation_samples, order=rotation_order)
    ):
        rotation_angles = tuple(float(angle) for angle in angles)
        try:
            projections.append(
                _compute_projection(skeleton_graph, rotation_angles, rotation_order)
            )
        except Exception as exc:
            errors.append(f"sample {sample_index}: {exc}")

    if not projections:
        details = "; ".join(errors) if errors else "no samples were generated"
        raise RuntimeError(f"All projection samples failed: {details}")
    return projections


def select_projection(
    skeleton_graph: nx.MultiGraph,
    *,
    rotation_angles: Optional[list[float] | tuple[float, float, float]] = None,
    rotation_order: str = "ZYX",
    num_rotation_samples: int = 10,
) -> ProjectionResult:
    """Select an explicit projection or the valid sample with fewest crossings."""
    skeleton_graph = ensure_embedding(skeleton_graph, copy=True, normalize=True)
    exact_angles = _normalize_rotation_angles(rotation_angles)
    if exact_angles is not None:
        return _compute_projection(skeleton_graph, exact_angles, rotation_order)

    projections = sample_projections(
        skeleton_graph,
        num_rotation_samples=num_rotation_samples,
        rotation_order=rotation_order,
    )
    return min(
        enumerate(projections),
        key=lambda indexed: (indexed[1].num_crossings, indexed[0]),
    )[1]


def compute_yamada_polynomial(
    skeleton_graph: nx.MultiGraph,
    variable: sp.Symbol,
    rotation_angles: Optional[list[float]] = None,
    rotation_order: str = "ZYX",
    num_rotation_samples: int = 10,
    crossing_warning_threshold: int = 10,
    normalize: bool = True,
    n_jobs: int = -1,
    method: str = "negami",
    return_result: bool = False,
) -> sp.Expr | YamadaComputationResult:
    """Compute the Yamada polynomial from the simplest valid projection."""
    projection = select_projection(
        skeleton_graph,
        rotation_angles=rotation_angles,
        rotation_order=rotation_order,
        num_rotation_samples=num_rotation_samples,
    )
    if (
        crossing_warning_threshold is not None
        and projection.num_crossings >= crossing_warning_threshold
    ):
        warnings.warn(
            "Selected planar diagram has "
            f"{projection.num_crossings} crossings; Yamada computation may be expensive.",
            RuntimeWarning,
            stacklevel=2,
        )

    polynomial = projection.processor.compute_yamada(
        variable,
        normalize=normalize,
        n_jobs=n_jobs,
        method=method,
    )
    if return_result:
        return YamadaComputationResult(polynomial=polynomial, projection=projection)
    return polynomial
