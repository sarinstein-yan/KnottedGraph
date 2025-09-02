import numpy as np
import sympy as sp
import networkx as nx
from shapely import Point, MultiPoint, LineString, MultiLineString
from shapely.ops import substring
from shapely.affinity import affine_transform
from shapely.strtree import STRtree
from typing import List, Tuple, Dict, Set, Optional
from .geom import Vertex, Crossing, Arc
from .util import get_rotation_matrix
from .polynomial import Yamada


__all__ = [
    "PDCode",
    "explode_to_segments",
    "find_all_crossings",
    "project_crossings_on_edge",
    "compute_pd_code",
    "compute_yamada_polynomial"
]


class PDCode:
    """Process a knotted graph to generate PD codes."""
    def __init__(self, skeleton_graph: nx.MultiGraph, tolerance: float = 1e-8):
        self.skeleton_graph = skeleton_graph
        self.tolerance = tolerance
        self.vertices: Dict[int, Vertex] = {}
        self.crossings: Dict[int, Crossing] = {}
        self.arcs: Dict[int, Arc] = {}
        self.node_key_to_vertex_id = {} # Map original node keys to vertex IDs
        self.edge_key_to_index = {} # Map original edge keys to edge indices
        self._cache = {}

    def compute(
            self, 
            rotation_angles: Optional[tuple[float]] = None,
            rotation_order: str = 'ZYX',
    ) -> str:
        """Compute the graph with optional rotation."""
        args = (tuple(rotation_angles or (0.,0.,0.)), rotation_order)
        if args in self._cache:
            return self._cache[args]
        
        # Clear Arc counter
        Arc.reset_counter()
        # Create shapely geometries
        node_points = MultiPoint([
            Point(node_data['pos']) 
            for node_data in self.skeleton_graph.nodes.values()
        ])
        edge_lines = MultiLineString([
            LineString(edge_data['pts']) 
            for edge_data in self.skeleton_graph.edges.values()
        ])

        # Apply rotation if provided
        if rotation_angles is not None:
            matrix = get_rotation_matrix(rotation_angles, rotation_order)
            rotation = matrix.ravel().tolist() + [0, 0, 0]
            node_points = affine_transform(node_points, rotation)
            edge_lines = affine_transform(edge_lines, rotation)

        # Initialize vertices from nodes
        self._initialize_vertices(node_points)
        
        # Find all crossings
        crossing_points = self._find_all_crossings(edge_lines)
        self._initialize_crossings(crossing_points)
        
        # Process each edge and split at crossings
        self._process_edges(edge_lines)

        # Determine crossing types (Over/Under)
        self._determine_crossing_types()

        self._cache[args] = self._generate_pd_code()
        return self._cache[args]

    def _initialize_vertices(self, node_points: MultiPoint):
        """Initialize vertices from graph nodes."""
        for i, (node_key, node_data) in enumerate(self.skeleton_graph.nodes.items()):
            vertex = Vertex(
                id=i,
                key=node_key,
                point=node_points.geoms[i]
            )
            self.vertices[i] = vertex
            self.node_key_to_vertex_id[node_key] = i
    

    def _initialize_crossings(self, crossing_points: List[Point]):
        """Initialize crossings from detected crossing points."""
        for i, point in enumerate(crossing_points):
            crossing = Crossing(id=i, point=point)
            self.crossings[i] = crossing
    

    @staticmethod
    def _find_all_crossings(multilines: MultiLineString) -> List[Point]:
        """Find all crossing points, including self-crossings."""
        segments = PDCode._explode_to_segments(multilines)
        tree = STRtree(segments)
        seen: Set[Tuple[float, float]] = set()
        
        for i, seg in enumerate(segments):
            for idx in tree.query(seg):
                other_seg = tree.geometries.take(idx)
                # Skip segments that are visited before or connect at endpoints
                if segments.index(other_seg) <= i or seg.touches(other_seg):
                    continue
                
                inter = seg.intersection(other_seg)
                if inter.is_empty:
                    continue
                
                gtype = inter.geom_type
                if gtype.startswith("Line") or gtype == "GeometryCollection":
                    raise ValueError("Found overlapping (colinear) segments")
                
                if gtype == "Point":
                    seen.add((inter.x, inter.y))
                elif gtype == "MultiPoint":
                    for p in inter.geoms:
                        seen.add((p.x, p.y))
        
        return [Point(xy) for xy in seen]
    

    @staticmethod
    def _explode_to_segments(lines: MultiLineString | LineString) -> list[LineString]:
        """Break every LineString into individual 2-point segments."""
        # Normalize: treat a single LineString as a one-element collection
        line_geoms = [lines] if isinstance(lines, LineString) else list(lines.geoms)
        return [
            LineString([line.coords[i], line.coords[i+1]])
            for line in line_geoms
            for i in range(len(line.coords) - 1)
        ]
    

    @staticmethod
    def _project_crossings_on_edge(
            edge: LineString, 
            crossings: List[Point],
            tolerance: float = 1e-8
        ) -> List[Tuple[float, int]]:
        """
        Find all intersections on an edge, handling self-crossings properly.
        Returns list of (distance_along_edge, crossing_id) tuples.
        """
        intersections = []
        coords = list(edge.coords)
        segment_start_dist = 0.
        # Check on each segment
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            # Check each crossing point against this segment
            for crossing_id, crossing_pt in enumerate(crossings):
                if segment.distance(crossing_pt) < tolerance:
                    dist_local = segment.project(crossing_pt)
                    total_dist = segment_start_dist + dist_local
                    intersections.append((total_dist, crossing_id))
            segment_start_dist += segment.length
        # Sort by distance and remove duplicates
        return sorted(set(intersections))


    def _process_edges(self, edge_lines: MultiLineString):
        """Process all edges, splitting at crossings."""
        # get edge keys and their indices
        edge_keys = list(self.skeleton_graph.edges.keys())
        crossing_list = list(self.crossings.values())
        crossing_points = [c.point for c in crossing_list]
        
        for i, (edge_line, edge_key) in enumerate(zip(edge_lines.geoms, edge_keys)):
            self.edge_key_to_index[edge_key] = i
            
            # Get start and end vertices using the edge key
            u, v, k = edge_key
            start_vertex_id = self.node_key_to_vertex_id[u]
            end_vertex_id = self.node_key_to_vertex_id[v]
            
            # Find all crossings on this edge
            intersections = PDCode._project_crossings_on_edge(
                edge_line, crossing_points, tolerance=self.tolerance
            )
            
            if not intersections:
                # No crossings - single arc from u to v
                arc = Arc(
                    edge_key=edge_key,
                    line=edge_line,
                    start_type='v',
                    start_id=start_vertex_id,
                    end_type='v',
                    end_id=end_vertex_id
                )
                self.arcs[arc.id] = arc
                self.vertices[start_vertex_id].add_incident_arc(arc.id)
                self.vertices[end_vertex_id].add_incident_arc(arc.id)
            else:
                # Split edge at crossings
                self._split_edge_at_crossings(
                    edge_line, edge_key, start_vertex_id, end_vertex_id, intersections,
                )
    

    def _split_edge_at_crossings(self, edge: LineString, edge_key: str, 
                                 start_vertex_id: int, end_vertex_id: int,
                                 intersections: List[Tuple[float, int]]):
        """Split an edge at crossing points."""
        # Add start and end to create cutting points
        cut_points = [0.0] + [dist for dist, _ in intersections] + [edge.length]
        crossing_ids = [None] + [cid for _, cid in intersections] + [None]
        
        # Create arcs between consecutive cut points
        for i in range(len(cut_points) - 1):
            start_dist = cut_points[i]
            end_dist = cut_points[i + 1]
            
            # Extract the line segment
            if end_dist - start_dist < self.tolerance:
                continue  # Skip tiny segments
            
            arc_line = substring(edge, start_dist, end_dist)
            
            # Determine start and end points
            if i == 0:
                # First arc: starts at vertex u
                start_type, start_id = 'v', start_vertex_id
            else:
                # Starts at a crossing
                start_type, start_id = 'x', crossing_ids[i]
            
            if i == len(cut_points) - 2:
                # Last arc: ends at vertex v
                end_type, end_id = 'v', end_vertex_id
            else:
                # Ends at a crossing
                end_type, end_id = 'x', crossing_ids[i + 1]
            
            # Create arc
            arc = Arc(
                edge_key=edge_key,
                line=arc_line,
                start_type=start_type,
                start_id=start_id,
                end_type=end_type,
                end_id=end_id
            )
            self.arcs[arc.id] = arc
            # Update incidences
            self._update_incidences(arc)
    

    def _update_incidences(self, arc: Arc):
        """Update vertex and crossing incidences for an arc."""
        # Update start point
        if arc.start_type == 'v':
            self.vertices[arc.start_id].add_incident_arc(arc.id)
        else:  # crossing
            # Calculate angle from crossing to arc
            crossing_pt = self.crossings[arc.start_id].point
            next_pt = Point(arc.line.coords[1])
            angle = np.arctan2(
                next_pt.y - crossing_pt.y,
                next_pt.x - crossing_pt.x
            )
            self.crossings[arc.start_id].add_incident_arc(arc.id, angle)
        
        # Update end point
        if arc.end_type == 'v':
            self.vertices[arc.end_id].add_incident_arc(arc.id)
        else:  # crossing
            # Calculate angle from arc to crossing
            crossing_pt = self.crossings[arc.end_id].point
            prev_pt = Point(arc.line.coords[-2])
            angle = np.arctan2(
                prev_pt.y - crossing_pt.y,
                prev_pt.x - crossing_pt.x
            )
            self.crossings[arc.end_id].add_incident_arc(arc.id, angle)
    

    def _determine_crossing_types(self):
        """Determine over/under strands for each crossing using Z-coordinates."""
        
        def _get_z_at_crossing(arc, xid):
            if arc.start_type == 'x' and arc.start_id == xid:
                coords = arc.line.coords[0]
            elif arc.end_type == 'x' and arc.end_id == xid:
                coords = arc.line.coords[-1]
            else:
                raise RuntimeError(f"Arc {arc.id} not properly connected to crossing {xid}.")
            return coords[2]
        
        for xid, x in self.crossings.items():
            raw_arc_ids = x._raw_ccw_ordered_arcs
            if not raw_arc_ids:
                continue # Skip trivial self-crossings
            a, b, _, _ = [self.arcs[i] for i in raw_arc_ids]
            z_a = _get_z_at_crossing(a, xid)
            z_b = _get_z_at_crossing(b, xid)
            # TODO: why is (left-lower, right-upper) undercrossing?
            x._correctly_overstrand = z_a < z_b


    def _generate_pd_code(self) -> str:
        """Generate the PD code string."""
        v_parts = [v.pd_code for v in self.vertices.values() if v.pd_code]
        x_parts = [c.pd_code for c in self.crossings.values() if c.pd_code]
        
        if not v_parts and not x_parts:
            return ""
        
        return ";".join(v_parts + x_parts)
    
    
    def compute_yamada(
            self, 
            variable: sp.Symbol,
            normalize: bool = True,
            n_jobs: int = -1
    ) -> sp.Expr:
        """
        Compute the Yamada polynomial for the knot diagram.
        """
        if not self._cache:
            raise ValueError("PD code must be computed before Yamada polynomial.")
        computer = Yamada(
            vertices=list(self.vertices.values()),
            crossings=list(self.crossings.values()),
            arcs=list(self.arcs.values())
        )
        return computer.compute(variable, normalize=normalize, n_jobs=n_jobs)


explode_to_segments = PDCode._explode_to_segments
find_all_crossings = PDCode._find_all_crossings
project_crossings_on_edge = PDCode._project_crossings_on_edge


def compute_pd_code(
        skeleton_graph: nx.MultiGraph, 
        rotation_angles: Optional[list[float]] = None,
        rotation_order: str = 'ZYX',
    ) -> str:
    """Compute the PD code for a given skeleton graph with optional rotation."""
    generator = PDCode(skeleton_graph)
    return generator.compute(rotation_angles=rotation_angles, 
                                rotation_order=rotation_order)


def compute_yamada_polynomial(
        skeleton_graph: nx.MultiGraph, 
        variable: sp.Symbol,
        rotation_angles: Optional[list[float]] = None,
        rotation_order: str = 'ZYX',
        normalize: bool = True,
        n_jobs: int = -1
    ) -> sp.Expr:
    """Compute the Yamada polynomial for a skeleton graph."""
    generator = PDCode(skeleton_graph)
    generator.compute(rotation_angles=rotation_angles, 
                      rotation_order=rotation_order)
    return generator.compute_yamada(variable, normalize=normalize, n_jobs=n_jobs)