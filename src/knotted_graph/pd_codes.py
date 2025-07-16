'''
This document includes my attempts to automate the construction of PD codes.
Currently, they are not very effective, and manually entering the angles to 
obtain PD codes seems to work better.
However, I am including them here to give you an idea of my previous efforts. 
If we are unable to make progress on this in later stages, we can simply 
remove the code.
'''

import numpy as np               # Numerical operations
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import plotly.graph_objects as go  # Interactive plotting library
from shapely.ops import split                   # Geometry splitting utility
from shapely.geometry import Point, LineString, GeometryCollection, MultiPoint
from shapely.strtree import STRtree


__all__ = [
    "compute_rotation_matrix",
    "split_line_at_crossings",
    "reconstruct_3d_coords",
    "planar_diagram_code",
    "find_best_view",
]


def compute_rotation_matrix(
        azimuth_deg: float, 
        elevation_deg: float
    ) -> np.ndarray:
    """
    Compute a combined rotation matrix for a given azimuth (yaw) and elevation (pitch).

    Parameters:
    - azimuth_deg: Rotation angle around Z-axis in degrees (yaw).
    - elevation_deg: Rotation angle around X-axis in degrees (pitch).

    Returns:
    - 3×3 numpy array representing the combined rotation.
    """
    # Convert degrees to radians
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    # Rotation around Z-axis (yaw)
    Rz = np.array([
        [ np.cos(-az), -np.sin(-az), 0],
        [ np.sin(-az),  np.cos(-az), 0],
        [           0,            0, 1]
    ])

    # Rotation around X-axis (pitch)
    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(el), -np.sin(el)],
        [0,  np.sin(el),  np.cos(el)]
    ])

    # First apply yaw, then pitch: overall rotation is Rx @ Rz
    return Rx @ Rz


def split_line_at_crossings(
        line: LineString,
        crossing_points: list[Point],
        tol: float = 1e-6
    ) -> list[LineString]:
    """
    Split a LineString at specified interior crossing points.

    Parameters:
    - line: The original LineString to split.
    - crossing_points: List of Shapely Point objects lying on the line.
    - tol: Distance tolerance for determining interior points.

    Returns:
    - List of LineString segments between successive split points.
    """
    # Project each crossing onto [0,1] normalized length
    t_vals = sorted({round(line.project(pt, normalized=True), 12)
                     for pt in crossing_points})
    # Include endpoints 0 and 1
    t_positions = [0.0] + t_vals + [1.0]

    segments = []
    # Create segment for each interval [t0, t1]
    for t0, t1 in zip(t_positions, t_positions[1:]):
        p_start = line.interpolate(t0, normalized=True).coords[0]
        p_end   = line.interpolate(t1, normalized=True).coords[0]
        # Retain any original vertices strictly between t0 and t1
        mids = [coord for coord in line.coords
                if t0 + tol < line.project(Point(coord), normalized=True) < t1 - tol]
        segments.append(LineString([p_start] + mids + [p_end]))

    return segments


def reconstruct_3d_coords(
        coords2d: list[tuple[float,float]],
        base2d: list[tuple[float,float]],
        base3d: list[tuple[float,float,float]],
        tol: float = 1e-6
    ) -> list[tuple[float,float,float]]:
    """
    Given a 2D split polyline and its original 2D↔3D correspondences,
    linearly interpolate to recover the 3D coordinates.

    Parameters:
    - coords2d: Coordinates of points along split segment in 2D.
    - base2d: Original 2D coordinates before splitting.
    - base3d: Original 3D coordinates corresponding to base2d.
    - tol: Tolerance for point-on-segment checks.

    Returns:
    - List of 3D coordinates for each 2D point in coords2d.
    """
    coords3d = []
    for x, y in coords2d:
        P = Point(x, y)
        # Find underlying base segment
        for i in range(len(base2d) - 1):
            seg = LineString([base2d[i], base2d[i+1]])
            if seg.distance(P) < tol:
                t = seg.project(P, normalized=True)
                x0, y0, z0 = base3d[i]
                x1, y1, z1 = base3d[i+1]
                coords3d.append((
                    x0 + t*(x1 - x0),
                    y0 + t*(y1 - y0),
                    z0 + t*(z1 - z0)
                ))
                break
        else:
            # Fallback: nearest original vertex
            _, idx = min(
                ((x - vx)**2 + (y - vy)**2, j)
                for j, (vx, vy) in enumerate(base2d)
            )
            coords3d.append(base3d[idx])
    return coords3d


def planar_diagram_code(
        graph,
        view: tuple[float,float] = (45, 30),
        crossing_tol: float = 5
    ) -> tuple[list[str], list[str]]:
    """
    Main function to annotate a graph for planar diagram codes:

    Steps:
      1. Rotate all nodes & edge points into camera frame and project to 2D.
      2. Cache per-edge point lists and segmentize for intersection tests.
      3. In parallel, detect true interior crossings (excluding nodes).
      4. Split edges at crossings, reconstruct 3D coords, build Plotly traces.
      5. Render 2D plot with nodes, arc lines, labels, and crossing markers.
      6. Assemble code parts V_parts (vertex incidences) and X_parts (crossings order).

    Returns:
      - V_parts: List of strings "V[...]" for each node.
      - X_parts: List of strings "X[...]" for each crossing.
    """
    tol=1e-6
    # --- Step 1: Rotate & project nodes ---
    R = compute_rotation_matrix(*view)
    node_positions_2d = {
        n: tuple((R @ np.array(data['pos']))[:2])
        for n, data in graph.nodes(data=True)
    }

    # --- Step 2: Cache per-edge 2D & 3D coords and small segments ---
    segs2d, segs3d, edge_segments = {}, {}, {}
    for eid, (u, v, k) in enumerate(graph.edges(keys=True)):
        raw_pts = [p for p in graph.edges[u, v, k]['pts'] if p is not None]
        rotated = [R @ np.array(p) for p in raw_pts]
        pts2d = [(float(x), float(y)) for x, y, _ in rotated]
        segs2d[eid] = pts2d
        segs3d[eid] = [tuple(pt) for pt in rotated]
        edge_segments[eid] = [LineString([pts2d[i], pts2d[i+1]])
                              for i in range(len(pts2d)-1)]

    # --- Step 3: Parallel interior crossing detection ---
    crossings = {eid: [] for eid in segs2d}
    def detect_crossings_for_edge(eid1: int) -> list[tuple[int, Point]]:
        hits = []
        for eid2, segs in edge_segments.items():
            if eid2 <= eid1:
                continue
            for seg1 in edge_segments[eid1]:
                for seg2 in segs:
                    inter = seg1.intersection(seg2)
                    if inter.is_empty:
                        continue
                    if isinstance(inter, Point):
                        pts = [inter]
                    elif isinstance(inter, (MultiPoint, GeometryCollection)):
                        # extract any Point children
                        pts = [g for g in inter.geoms if isinstance(g, Point)]
                    elif isinstance(inter, LineString):
                        # overlap: you might choose to sample endpoints, midpoints, or skip
                        # here we’ll take the segment’s end‑points as “crossings”
                        pts = [Point(c) for c in inter.coords]
                    else:
                        # e.g. MultiLineString or unexpected types: skip
                        pts = []
                    for pt in pts:
                        if seg1.distance(pt) > crossing_tol or seg2.distance(pt) > crossing_tol:
                            continue
                        # Exclude node positions
                        if any(abs(pt.x - x) < crossing_tol and abs(pt.y - y) < crossing_tol
                               for x, y in node_positions_2d.values()):
                            continue
                        hits.append((eid1, pt))
                        hits.append((eid2, pt))
        return hits

    with ThreadPoolExecutor() as executor:
        for hits in executor.map(detect_crossings_for_edge, segs2d.keys()):
            for eid, pt in hits:
                if not any(pt.distance(old) < tol for old in crossings[eid]):
                    crossings[eid].append(pt)

    # --- Step 4: Split edges & build arc traces ---
    edge_traces, arcs = [], []
    arc_x, arc_y, arc_text = [], [], []
    arc_id = 1
    for eid, pts2d in segs2d.items():
        # Deduplicate crossing points
        uniq_pts = []
        for pt in crossings[eid]:
            if any(pt.distance(o) < crossing_tol for o in uniq_pts):
                continue
            uniq_pts.append(pt)

        pieces = split_line_at_crossings(LineString(pts2d), uniq_pts, tol)
        for segment in pieces:
            c2d = list(segment.coords)
            c3d = reconstruct_3d_coords(c2d, pts2d, segs3d[eid], tol)
            z_mid = c3d[len(c3d)//2][2]

            xs, ys = zip(*c2d)
            hover_labels = [f"arc {arc_id} | z_mid = {z_mid:.4f}"] * len(xs)
            edge_traces.append(
                go.Scatter(x=xs, y=ys, mode='lines',
                           line=dict(color='blue', width=3),
                           hoverinfo='text', hovertext=hover_labels,
                           showlegend=False)
            )

            arcs.append({'id': arc_id, 'coords2d': c2d, 'coords3d': c3d})
            mid_idx = len(xs)//2
            arc_x.append(xs[mid_idx])
            arc_y.append(ys[mid_idx])
            arc_text.append(str(arc_id))
            arc_id += 1

    # --- Step 5: Render plot with nodes, labels, crossings ---
    node_x, node_y = zip(*node_positions_2d.values())
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                            marker=dict(color='red', size=12),
                            showlegend=False)
    label_trace = go.Scatter(x=arc_x, y=arc_y, mode='text',
                             text=arc_text, textfont=dict(size=16),
                             showlegend=False)

    uniq_cross_points = []
    for pts in crossings.values():
        for pt in pts:
            if not any(pt.equals(existing) for existing in uniq_cross_points):
                uniq_cross_points.append(pt)
    cross_x = [pt.x for pt in uniq_cross_points]
    cross_y = [pt.y for pt in uniq_cross_points]
    cross_trace = go.Scatter(x=cross_x, y=cross_y, mode='markers',
                             marker=dict(symbol='x', size=12, color='black'),
                             showlegend=False)

    fig = go.Figure(data=edge_traces + [node_trace, label_trace, cross_trace])
    fig.update_layout(title=f"View (az,el)={view}",
                      xaxis=dict(scaleanchor='y', scaleratio=1),
                      width=700, height=700,
                      showlegend=False)
    fig.show()

    # --- Step 6: Assemble planar diagram code parts ---
    V_parts, X_parts = [], []
    for node in sorted(graph.nodes()):
        incident_arcs = [arc['id'] for arc in arcs
                         if np.allclose(arc['coords2d'][0], node_positions_2d[node], tol)
                         or np.allclose(arc['coords2d'][-1], node_positions_2d[node], tol)]
        V_parts.append(f"V[{','.join(map(str, sorted(incident_arcs)))}]")
    all_meet=[]
    for pt in uniq_cross_points:
        meet = []
        for arc in arcs:
            line2d = LineString(arc['coords2d'])
            if line2d.distance(pt) < tol:
                # find which 2D segment is hit
                z_before = None
                for i, (p0, p1) in enumerate(zip(arc['coords2d'], arc['coords2d'][1:])):
                    seg = LineString([p0, p1])
                    if seg.distance(pt) < tol:
                        # take the Z of the vertex just before the crossing
                        z_before = arc['coords3d'][i][2]
                        break

                # fallback if nothing matched
                if z_before is None:
                    # e.g. crossing at final vertex, just grab last
                    z_before = arc['coords3d'][-1][2]

                start, end = arc['coords2d'][0], arc['coords2d'][-1]
                fx, fy = end if np.allclose(start, (pt.x, pt.y), tol) else start
                angle = np.arctan2(pt.y - fy, pt.x - fx) % (2*np.pi)

                meet.append((arc['id'], z_before, angle))
        meet.sort(key=lambda x: x[1])
        sorted_ids = sorted(item[0] for item in meet)
        all_meet.append(meet)
        X_parts.append(f"X[{','.join(map(str, sorted_ids))}]")
        
    return V_parts, X_parts,all_meet

 


def find_best_view(
    simplified_graph,
    max_pts=20,
    init_view=(50, 90),
    T0=200,
    Tmin=0.5,
    alpha=0.8,
    steps=40,
    tol=3.0,
    cross_penal_factor=20,
    cross_penal_dist=10.0,
    cross_dist_penal_factor=15,
    node_penal_dist=25.0,
    node_penal_factor=40,
):
    """
    Finds the best 2D viewing angles (azimuth, elevation) for projecting
    a 3D graph by minimizing a penalized count of edge crossings and
    node‑proximity violations.

    Args:
        simplified_graph: NetworkX graph whose edges carry 'pts' = list of 3D coords.
        max_pts: Maximum number of points per edge to keep when down‑sampling.
        init_view: Starting (azimuth, elevation) in degrees for the optimizer.
        T0: Initial "temperature" for simulated annealing.
        Tmin: Minimum temperature at which to stop annealing.
        alpha: Cooling rate (temperature multiplier per cycle).
        steps: Number of random proposals per temperature.
        tol: Distance tolerance (in 2D) to register a true crossing.
        cross_penal_factor=Penalty for number of crossings
        cross_penal_dist: Radius around each node within which crossings incur extra penalty.
        cross_dist_penal_factor: Penalty for near‑node crossings.
        node_penal_dist: Minimum allowed distance between any two node projections.
        node_penal_factor: Penalty per node‑node proximity violation.
        
    Returns:
        Tuple (best_azimuth, best_elevation) in degrees.
    """

    # ──────────────────────────────────────────────────────────────
    # 1) Down‑sample each edge’s 3D polyline to at most max_pts points
    # ──────────────────────────────────────────────────────────────
    for u, v, key, data in simplified_graph.edges(keys=True, data=True):
        # Convert raw pts to NumPy arrays
        raw3d = [np.array(p, dtype=float) for p in data['pts'] if p is not None]
        N = len(raw3d)

        if N > max_pts:
            # Uniformly pick ~max_pts along the chain
            step = max(1, N // max_pts)
            ds = raw3d[::step]
            # Ensure last point is included
            if not np.array_equal(ds[-1], raw3d[-1]):
                ds.append(raw3d[-1])
        else:
            # Keep all points if already short
            ds = raw3d

        # Store down‑sampled coords for later projection
        data['_ds3d'] = ds

    # ──────────────────────────────────────────────────────────────
    # 2) Define the scoring function for a given view
    # ──────────────────────────────────────────────────────────────
    def count_with_penalty_2d(H, view):
        """
        Projects nodes & edges to 2D using `view` (az,el), then counts:
          - true crossings (distance < tol to both segments)
          - crossings near any node (extra penalty)
          - any two nodes projected too close (node‑node violation)

        Returns a weighted sum (lower is better).
        """
        # Rotation matrix for given azimuth & elevation
        R = compute_rotation_matrix(*view)

        # 2a) Project nodes into 2D
        node_pts = []
        for n, d in H.nodes(data=True):
            x, y = (R @ np.array(d['pos']))[:2]
            node_pts.append((float(x), float(y)))

        # 2b) Build all 2D line segments from down‑sampled edge pts
        segs = []
        for u, v, key in H.edges(keys=True):
            pts3d = H.edges[u, v, key]['_ds3d']
            # apply same rotation & drop Z
            rot2d = [tuple((R @ p)[:2]) for p in pts3d]
            # break into segment‑by‑segment LineStrings
            for i in range(len(rot2d) - 1):
                segs.append(LineString([rot2d[i], rot2d[i+1]]))
        # 2c) Build spatial index for fast crossing queries
        tree = STRtree(segs)
        crossings = set()
        close_cross_dists = []

        # 2d) Identify all true crossings (excluding node intersections)
        for i, seg in enumerate(segs):
            for j in tree.query(seg):
                if j <= i:
                    continue
                other = segs[j]
                inter = seg.intersection(other)
                # Must be a point and within tol on both segments
                if not isinstance(inter, Point):
                    continue
                if seg.distance(inter) >= tol or other.distance(inter) >= tol:
                    continue
                x, y = inter.x, inter.y
                # Skip if this crossing is actually a node location
                if any(np.hypot(x - nx, y - ny) < tol for nx, ny in node_pts):
                    continue
                coord = (round(x, 6), round(y, 6))
                crossings.add(coord)
                # Record exact distance to each node for penalty
                for nx, ny in node_pts:
                    d = np.hypot(x - nx, y - ny)
                    if d < cross_penal_dist:
                        close_cross_dists.append(d)

        # 2e) Collect node‑node proximity distances
        node_distances = []
        L = len(node_pts)
        for i in range(L):
            for j in range(i + 1, L):
                d = np.hypot(*(np.subtract(node_pts[i], node_pts[j])))
                if d < node_penal_dist:
                    node_distances.append(d)

        # 2f) Compute weighted score with exponential penalties
        cross_penalty = cross_penal_factor * np.exp(len(crossings))
        cross_dist_penalty = cross_dist_penal_factor * sum(np.exp(-d / cross_penal_dist)
                                                            for d in close_cross_dists)
        node_penalty = node_penal_factor * sum(np.exp(-d / node_penal_dist)
                                                for d in node_distances)
        score = cross_penalty + cross_dist_penalty + node_penalty
        return score


    # ──────────────────────────────────────────────────────────────
    # 3) Simulated annealing: explore different (az,el) to minimize score
    # ──────────────────────────────────────────────────────────────
    def optimize_view_sa(H, count_fn):
        current = np.array(init_view, float)
        best = current.copy()
        best_score = count_fn(H, tuple(current))
        T = T0

        while T > Tmin:
            for _ in range(steps):
                az, el = current
                # Propose small random change in view
                cand = (
                    (az + np.random.uniform(-10, 10)) % 360,
                    np.clip(el + np.random.uniform(-10, 10), 0, 360),
                )
                s = count_fn(H, cand)
                dE = s - best_score
                # Accept if better, or with Boltzmann probability
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    current = np.array(cand)
                    if s < best_score:
                        best_score = s
                        best = current.copy()
            T *= alpha

        return (best[0], best[1]), best_score

    # ──────────────────────────────────────────────────────────────
    # 4) Run optimizer and return the winning view
    # ──────────────────────────────────────────────────────────────
    best_view, _ = optimize_view_sa(simplified_graph, count_with_penalty_2d)
    return best_view