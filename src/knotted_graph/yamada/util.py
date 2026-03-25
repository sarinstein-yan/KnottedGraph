
import numpy as np
import networkx as nx
from shapely.ops import substring
from shapely.geometry import Point, LineString
import re
import itertools
import hashlib
from functools import cache
from collections import defaultdict
from typing import Sequence
from numpy.typing import NDArray


__all__ = [
    "get_rotation_matrix",
    "generate_isotopy_angles",
    "cut_line_string",
    "parse_pd_code_string",
    "build_state_graph",
    "multigraph_key",
]


def get_rotation_matrix(
    angles: Sequence[float],
    order: str = "xyz",
    use_radians: bool = False,
) -> NDArray:
    """
    Construct a 3‑D rotation matrix from an ordered triple of elementary rotations.

    Parameters
    ----------
    angles : (3,) sequence
        The three rotation angles (α, β, γ) applied in the specified order.
    order : str, default "xyz"
        Three‑letter string giving the rotation axes.
        • Lower‑case  → intrinsic (rotations about *body‑fixed* axes)
        • Upper‑case  → extrinsic (rotations about *world* axes)
        Accepted letters: x/X, y/Y, z/Z.
    use_radians : bool, default False
        Supply angles in radians if True, else in degrees.

    Returns
    -------
    R : (3, 3) ndarray
        Combined rotation matrix.
    
    Examples
    --------
    1) Classic aerospace yaw‑pitch‑roll (extrinsic Z‑Y‑X)
    >>> yaw, pitch, roll = 30, 15, 5    # degrees
    >>> R = get_rotation_matrix((yaw, pitch, roll), order="ZYX")

    2) Intrinsic roll‑pitch‑yaw for camera pose (x‑y‑z in body frame)
    >>> R_cam = get_rotation_matrix((5, 15, 30), order="xyz")

    3) Euler ZXZ (commonly used in molecular crystallography)
    >>> phi, theta, psi = 45, 60, 10
    >>> R_euler = get_rotation_matrix((phi, theta, psi), order="zxz")
    """
    if len(order) != 3 or any(ax.lower() not in "xyz" for ax in order):
        raise ValueError("order must be a 3‑letter string with characters "\
                         "x, y, z (case‑sensitive).")

    # Convert to radians if necessary
    a, b, c = angles if use_radians else np.radians(angles)

    def _single_axis(axis: str, theta: float) -> NDArray:
        """Elementary rotation about world axes (lowercase) or body axes (uppercase)."""
        ct, st = np.cos(theta), np.sin(theta)
        if axis.lower() == "x":
            R = np.array([[1, 0, 0],
                          [0, ct, -st],
                          [0, st,  ct]])
        elif axis.lower() == "y":
            R = np.array([[ ct, 0, st],
                          [  0, 1, 0],
                          [-st, 0, ct]])
        else:  # 'z'
            R = np.array([[ct, -st, 0],
                          [st,  ct, 0],
                          [ 0,   0, 1]])
        return R

    # Build the three elementary matrices
    thetas = [a, b, c]
    R_elems = [_single_axis(ax.lower(), th) for ax, th in zip(order, thetas)]

    # Combine them: intrinsic (lower‑case) multiplies right‑to‑left; 
    # extrinsic (upper‑case) left‑to‑right
    R = np.eye(3)
    for ax, Rk in zip(order, R_elems):
        R = R @ Rk if ax.isupper() else Rk @ R

    return R


def generate_isotopy_angles(
    N: int,
    order: str = "ZYX",
    use_radians: bool = False,
) -> NDArray:
    r"""
    Return *N* Euler‑angle triples that are (approximately) uniformly
    distributed over SO(3) / ~, where the equivalence ~ removes
    
        • the sign of the view direction  (v and −v give isotopic diagrams)  
        • rotations about the view axis   (in‑plane diagram spin)

    Parameters
    ----------
    N : int
        Number of representative rotations desired.
    order : str, default "ZYX"
        Three‑letter Euler sequence, upper‑case = extrinsic, lower‑case = intrinsic.
        Must match the `rotation_matrix` helper you already have.
    use_radians : bool, default False
        If False the function returns angles in **degrees** (handy when driving 
        Matplotlib, PyVista, etc.); otherwise in radians.

    Returns
    -------
    angles : (N, 3) ndarray
        Each row is *(α, β, γ)* in the requested `order`.
        γ (roll) is always 0 because in‑plane rotation is isotopic.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))      # ~2.399963..., offsets points nicely
    # --- 1. Fibonacci spiral sampling on the upper hemisphere -----------------
    i = np.arange(N)
    phi = i * GOLDEN_ANGLE                       # azimuth ∈ [0, 2π)
    z   = (i + 0.5) / N                          # uniform height in (0, 1]
    r   = np.sqrt(1.0 - z**2)                    # radius in XY‑plane
    x, y = r * np.cos(phi), r * np.sin(phi)
    dirs = np.stack((x, y, z), axis=1)           # unit view‑direction vectors

    # --- 2. Convert each direction to Euler yaw–pitch (roll = 0) --------------
    # For extrinsic ZYX:
    #   yaw   = atan2(y, x)
    #   pitch = atan2(√(x²+y²), z)
    yaw   = np.arctan2(y, x)
    pitch = np.arctan2(np.sqrt(x**2 + y**2), z)
    roll  = np.zeros_like(yaw)

    euler_ZYX = np.vstack((yaw, pitch, roll)).T  # shape (N, 3)

    # --- 3. Re‑order angles if caller wants a different convention ------------
    idx = {axis.lower(): k for k, axis in enumerate("zyx")}
    perm = [idx[a.lower()] for a in order]       # where to pick yaw/pitch/roll
    angles = euler_ZYX[:, perm]

    if not use_radians:
        angles = np.degrees(angles)

    return angles


def cut_line_string(line, distances, *, tol=1e-12):
    """
    Split a LineString at one or more distances measured from its start.

    Parameters
    ----------
    line : shapely.geometry.LineString
    distances : float | Iterable[float]
        A single distance or an iterable of distances along the line
        (same units as line.length).  Values outside (0, line.length)
        or closer than *tol* to a previous split are ignored.
    tol : float, optional
        Numerical tolerance when comparing distances (default 1e‑12).

    Returns
    -------
    list[LineString]
        Ordered sub‑segments whose concatenation equals *line*.
    """
    # ── 1. normalise the split positions ──────────────────────────────────
    if np.ndim(distances) == 0:
        distances = [float(distances)]
    else:
        distances = [float(d) for d in distances]

    L = line.length
    # keep only unique, in‑range breakpoints and sort them
    cuts = sorted(
        {d for d in distances if tol < d < L - tol},
    )
    if not cuts:
        return [LineString(line)]

    # ── 2. iterate through [0, d1], [d1, d2], …, [dk, L] ───────────────────
    segments = []
    start = 0.0
    for d in cuts + [L]:
        if d - start > tol:           # skip zero‑length chunks
            # Shapely ≥ 2.0: use fast substring
            try:
                seg = substring(line, start, d)
            except ImportError:
                # manual fallback: interpolate the two endpoints and rebuild
                p0 = line.interpolate(start)
                p1 = line.interpolate(d)
                # collect intermediate vertices between the two points
                coords = [p0.coords[0]]
                for x, y, *z in line.coords:
                    pd = line.project(Point(x, y))
                    if start < pd < d:
                        coords.append((x, y, *z))
                coords.append(p1.coords[0])
                seg = LineString(coords)
            segments.append(seg)
        start = d
    return segments


def parse_pd_code_string(pd_str):
    vertices, crossings = [], []
    # allow zero or more digits/commas inside the brackets
    tokpat = re.compile(r'^(V|X)\[\s*([\d,]*)\s*\]$')
    for raw in pd_str.strip().split(';'):
        token = raw.strip()
        if not token:
            continue

        m = tokpat.match(token)
        if not m:
            raise ValueError(f"Bad token: {token!r}")

        kind, nums = m.groups()
        # if nums is empty, we want an empty list
        labels = [int(n) for n in nums.split(',')] if nums else []

        if kind == 'V':
            vertices.append(labels)
        else:
            crossings.append(labels)

    return vertices, crossings


def build_state_graph(vertices, crossings, state):
    G = nx.MultiGraph()
    nV, nX = len(vertices), len(crossings)
    G.add_nodes_from(range(nV),   kind='V')
    G.add_nodes_from(range(nV, nV+nX), kind='X')
    label_ends = defaultdict(list)
    for vidx, v_lbls in enumerate(vertices):
        for lbl in v_lbls:
            label_ends[lbl].append(vidx)
    for xidx, (i0,i1,i2,i3) in enumerate(crossings):
        Xnode = nV + xidx
        for lbl in (i0,i1,i2,i3):
            label_ends[lbl].append(Xnode)
    for lbl, ends in label_ends.items():
        if len(ends) != 2:
            raise ValueError(f"Label {lbl} appears {len(ends)} times; expected 2")
        G.add_edge(*ends, label=lbl)
    for xidx, res in enumerate(state):
        Xnode = nV + xidx
        i0,i1,i2,i3 = crossings[xidx]
        nbr = {}
        for lbl in (i0,i1,i2,i3):
            a,b = label_ends[lbl]
            nbr[lbl] = b if a==Xnode else a
        if res == 0:
            G.add_edge(nbr[i0], nbr[i3])
            G.add_edge(nbr[i1], nbr[i2])
            G.remove_node(Xnode)
        elif res == 1:
            G.add_edge(nbr[i0], nbr[i2])
            G.add_edge(nbr[i1], nbr[i3])
            G.remove_node(Xnode)
        elif res == 2:
            G.nodes[Xnode]['kind'] = 'V'
        else:
            raise ValueError(f"Invalid state {res}")
    return G


def multigraph_key(G):
    """Canonical unlabeled multigraph key."""
    try:
        import igraph as ig
        def _key(G):
            # Map NetworkX node labels to 0..n-1
            idx = {n: i for i, n in enumerate(G.nodes())}
            # Build edge list (parallel edges appear multiple times)
            edges = [(idx[u], idx[v]) for u, v, _ in G.edges(keys=True)]
            # Build igraph (undirected)
            igG = ig.Graph(len(idx), edges=edges, directed=False)
            # Uniform color partition
            igG.vs['color'] = 0
            # Canonical permutation (Bliss)
            perm = igG.canonical_permutation(color=igG.vs['color'])
            canon = igG.permute_vertices(perm)
            return tuple(sorted(canon.get_edgelist()))
    
    except ImportError:
        # Fallback to NetworkX for environments without igraph
        def _key(G):
            sigs = {}
            deg = dict(G.degree())
            loop_mult = {n: 0 for n in G.nodes()}
            for u, v in G.edges():
                if u == v:
                    loop_mult[u] += 1
            for n in G.nodes():
                # multiset of neighbor degrees, counting multiplicities of edges
                neigh_multideg = []
                for nbr in G.neighbors(n):
                    mult = sum(1 for _ in G.get_edge_data(n, nbr).values())
                    neigh_multideg.extend([deg[nbr]] * mult)
                sigs[n] = (
                    deg[n],
                    loop_mult[n],
                    tuple(sorted(neigh_multideg))
                )
            nodes = sorted(G.nodes(), key=lambda n: (sigs[n], n))
            idx = {n:i for i,n in enumerate(nodes)}
            edges = []
            for u,v in G.edges():
                i,j = idx[u], idx[v]
                if i>j: i,j = j,i
                edges.append((i,j))
            return tuple(sorted(edges))
    
    key = _key(G)
    return hashlib.sha256(repr(key).encode()).hexdigest()


