from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Callable

import networkx as nx

__all__ = [
    "GraphFamilySpec",
    "canonical_pair",
    "ensure_multigraph",
    "graph_summary",
    "numeric_labels",
    "build_simple_multigraph",
    "regular_ngon_positions",
    "column_positions",
    "normalize_positions",
    "build_bouquet_graph",
    "bouquet_layout",
    "build_theta_graph",
    "theta_layout",
    "build_periodic_theta_graph",
    "periodic_theta_layout",
    "build_sierpinski_graph",
    "sierpinski_layout",
    "build_complete_graph",
    "complete_graph_layout",
    "build_complete_bipartite_graph",
    "complete_bipartite_layout",
    "build_complete_multipartite_graph",
    "complete_multipartite_layout",
    "build_fan_graph",
    "fan_layout",
    "build_wheel_graph",
    "wheel_layout",
    "build_ladder_graph",
    "ladder_layout",
    "build_circular_ladder_graph",
    "circular_ladder_layout",
    "build_grid_graph",
    "grid_layout",
    "build_cylinder_graph",
    "cylinder_layout",
    "build_hypercube_graph",
    "hypercube_layout",
    "build_friendship_graph",
    "friendship_layout",
    "build_windmill_graph",
    "windmill_layout",
    "build_circulant_graph",
    "circulant_layout",
    "build_generalized_petersen_graph",
    "generalized_petersen_layout",
    "build_mobius_ladder_graph",
    "mobius_ladder_layout",
    "edge_curvatures",
    "edge_radii",
    "loop_geometries",
    "plot_structured_multigraph",
    "GRAPH_FAMILY_CATALOG",
    "build_graph_case",
    "graph_family_names",
    "FEATURED_COMPUTE_EXAMPLES",
    "PRACTICAL_COMPUTE_EXAMPLES",
    "PLOT_GALLERY_EXAMPLES",
    "NOTEBOOK_YAMADA_EXAMPLES",
    "DEFAULT_CUSTOM_EXAMPLE",
]


@dataclass(frozen=True)
class GraphFamilySpec:
    builder: Callable[..., nx.MultiGraph]
    layout: Callable[..., dict]
    sample_args: tuple
    note: str


def canonical_pair(u, v):
    return (u, v) if repr(u) <= repr(v) else (v, u)


def ensure_multigraph(G: nx.Graph | nx.MultiGraph) -> nx.MultiGraph:
    return G.copy() if isinstance(G, nx.MultiGraph) else nx.MultiGraph(G)


def graph_summary(G: nx.Graph | nx.MultiGraph) -> dict[str, int]:
    G = ensure_multigraph(G)
    edge_groups: dict[tuple, int] = defaultdict(int)
    loops = 0
    for u, v, _key in G.edges(keys=True):
        if u == v:
            loops += 1
        edge_groups[canonical_pair(u, v)] += 1

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "loops": loops,
        "max_multiplicity": max(edge_groups.values(), default=0),
    }


def numeric_labels(G: nx.Graph | nx.MultiGraph) -> dict:
    return {node: i for i, node in enumerate(sorted(G.nodes(), key=repr))}


def build_simple_multigraph(nodes, pairs) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_nodes_from(nodes)
    seen = set()
    for u, v in pairs:
        pair = canonical_pair(u, v)
        if pair not in seen:
            seen.add(pair)
            G.add_edge(*pair)
    return G


def regular_ngon_positions(labels, radius: float = 2.0, phase: float = math.pi / 2):
    labels = list(labels)
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: (0.0, 0.0)}

    pos = {}
    for i, label in enumerate(labels):
        angle = phase + 2 * math.pi * i / len(labels)
        pos[label] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos


def column_positions(labels, x: float, y_gap: float = 1.2):
    labels = list(labels)
    if not labels:
        return {}
    height = y_gap * (len(labels) - 1)
    return {label: (x, height / 2 - i * y_gap) for i, label in enumerate(labels)}


def normalize_positions(pos, target_radius: float = 2.8):
    if not pos:
        return {}

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    extent = max(x_max - x_min, y_max - y_min, 1.0)
    scale = (2 * target_radius) / extent

    return {
        node: ((x - x_mid) * scale, (y - y_mid) * scale)
        for node, (x, y) in pos.items()
    }


def build_bouquet_graph(n_loops: int) -> nx.MultiGraph:
    if n_loops < 0:
        raise ValueError("n_loops must be >= 0")
    G = nx.MultiGraph()
    G.add_node(0)
    for _ in range(n_loops):
        G.add_edge(0, 0)
    return G


def bouquet_layout(n_loops: int):
    return {0: (0.0, 0.0)}


def build_theta_graph(s_parallel_edges: int) -> nx.MultiGraph:
    if s_parallel_edges < 1:
        raise ValueError("s_parallel_edges must be >= 1")
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1])
    for _ in range(s_parallel_edges):
        G.add_edge(0, 1)
    return G


def theta_layout(s_parallel_edges: int):
    return {0: (-1.9, 0.0), 1: (1.9, 0.0)}


def build_periodic_theta_graph(s: int) -> nx.MultiGraph:
    if s < 1:
        raise ValueError("s must be >= 1")

    G = nx.MultiGraph()
    G.add_nodes_from(["top", "bottom"])

    for i in range(s):
        midpoint = ("m", i)
        G.add_edge("top", midpoint)
        G.add_edge(midpoint, "bottom")

    for i in range(s):
        G.add_edge(("m", i), ("m", (i + 1) % s))

    return G


def periodic_theta_layout(s: int, midpoint_radius: float = 1.65):
    pos = {"top": (0.0, 2.3), "bottom": (0.0, -2.3)}
    if s == 1:
        pos[("m", 0)] = (0.0, 0.0)
        return pos

    pos.update(regular_ngon_positions([("m", i) for i in range(s)], radius=midpoint_radius))
    return pos


def build_sierpinski_graph(n: int, t: int) -> nx.MultiGraph:
    if n < 2:
        raise ValueError("n must be >= 2")
    if t < 1:
        raise ValueError("t must be >= 1")

    if t == 1:
        nodes = [(i,) for i in range(n)]
        return build_simple_multigraph(nodes, combinations(nodes, 2))

    H = build_sierpinski_graph(n, t - 1)
    G = nx.MultiGraph()

    for i in range(n):
        for node in H.nodes():
            G.add_node((i,) + node)
        for u, v, _key in H.edges(keys=True):
            G.add_edge((i,) + u, (i,) + v)

    for i in range(n):
        for j in range(i + 1, n):
            u = (i,) + (j,) * (t - 1)
            v = (j,) + (i,) * (t - 1)
            G.add_edge(u, v)

    return G


def sierpinski_layout(n: int, t: int, radius: float = 3.0, scale: float | None = None):
    if scale is None:
        scale = 0.5 if n == 3 else 0.34

    centers = regular_ngon_positions(range(n), radius=radius)
    if t == 1:
        return {(i,): centers[i] for i in range(n)}

    sub_pos = sierpinski_layout(n, t - 1, radius=radius * scale, scale=scale)
    pos = {}
    for i in range(n):
        cx, cy = centers[i]
        for node, (x, y) in sub_pos.items():
            pos[(i,) + node] = (cx + x, cy + y)
    return pos


def build_complete_graph(n_vertices: int) -> nx.MultiGraph:
    if n_vertices < 1:
        raise ValueError("n_vertices must be >= 1")
    return build_simple_multigraph(range(n_vertices), combinations(range(n_vertices), 2))


def complete_graph_layout(n_vertices: int, radius: float = 2.35):
    return regular_ngon_positions(range(n_vertices), radius=radius)


def build_complete_bipartite_graph(left_size: int, right_size: int) -> nx.MultiGraph:
    if left_size < 1 or right_size < 1:
        raise ValueError("left_size and right_size must be >= 1")

    left_nodes = [("L", i) for i in range(left_size)]
    right_nodes = [("R", j) for j in range(right_size)]
    pairs = [(u, v) for u in left_nodes for v in right_nodes]
    return build_simple_multigraph(left_nodes + right_nodes, pairs)


def complete_bipartite_layout(left_size: int, right_size: int, x_gap: float = 3.2, y_gap: float = 1.2):
    pos = {}
    pos.update(column_positions([("L", i) for i in range(left_size)], -x_gap / 2, y_gap=y_gap))
    pos.update(column_positions([("R", j) for j in range(right_size)], x_gap / 2, y_gap=y_gap))
    return pos


def build_complete_multipartite_graph(*parts: int) -> nx.MultiGraph:
    if len(parts) < 2:
        raise ValueError("at least two parts are required")
    if any(part < 1 for part in parts):
        raise ValueError("all part sizes must be >= 1")

    part_nodes = [[(i, j) for j in range(size)] for i, size in enumerate(parts)]
    all_nodes = [node for nodes in part_nodes for node in nodes]
    pairs = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pairs.extend((u, v) for u in part_nodes[i] for v in part_nodes[j])
    return build_simple_multigraph(all_nodes, pairs)


def complete_multipartite_layout(*parts: int, x_gap: float = 2.7, y_gap: float = 1.2):
    pos = {}
    width = x_gap * (len(parts) - 1)
    for i, size in enumerate(parts):
        x = i * x_gap - width / 2
        pos.update(column_positions([(i, j) for j in range(size)], x, y_gap=y_gap))
    return pos


def build_fan_graph(n_path_vertices: int) -> nx.MultiGraph:
    if n_path_vertices < 2:
        raise ValueError("n_path_vertices must be >= 2")
    hub = "h"
    path_nodes = list(range(n_path_vertices))
    pairs = [(i, i + 1) for i in range(n_path_vertices - 1)]
    pairs.extend((hub, node) for node in path_nodes)
    return build_simple_multigraph([hub] + path_nodes, pairs)


def fan_layout(n_path_vertices: int, spacing: float = 1.3):
    pos = {i: (spacing * i, 0.0) for i in range(n_path_vertices)}
    pos["h"] = (spacing * (n_path_vertices - 1) / 2, 1.95)
    return pos


def build_wheel_graph(n_rim_vertices: int) -> nx.MultiGraph:
    if n_rim_vertices < 3:
        raise ValueError("n_rim_vertices must be >= 3")
    center = "c"
    rim = list(range(n_rim_vertices))
    pairs = [(i, (i + 1) % n_rim_vertices) for i in rim]
    pairs.extend((center, i) for i in rim)
    return build_simple_multigraph([center] + rim, pairs)


def wheel_layout(n_rim_vertices: int, radius: float = 2.25):
    pos = {"c": (0.0, 0.0)}
    pos.update(regular_ngon_positions(range(n_rim_vertices), radius=radius))
    return pos


def build_ladder_graph(n_rungs: int) -> nx.MultiGraph:
    if n_rungs < 2:
        raise ValueError("n_rungs must be >= 2")
    nodes = [("top", i) for i in range(n_rungs)] + [("bottom", i) for i in range(n_rungs)]
    pairs = [(("top", i), ("top", i + 1)) for i in range(n_rungs - 1)]
    pairs += [(("bottom", i), ("bottom", i + 1)) for i in range(n_rungs - 1)]
    pairs += [(("top", i), ("bottom", i)) for i in range(n_rungs)]
    return build_simple_multigraph(nodes, pairs)


def ladder_layout(n_rungs: int, x_spacing: float = 1.55, y_gap: float = 1.65):
    pos = {}
    for i in range(n_rungs):
        pos[("top", i)] = (x_spacing * i, y_gap / 2)
        pos[("bottom", i)] = (x_spacing * i, -y_gap / 2)
    return pos


def build_circular_ladder_graph(n_rungs: int) -> nx.MultiGraph:
    if n_rungs < 3:
        raise ValueError("n_rungs must be >= 3")
    nodes = [("top", i) for i in range(n_rungs)] + [("bottom", i) for i in range(n_rungs)]
    pairs = [(("top", i), ("top", (i + 1) % n_rungs)) for i in range(n_rungs)]
    pairs += [(("bottom", i), ("bottom", (i + 1) % n_rungs)) for i in range(n_rungs)]
    pairs += [(("top", i), ("bottom", i)) for i in range(n_rungs)]
    return build_simple_multigraph(nodes, pairs)


def circular_ladder_layout(n_rungs: int, outer_radius: float = 2.65, inner_radius: float = 1.55):
    pos = regular_ngon_positions([("top", i) for i in range(n_rungs)], radius=outer_radius)
    pos.update(regular_ngon_positions([("bottom", i) for i in range(n_rungs)], radius=inner_radius))
    return pos


def build_grid_graph(rows: int, cols: int) -> nx.MultiGraph:
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    pairs = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                pairs.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                pairs.append(((r, c), (r + 1, c)))
    return build_simple_multigraph(nodes, pairs)


def grid_layout(rows: int, cols: int, spacing: float = 1.3):
    return {(r, c): (spacing * c, -spacing * r) for r in range(rows) for c in range(cols)}


def build_cylinder_graph(rows: int, cols: int) -> nx.MultiGraph:
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must be >= 2")
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    pairs = []
    for r in range(rows):
        for c in range(cols):
            pairs.append(((r, c), (r, (c + 1) % cols)))
            if r + 1 < rows:
                pairs.append(((r, c), (r + 1, c)))
    return build_simple_multigraph(nodes, pairs)


def cylinder_layout(rows: int, cols: int, spacing: float = 1.3):
    return grid_layout(rows, cols, spacing=spacing)


def build_hypercube_graph(dimension: int) -> nx.MultiGraph:
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    nodes = list(product([0, 1], repeat=dimension))
    pairs = []
    for node in nodes:
        for i in range(dimension):
            neighbor = node[:i] + (1 - node[i],) + node[i + 1 :]
            if node < neighbor:
                pairs.append((node, neighbor))
    return build_simple_multigraph(nodes, pairs)


def hypercube_layout(dimension: int, x_gap: float = 1.15, y_gap: float = 1.45):
    layers = {}
    for node in sorted(product([0, 1], repeat=dimension)):
        layers.setdefault(sum(node), []).append(node)

    pos = {}
    for weight in range(dimension + 1):
        layer_nodes = layers.get(weight, [])
        width = x_gap * (len(layer_nodes) - 1)
        for i, node in enumerate(layer_nodes):
            pos[node] = (i * x_gap - width / 2, -weight * y_gap)
    return pos


def build_friendship_graph(n_triangles: int) -> nx.MultiGraph:
    if n_triangles < 1:
        raise ValueError("n_triangles must be >= 1")
    center = "c"
    G = nx.MultiGraph()
    G.add_node(center)
    for i in range(n_triangles):
        a = ("a", i)
        b = ("b", i)
        G.add_edge(center, a)
        G.add_edge(center, b)
        G.add_edge(a, b)
    return G


def friendship_layout(n_triangles: int, radius: float = 2.15):
    pos = {"c": (0.0, 0.0)}
    for i in range(n_triangles):
        angle = math.pi / 2 + 2 * math.pi * i / n_triangles
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        dx = 0.58 * math.cos(angle + math.pi / 2)
        dy = 0.58 * math.sin(angle + math.pi / 2)
        pos[("a", i)] = (cx + dx, cy + dy)
        pos[("b", i)] = (cx - dx, cy - dy)
    return pos


def build_windmill_graph(clique_size: int, copies: int) -> nx.MultiGraph:
    if clique_size < 2:
        raise ValueError("clique_size must be >= 2")
    if copies < 1:
        raise ValueError("copies must be >= 1")

    center = "c"
    G = nx.MultiGraph()
    G.add_node(center)
    for i in range(copies):
        leaves = [(i, j) for j in range(clique_size - 1)]
        G.add_nodes_from(leaves)
        clique_nodes = [center] + leaves
        for u, v in combinations(clique_nodes, 2):
            G.add_edge(u, v)
    return G


def windmill_layout(clique_size: int, copies: int, radius: float = 2.55):
    pos = {"c": (0.0, 0.0)}
    leaf_count = clique_size - 1
    for i in range(copies):
        angle = math.pi / 2 + 2 * math.pi * i / copies
        bx = radius * math.cos(angle)
        by = radius * math.sin(angle)
        local = regular_ngon_positions(range(leaf_count), radius=0.78, phase=angle)
        for j in range(leaf_count):
            lx, ly = local[j]
            pos[(i, j)] = (bx + lx, by + ly)
    return pos


def build_circulant_graph(n_vertices: int, jumps) -> nx.MultiGraph:
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3")
    cleaned_jumps = sorted({int(j) % n_vertices for j in jumps if int(j) % n_vertices != 0})
    if not cleaned_jumps:
        raise ValueError("at least one nonzero jump is required")

    pairs = []
    for i in range(n_vertices):
        for jump in cleaned_jumps:
            pairs.append((i, (i + jump) % n_vertices))
    return build_simple_multigraph(range(n_vertices), pairs)


def circulant_layout(n_vertices: int, jumps, radius: float = 2.45):
    return regular_ngon_positions(range(n_vertices), radius=radius)


def build_generalized_petersen_graph(n_vertices: int, step: int) -> nx.MultiGraph:
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3")
    if step < 1:
        raise ValueError("step must be >= 1")

    outer = [("o", i) for i in range(n_vertices)]
    inner = [("i", i) for i in range(n_vertices)]
    pairs = []
    for i in range(n_vertices):
        pairs.append((("o", i), ("o", (i + 1) % n_vertices)))
        pairs.append((("o", i), ("i", i)))
        pairs.append((("i", i), ("i", (i + step) % n_vertices)))
    return build_simple_multigraph(outer + inner, pairs)


def generalized_petersen_layout(
    n_vertices: int,
    step: int,
    outer_radius: float = 2.65,
    inner_radius: float = 1.45,
):
    pos = regular_ngon_positions([("o", i) for i in range(n_vertices)], radius=outer_radius)
    pos.update(regular_ngon_positions([("i", i) for i in range(n_vertices)], radius=inner_radius))
    return pos


def build_mobius_ladder_graph(n_rungs: int) -> nx.MultiGraph:
    if n_rungs < 2:
        raise ValueError("n_rungs must be >= 2")
    n_vertices = 2 * n_rungs
    pairs = [(i, (i + 1) % n_vertices) for i in range(n_vertices)]
    pairs += [(i, i + n_rungs) for i in range(n_rungs)]
    return build_simple_multigraph(range(n_vertices), pairs)


def mobius_ladder_layout(n_rungs: int, radius: float = 2.45):
    return regular_ngon_positions(range(2 * n_rungs), radius=radius)


def edge_curvatures(count: int, step: float = 0.2) -> list[float]:
    if count <= 1:
        return [0.0]

    curvatures = []
    magnitude = 1
    while len(curvatures) < count:
        curvatures.append(-magnitude * step)
        if len(curvatures) < count:
            curvatures.append(magnitude * step)
        magnitude += 1
    return sorted(curvatures)


def _wrap_edge_curvature(family_name, family_args, u, v):
    if family_name != "cylinder":
        return None

    if not (
        isinstance(u, tuple)
        and isinstance(v, tuple)
        and len(u) == 2
        and len(v) == 2
        and all(isinstance(value, int) for value in u + v)
    ):
        return None

    rows, cols = family_args
    (r1, c1), (r2, c2) = u, v
    row_mid = (rows - 1) / 2

    if r1 == r2 and abs(c1 - c2) == cols - 1:
        row_offset = r1 - row_mid
        sign = -1 if row_offset < 0 else 1
        if row_offset == 0:
            sign = -1 if r1 % 2 == 0 else 1
        magnitude = 0.45 + 0.08 * abs(row_offset) + 0.03 * max(cols - 3, 0)
        return sign * magnitude

    return None


def edge_radii(family_name, family_args, u, v, count: int) -> list[float]:
    base = _wrap_edge_curvature(family_name, family_args, u, v)
    if count == 1:
        return [0.0 if base is None else base]
    offsets = edge_curvatures(count)
    if base is None:
        return offsets
    return [base + offset for offset in offsets]


def loop_geometries(center, count: int) -> list[dict[str, float | tuple[float, float]]]:
    x, y = center

    if count == 1:
        angles = [math.pi / 2]
        loop_radius = 0.26
    elif count == 2:
        angles = [2 * math.pi / 3, math.pi / 3]
        loop_radius = 0.25
    elif count == 3:
        angles = [5 * math.pi / 6, math.pi / 2, math.pi / 6]
        loop_radius = 0.24
    elif count == 4:
        angles = [3 * math.pi / 4, math.pi / 4, -math.pi / 4, -3 * math.pi / 4]
        loop_radius = 0.23
    else:
        angles = [math.pi / 2 + 2 * math.pi * i / count for i in range(count)]
        loop_radius = max(0.18, 0.22 - 0.005 * (count - 5))

    orbit_radius = loop_radius + 0.10
    return [
        {
            "angle": angle,
            "center": (
                x + orbit_radius * math.cos(angle),
                y + orbit_radius * math.sin(angle),
            ),
            "radius": loop_radius,
        }
        for angle in angles
    ]


def _curved_edge_bound(start, end, rad: float) -> tuple[float, float]:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0 or rad == 0:
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    px = -dy / length
    py = dx / length
    bulge = 0.85 * rad * length
    return (mx + bulge * px, my + bulge * py)


def plot_structured_multigraph(
    G: nx.Graph | nx.MultiGraph,
    pos: dict,
    *,
    family_name: str | None = None,
    family_args: tuple = (),
    ax=None,
    node_size: int = 700,
    edge_width: float = 2.2,
    node_color: str = "white",
    node_edge_color: str = "#d62728",
    edge_color: str = "#1f77b4",
    with_labels: bool = True,
    label_size: int = 11,
    show: bool = True,
):
    """Draw a small planar multigraph with visible loops and parallel edges.

    The helper is meant for mathematical Yamada examples, where users need to
    see loops and edge multiplicities rather than a generic spring-layout graph.
    It returns the Matplotlib axis so callers can arrange examples in grids.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    G = ensure_multigraph(G)
    labels = numeric_labels(G)
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(5.8, 5.8))

    edge_groups = defaultdict(list)
    for u, v, key in G.edges(keys=True):
        edge_groups[canonical_pair(u, v)].append((u, v, key))

    bounds = list(pos.values())
    for (u, v), entries in edge_groups.items():
        if u == v:
            for item in loop_geometries(pos[u], len(entries)):
                cx, cy = item["center"]
                radius = item["radius"]
                angle = item["angle"]
                ux = math.cos(angle)
                uy = math.sin(angle)
                start_x = pos[u][0] + 0.12 * ux
                start_y = pos[u][1] + 0.12 * uy
                attach_x = cx - radius * ux
                attach_y = cy - radius * uy
                ax.plot(
                    [start_x, attach_x],
                    [start_y, attach_y],
                    color=edge_color,
                    linewidth=edge_width,
                    alpha=0.95,
                    zorder=1,
                )
                patch = Circle(
                    (cx, cy),
                    radius,
                    fill=False,
                    linewidth=edge_width,
                    edgecolor=edge_color,
                    alpha=0.95,
                    zorder=1,
                )
                ax.add_patch(patch)
                bounds.extend(
                    [
                        (cx - radius, cy - radius),
                        (cx + radius, cy + radius),
                        (start_x, start_y),
                        (attach_x, attach_y),
                    ]
                )
        else:
            start = pos[u]
            end = pos[v]
            for rad in edge_radii(family_name, family_args, u, v, len(entries)):
                patch = FancyArrowPatch(
                    start,
                    end,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    mutation_scale=10,
                    linewidth=edge_width,
                    color=edge_color,
                    alpha=0.92,
                    zorder=1,
                )
                ax.add_patch(patch)
                bounds.append(_curved_edge_bound(start, end, rad))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edge_color,
        linewidths=2.2,
        ax=ax,
    )

    if with_labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_size, ax=ax)

    xs = [point[0] for point in bounds]
    ys = [point[1] for point in bounds]
    pad = 0.7
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if created_ax and show:
        plt.show()
    return ax


GRAPH_FAMILY_CATALOG = {
    "bouquet": GraphFamilySpec(
        builder=build_bouquet_graph,
        layout=bouquet_layout,
        sample_args=(4,),
        note="single-vertex loop bouquet",
    ),
    "theta": GraphFamilySpec(
        builder=build_theta_graph,
        layout=theta_layout,
        sample_args=(5,),
        note="parallel-edge theta family",
    ),
    "periodic_theta": GraphFamilySpec(
        builder=build_periodic_theta_graph,
        layout=periodic_theta_layout,
        sample_args=(4,),
        note="theta strands with midpoint cycle",
    ),
    "sierpinski": GraphFamilySpec(
        builder=build_sierpinski_graph,
        layout=sierpinski_layout,
        sample_args=(3, 2),
        note="recursive S(n,t) family",
    ),
    "complete_graph": GraphFamilySpec(
        builder=build_complete_graph,
        layout=complete_graph_layout,
        sample_args=(4,),
        note="complete graph K_n",
    ),
    "complete_bipartite": GraphFamilySpec(
        builder=build_complete_bipartite_graph,
        layout=complete_bipartite_layout,
        sample_args=(2, 3),
        note="bipartite graph K_{m,n}",
    ),
    "complete_multipartite": GraphFamilySpec(
        builder=build_complete_multipartite_graph,
        layout=complete_multipartite_layout,
        sample_args=(2, 1, 1),
        note="multipartite graph with arbitrary part sizes",
    ),
    "fan": GraphFamilySpec(
        builder=build_fan_graph,
        layout=fan_layout,
        sample_args=(5,),
        note="hub joined to a path",
    ),
    "wheel": GraphFamilySpec(
        builder=build_wheel_graph,
        layout=wheel_layout,
        sample_args=(5,),
        note="hub joined to a cycle",
    ),
    "ladder": GraphFamilySpec(
        builder=build_ladder_graph,
        layout=ladder_layout,
        sample_args=(4,),
        note="two rails connected by rungs",
    ),
    "circular_ladder": GraphFamilySpec(
        builder=build_circular_ladder_graph,
        layout=circular_ladder_layout,
        sample_args=(3,),
        note="circular ladder / prism graph",
    ),
    "grid": GraphFamilySpec(
        builder=build_grid_graph,
        layout=grid_layout,
        sample_args=(2, 3),
        note="rectangular grid",
    ),
    "cylinder": GraphFamilySpec(
        builder=build_cylinder_graph,
        layout=cylinder_layout,
        sample_args=(2, 3),
        note="grid periodic in one direction",
    ),
    "hypercube": GraphFamilySpec(
        builder=build_hypercube_graph,
        layout=hypercube_layout,
        sample_args=(3,),
        note="hypercube Q_d",
    ),
    "friendship": GraphFamilySpec(
        builder=build_friendship_graph,
        layout=friendship_layout,
        sample_args=(3,),
        note="triangles sharing one center",
    ),
    "windmill": GraphFamilySpec(
        builder=build_windmill_graph,
        layout=windmill_layout,
        sample_args=(3, 3),
        note="copies of K_k sharing one vertex",
    ),
    "circulant": GraphFamilySpec(
        builder=build_circulant_graph,
        layout=circulant_layout,
        sample_args=(6, (1, 2)),
        note="circulant graph C(n; jumps)",
    ),
    "generalized_petersen": GraphFamilySpec(
        builder=build_generalized_petersen_graph,
        layout=generalized_petersen_layout,
        sample_args=(5, 2),
        note="generalized Petersen graph G(n,k)",
    ),
    "mobius_ladder": GraphFamilySpec(
        builder=build_mobius_ladder_graph,
        layout=mobius_ladder_layout,
        sample_args=(3,),
        note="cycle with opposite twisted chords",
    ),
}


def build_graph_case(
    family_name: str,
    *args,
    normalize_layout: bool = True,
    target_radius: float = 2.8,
):
    if family_name not in GRAPH_FAMILY_CATALOG:
        raise KeyError(f"unknown family: {family_name}")

    spec = GRAPH_FAMILY_CATALOG[family_name]
    G = spec.builder(*args)
    pos = spec.layout(*args)
    if normalize_layout:
        pos = normalize_positions(pos, target_radius=target_radius)
    return G, pos


def graph_family_names() -> tuple[str, ...]:
    return tuple(GRAPH_FAMILY_CATALOG.keys())


FEATURED_COMPUTE_EXAMPLES = [
    ("bouquet", (4,), "Bouquet with 4 loops"),
    ("theta", (5,), "Theta graph with 5 parallel edges"),
    ("periodic_theta", (4,), "Periodic theta graph with 4 strands"),
    ("sierpinski", (3, 2), "Sierpinski graph S(3,2)"),
]


PRACTICAL_COMPUTE_EXAMPLES = [
    ("complete_graph", (4,), "Complete graph K4"),
    ("complete_bipartite", (2, 3), "Complete bipartite graph K2,3"),
    ("fan", (4,), "Fan graph F4"),
    ("wheel", (4,), "Wheel graph W4"),
    ("ladder", (3,), "Ladder graph with 3 rungs"),
    ("grid", (2, 3), "Grid graph 2x3"),
    ("friendship", (2,), "Friendship graph with 2 triangles"),
]


PLOT_GALLERY_EXAMPLES = [
    ("complete_multipartite", (2, 2, 1), "Complete multipartite K(2,2,1)"),
    ("circular_ladder", (6,), "Circular ladder with 6 rungs"),
    ("cylinder", (3, 5), "Cylinder 3x5"),
    ("hypercube", (4,), "Hypercube Q4"),
    ("windmill", (4, 4), "Windmill of four K4 copies"),
    ("circulant", (9, (1, 3)), "Circulant C(9; 1,3)"),
    ("generalized_petersen", (7, 2), "Generalized Petersen G(7,2)"),
    ("mobius_ladder", (5,), "Mobius ladder M5"),
    ("sierpinski", (3, 3), "Sierpinski graph S(3,3)"),
    ("sierpinski", (5, 2), "Sierpinski graph S(5,2)"),
]


NOTEBOOK_YAMADA_EXAMPLES = [
    ("bouquet", (4,), "Bouquet with 4 loops"),
    ("theta", (5,), "Theta graph with 5 parallel edges"),
    ("periodic_theta", (4,), "Periodic theta graph with 4 strands"),
    ("sierpinski", (3, 2), "Sierpinski graph S(3,2)"),
    ("complete_graph", (4,), "Complete graph K4"),
    ("complete_bipartite", (2, 3), "Complete bipartite graph K2,3"),
    ("complete_multipartite", (2, 1, 1), "Complete multipartite graph K(2,1,1)"),
    ("fan", (4,), "Fan graph F4"),
    ("wheel", (4,), "Wheel graph W4"),
    ("ladder", (3,), "Ladder graph with 3 rungs"),
    ("circular_ladder", (3,), "Circular ladder with 3 rungs"),
    ("grid", (2, 3), "Grid graph 2x3"),
    ("cylinder", (2, 3), "Cylinder graph 2x3"),
    ("friendship", (2,), "Friendship graph with 2 triangles"),
    ("mobius_ladder", (3,), "Mobius ladder M3"),
]


DEFAULT_CUSTOM_EXAMPLE = ("cylinder", (3, 4), "Cylinder 3x4")
