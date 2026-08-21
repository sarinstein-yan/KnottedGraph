from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from knotted_graph.projection import PDCode

import certify_theta_complement_group as cg
import discover_yamada_theta_collisions as core

# Prime, exact-eight, same-Yamada, same-constituent candidate and mirror.
TARGET_PAIRS = [
    ("pair13", (32, 58, 0.12), (39, 153, 0.05)),
    ("pair16", (32, 197, 0.12), (39, 102, 0.05)),
]

# Jang--Oshiro Example 2.3:
# X_Q = Z_2 x F_2[t,t^-1]/(t^2+t+1), |X_Q|=8,
# rho(i,a)=(i+1,a).  Represent F_4 by two bits a0+a1*t with t^2=t+1.
ELEMENTS = tuple((i, a) for i in range(2) for a in range(4))
EINDEX = {x: k for k, x in enumerate(ELEMENTS)}


def f4_add(a: int, b: int) -> int:
    return a ^ b


def f4_mul(a: int, b: int) -> int:
    # polynomial multiplication mod t^2+t+1 over F2
    a0, a1 = a & 1, (a >> 1) & 1
    b0, b1 = b & 1, (b >> 1) & 1
    c0 = a0 * b0
    c1 = (a0 * b1) ^ (a1 * b0)
    c2 = a1 * b1
    # t^2 = t+1
    return (c0 ^ c2) | ((c1 ^ c2) << 1)


T = 2  # polynomial element t
ONE_PLUS_T = 3


def qop_index(x: int, y: int) -> int:
    i, a = ELEMENTS[x]
    j, b = ELEMENTS[y]
    if j == 0:
        value = (i, f4_add(f4_mul(T, a), f4_mul(ONE_PLUS_T, b)))
    else:
        value = (i, f4_add(f4_mul(ONE_PLUS_T, a), f4_mul(T, b)))
    return EINDEX[value]


QOP = [[qop_index(x, y) for y in range(8)] for x in range(8)]
RHO = [EINDEX[(i ^ 1, a)] for i, a in ELEMENTS]


def verify_symmetric_quandle() -> None:
    for x in range(8):
        assert QOP[x][x] == x
    for y in range(8):
        assert len({QOP[x][y] for x in range(8)}) == 8
    for x in range(8):
        for y in range(8):
            assert RHO[QOP[x][y]] == QOP[RHO[x]][y]
            # x ^ rho(y) = x ^ y^{-1}
            rhs = next(z for z in range(8) if QOP[z][y] == x)
            assert QOP[x][RHO[y]] == rhs
            for z in range(8):
                assert QOP[QOP[x][y]][z] == QOP[QOP[x][z]][QOP[y][z]]


class UnionFind:
    def __init__(self, values):
        self.parent = {x: x for x in values}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def edge_orientation_tangent(arc, endpoint: str) -> np.ndarray:
    """Tangent of the original theta edge oriented u -> v at an arc endpoint."""
    pts = np.asarray(arc.line.coords, dtype=float)
    # Core spatial_theta stores each role edge from u to v. Arc LineStrings
    # inherit that order from PDCode._process_edges.
    if endpoint == "start":
        vec = pts[1, :2] - pts[0, :2]
    else:
        vec = pts[-1, :2] - pts[-2, :2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-14:
        raise AssertionError("degenerate arc tangent")
    return vec / norm


def left_normal(tangent: np.ndarray) -> np.ndarray:
    return np.asarray([-tangent[1], tangent[0]], dtype=float)


def diagram_constraints(graph):
    """Convert PDCode into Jang--Oshiro crossing/vertex constraints.

    We fix the normal orientation of every quandle arc to be the left normal
    of the underlying theta edge oriented u->v.  Every symmetric-quandle
    coloring class has exactly one representative with these chosen normals,
    since a basic inversion reverses a normal and applies rho to its color.

    Their arcs terminate only at *under*-crossings and vertices, so PDCode's
    two over-passing arc pieces are first merged at every crossing.
    """
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arc_ids = sorted(pd.arcs)
    uf = UnionFind(arc_ids)

    crossing_local = []
    for xid, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        incident = [pd.arcs[a] for a in incident_ids]
        by_z = sorted(incident, key=lambda arc: cg._endpoint_z(arc, xid))
        under = by_z[:2]
        over = by_z[2:]
        over_in, over_out = cg._incoming_outgoing(over, xid)
        uf.union(over_in.id, over_out.id)
        under_in, under_out = cg._incoming_outgoing(under, xid)
        crossing_local.append((xid, under_in.id, under_out.id, over_in.id))

    roots = sorted({uf.find(a) for a in arc_ids})
    root_to_var = {root: i for i, root in enumerate(roots)}
    var_of_arc = {a: root_to_var[uf.find(a)] for a in arc_ids}

    crossings = []
    for xid, under_in_id, under_out_id, over_id in crossing_local:
        crossing = pd.crossings[xid]
        p = np.asarray(crossing.point.coords[0][:2], dtype=float)
        under_in = pd.arcs[under_in_id]
        under_out = pd.arcs[under_out_id]
        over_arc = pd.arcs[over_id]

        # The fixed normal of the over edge. Use its u->v tangent at crossing.
        if over_arc.start_type == "x" and over_arc.start_id == xid:
            t_over = edge_orientation_tangent(over_arc, "start")
        elif over_arc.end_type == "x" and over_arc.end_id == xid:
            t_over = edge_orientation_tangent(over_arc, "end")
        else:
            raise AssertionError("over arc not incident to crossing")
        n_over = left_normal(t_over)

        def near_point(arc):
            pts = np.asarray(arc.line.coords, dtype=float)
            if arc.start_type == "x" and arc.start_id == xid:
                return pts[1, :2]
            if arc.end_type == "x" and arc.end_id == xid:
                return pts[-2, :2]
            raise AssertionError("under arc not incident to crossing")

        a_id, b_id = under_in_id, under_out_id
        a_proj = float(np.dot(near_point(pd.arcs[a_id]) - p, n_over))
        b_proj = float(np.dot(near_point(pd.arcs[b_id]) - p, n_over))
        if abs(a_proj - b_proj) < 1e-10:
            raise AssertionError("could not order under-arcs by over normal")
        # Jang--Oshiro: over normal points from e0 to e1, and our under
        # normals are coherent because both inherit the same oriented edge.
        e0_id, e1_id = (a_id, b_id) if a_proj < b_proj else (b_id, a_id)
        crossings.append(
            (var_of_arc[e0_id], var_of_arc[over_id], var_of_arc[e1_id])
        )

    vertices = []
    for vertex_key in ("u", "v"):
        vertex = next(v for v in pd.vertices.values() if v.key == vertex_key)
        # PD gives CCW; paper states clockwise. Reverse once globally.
        clockwise_arc_ids = list(reversed(vertex.ccw_ordered_arcs))
        vars_and_prime = []
        center = np.asarray(vertex.point.coords[0][:2], dtype=float)
        rays = []
        for arc_id in clockwise_arc_ids:
            arc = pd.arcs[arc_id]
            pts = np.asarray(arc.line.coords, dtype=float)
            if arc.start_type == "v" and arc.start_id == vertex.id:
                ray = pts[1, :2] - center
                tangent_uv = edge_orientation_tangent(arc, "start")
            elif arc.end_type == "v" and arc.end_id == vertex.id:
                ray = pts[-2, :2] - center
                tangent_uv = edge_orientation_tangent(arc, "end")
            else:
                raise AssertionError("vertex arc incidence mismatch")
            ray /= np.linalg.norm(ray)
            rays.append((arc_id, ray, left_normal(tangent_uv)))

        n = len(rays)
        if n != 3:
            raise AssertionError(f"expected trivalent theta vertex, got {n}")
        for idx, (arc_id, ray, normal) in enumerate(rays):
            prev_ray = rays[(idx - 1) % n][1]
            next_ray = rays[(idx + 1) % n][1]
            # Determine whether the normal points from e_{i-1} toward e_{i+1}.
            # A small displacement along the normal is compared with the two
            # neighboring ray directions by signed angular orientation.
            score = float(np.dot(normal, next_ray - prev_ray))
            if abs(score) < 1e-10:
                # Generic diagrams should avoid equality; use oriented area as fallback.
                score = float(np.cross(prev_ray, normal) + np.cross(normal, next_ray))
            prime_uses_rho = score < 0.0
            vars_and_prime.append((var_of_arc[arc_id], prime_uses_rho))
        vertices.append(tuple(vars_and_prime))

    return {
        "variable_count": len(roots),
        "crossings": crossings,
        "vertices": vertices,
        "pd_code": pd._generate_pd_code(),
    }


def vertex_ok(colors: tuple[int, int, int], rho_flags: tuple[bool, bool, bool]) -> bool:
    operators = [RHO[c] if flag else c for c, flag in zip(colors, rho_flags)]
    for x in range(8):
        value = x
        for op in operators:
            value = QOP[value][op]
        if value != x:
            return False
    return True


def solve_coloring(constraints):
    n = constraints["variable_count"]
    crossings = constraints["crossings"]
    vertices = constraints["vertices"]

    # Precompute allowed triples for each local constraint.
    crossing_allowed = []
    for a, b, c in crossings:
        table = []
        for xa in range(8):
            for xb in range(8):
                table.append((xa, xb, QOP[xa][xb]))
        crossing_allowed.append(((a, b, c), tuple(table)))

    vertex_allowed = []
    for vertex in vertices:
        vars_ = tuple(v for v, _ in vertex)
        flags = tuple(flag for _, flag in vertex)
        table = tuple(
            colors
            for colors in __import__("itertools").product(range(8), repeat=3)
            if vertex_ok(colors, flags)
        )
        vertex_allowed.append((vars_, table))

    local_constraints = crossing_allowed + vertex_allowed
    by_var = defaultdict(list)
    for ci, (vars_, _) in enumerate(local_constraints):
        for v in set(vars_):
            by_var[v].append(ci)

    domains = [set(range(8)) for _ in range(n)]
    solutions = []

    def propagate(domains):
        queue = deque(range(len(local_constraints)))
        queued = set(queue)
        while queue:
            ci = queue.popleft(); queued.discard(ci)
            vars_, allowed = local_constraints[ci]
            viable = [
                tup for tup in allowed
                if all(tup[pos] in domains[var] for pos, var in enumerate(vars_))
            ]
            if not viable:
                return False
            changed_vars = []
            for pos, var in enumerate(vars_):
                supported = {tup[pos] for tup in viable}
                new_domain = domains[var] & supported
                if not new_domain:
                    return False
                if new_domain != domains[var]:
                    domains[var] = new_domain
                    changed_vars.append(var)
            for var in changed_vars:
                for other_ci in by_var[var]:
                    if other_ci not in queued:
                        queue.append(other_ci); queued.add(other_ci)
        return True

    def recurse(domains):
        domains = [set(d) for d in domains]
        if not propagate(domains):
            return
        unresolved = [i for i, d in enumerate(domains) if len(d) > 1]
        if not unresolved:
            solutions.append(tuple(next(iter(d)) for d in domains))
            return
        var = min(unresolved, key=lambda i: len(domains[i]))
        for value in sorted(domains[var]):
            child = [set(d) for d in domains]
            child[var] = {value}
            recurse(child)

    recurse(domains)
    return solutions, {
        "crossing_constraint_count": len(crossings),
        "vertex_allowed_sizes": [len(table) for _, table in vertex_allowed],
    }


def canonical_color_histogram(solutions):
    # The first Z2 coordinate is useful diagnostic structure: count how many
    # variables of each solution lie in each first-factor sheet. This is not
    # needed for invariance; total coloring count is the theorem-level output.
    hist = Counter()
    for sol in solutions:
        sheet_ones = sum(ELEMENTS[c][0] for c in sol)
        hist[sheet_ones] += 1
    return dict(sorted(hist.items()))


def reconstruct(shadows, descriptor):
    shadow, bits, fraction = descriptor
    graph, _ = core.spatial_theta(shadows[shadow], bits, approach_fraction=fraction)
    return graph


def run(plantri: str, output: Path):
    verify_symmetric_quandle()
    shadows = {s.index: s for s in core.generate_shadows(plantri, 8)}
    results = []
    for pair_name, ld, rd in TARGET_PAIRS:
        record = {"pair": pair_name, "sides": {}}
        counts = {}
        for side, desc in (("left", ld), ("right", rd)):
            graph = reconstruct(shadows, desc)
            constraints = diagram_constraints(graph)
            solutions, diagnostics = solve_coloring(constraints)
            side_record = {
                "shadow": desc[0],
                "bits": desc[1],
                "coloring_count": len(solutions),
                "sheet_histogram": canonical_color_histogram(solutions),
                "variable_count": constraints["variable_count"],
                "crossing_count": len(constraints["crossings"]),
                "vertex_constraints": [
                    [[int(v), bool(flag)] for v, flag in vertex]
                    for vertex in constraints["vertices"]
                ],
                "diagnostics": diagnostics,
            }
            record["sides"][side] = side_record
            counts[side] = len(solutions)
        record["symmetric_quandle_count_distinguishes"] = counts["left"] != counts["right"]
        results.append(record)
        print("SYMMETRIC_QUANDLE_RESULT=" + json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "quandle": "Jang-Oshiro Example 2.3 X_Q, |X|=8, rho(i,a)=(i+1,a)",
        "invariant": "number of (X_Q,rho)-colorings of the unoriented spatial graph",
        "pairs": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output)


if __name__ == "__main__":
    main()
