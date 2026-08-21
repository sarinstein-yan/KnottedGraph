from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from knotted_graph.projection import PDCode

import certify_theta_complement_group as cg
import certify_theta_symmetric_quandle as sq


def corrected_diagram_constraints(graph):
    """Jang--Oshiro constraints with an exact vertex-side normal test."""
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arc_ids = sorted(pd.arcs)
    uf = sq.UnionFind(arc_ids)

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
        over_arc = pd.arcs[over_id]
        if over_arc.start_type == "x" and over_arc.start_id == xid:
            t_over = sq.edge_orientation_tangent(over_arc, "start")
        elif over_arc.end_type == "x" and over_arc.end_id == xid:
            t_over = sq.edge_orientation_tangent(over_arc, "end")
        else:
            raise AssertionError("over arc not incident to crossing")
        n_over = sq.left_normal(t_over)

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
        e0_id, e1_id = (a_id, b_id) if a_proj < b_proj else (b_id, a_id)
        crossings.append((var_of_arc[e0_id], var_of_arc[over_id], var_of_arc[e1_id]))

    vertices = []
    for vertex_key in ("u", "v"):
        vertex = next(v for v in pd.vertices.values() if v.key == vertex_key)
        clockwise_arc_ids = list(reversed(vertex.ccw_ordered_arcs))
        center = np.asarray(vertex.point.coords[0][:2], dtype=float)
        rays = []
        for arc_id in clockwise_arc_ids:
            arc = pd.arcs[arc_id]
            pts = np.asarray(arc.line.coords, dtype=float)
            if arc.start_type == "v" and arc.start_id == vertex.id:
                ray = pts[1, :2] - center
                tangent_uv = sq.edge_orientation_tangent(arc, "start")
            elif arc.end_type == "v" and arc.end_id == vertex.id:
                ray = pts[-2, :2] - center
                tangent_uv = sq.edge_orientation_tangent(arc, "end")
            else:
                raise AssertionError("vertex arc incidence mismatch")
            ray = ray / np.linalg.norm(ray)
            normal = sq.left_normal(tangent_uv)
            rays.append((arc_id, ray, normal))

        if len(rays) != 3:
            raise AssertionError("theta vertex is not trivalent")
        vars_and_prime = []
        for idx, (arc_id, ray, normal) in enumerate(rays):
            prev_ray = rays[(idx - 1) % 3][1]
            next_ray = rays[(idx + 1) % 3][1]
            left_of_edge = sq.left_normal(ray)
            prev_side = float(np.sign(np.cross(ray, prev_ray)))
            next_side = float(np.sign(np.cross(ray, next_ray)))
            if prev_side == 0 or next_side == 0 or prev_side == next_side:
                raise AssertionError(
                    f"non-generic trivalent projection at {vertex_key}: sides {prev_side},{next_side}"
                )
            # Desired normal points from the half-plane containing e_{i-1}
            # to the half-plane containing e_{i+1}.  The left half-plane has
            # sign +1 under cross(ray, neighbor), the right has sign -1.
            desired = next_side * left_of_edge
            points_prev_to_next = float(np.dot(normal, desired)) > 0.0
            prime_uses_rho = not points_prev_to_next
            vars_and_prime.append((var_of_arc[arc_id], prime_uses_rho))
        vertices.append(tuple(vars_and_prime))

    return {
        "variable_count": len(roots),
        "crossings": crossings,
        "vertices": vertices,
        "pd_code": pd._generate_pd_code(),
    }


sq.diagram_constraints = corrected_diagram_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sq.run(args.plantri, args.output)


if __name__ == "__main__":
    main()
