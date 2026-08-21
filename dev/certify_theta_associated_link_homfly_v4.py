from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from knotted_graph.projection import PDCode

import certify_theta_associated_link_homfly as base
import certify_theta_associated_link_homfly_v3 as v3
import certify_theta_complement_group as cg


def cyclic_equal(a, b):
    return any(a[k:] + a[:k] == b for k in range(len(a)))


def convention_crossing_data(graph):
    """Compute (4.1) after imposing Vesnin--Oshmarina's edge convention.

    Their formula assumes CCW order (e3,e2,e1) at v1 and (e1,e2,e3) at v2,
    with all edges oriented v1->v2.  KnottedGraph's role numbers are tracing
    labels only, so we derive the required e_i labels from the actual projected
    cyclic order before summing signed crossings and then map the resulting
    twist parameters back to edge roles for the geometric ribbon builder.
    """
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))

    def role_order(vertex_key):
        vertex = next(v for v in pd.vertices.values() if v.key == vertex_key)
        order = []
        for arc_id in vertex.ccw_ordered_arcs:
            role = int(pd.skeleton_graph.edges[pd.arcs[arc_id].edge_key]["role"])
            if role not in order:
                order.append(role)
        if len(order) != 3:
            raise AssertionError(f"{vertex_key}: invalid theta role order {order}")
        return order

    u_order = role_order("u")
    v_order = role_order("v")
    # Choose u=v1.  u CCW is (e3,e2,e1).
    e_to_role = {2: u_order[0], 1: u_order[1], 0: u_order[2]}
    expected_v = [e_to_role[0], e_to_role[1], e_to_role[2]]
    if not cyclic_equal(v_order, expected_v):
        # Reversing the convention for what PD calls CCW is harmless provided
        # it is done globally.  Try the reflected u order as a diagnostic.
        reflected = [u_order[0], u_order[2], u_order[1]]
        trial = {2: reflected[0], 1: reflected[1], 0: reflected[2]}
        expected = [trial[0], trial[1], trial[2]]
        if not cyclic_equal(v_order, expected):
            raise AssertionError(
                f"vertex cyclic orders incompatible with orientable theta band: u={u_order}, v={v_order}"
            )
        e_to_role = trial
    role_to_e = {role: e for e, role in e_to_role.items()}

    v = np.zeros((3, 3), dtype=int)
    details = []
    for crossing_id, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        incident = [pd.arcs[a] for a in incident_ids]
        by_z = sorted(incident, key=lambda arc: cg._endpoint_z(arc, crossing_id))
        under_in, under_out = cg._incoming_outgoing(by_z[:2], crossing_id)
        over_in, over_out = cg._incoming_outgoing(by_z[2:], crossing_id)
        over_t = cg._outgoing_tangent(over_out, crossing_id)
        under_t = cg._outgoing_tangent(under_out, crossing_id)
        det = float(over_t[0] * under_t[1] - over_t[1] * under_t[0])
        if abs(det) < 1e-12:
            raise AssertionError("degenerate crossing")
        sign = 1 if det > 0 else -1
        role_under = int(pd.skeleton_graph.edges[under_out.edge_key]["role"])
        role_over = int(pd.skeleton_graph.edges[over_out.edge_key]["role"])
        i, j = role_to_e[role_under], role_to_e[role_over]
        v[i, j] += sign
        if i != j:
            v[j, i] += sign
        details.append(
            {"crossing": crossing_id, "roles": [role_under, role_over], "e_indices": [i+1, j+1], "sign": sign}
        )

    m_e = [
        int(-2 * v[0, 0] + v[0, 1] + v[0, 2] - v[1, 2]),
        int(-2 * v[1, 1] + v[0, 1] + v[1, 2] - v[0, 2]),
        int(-2 * v[2, 2] + v[0, 2] + v[1, 2] - v[0, 1]),
    ]
    m_role = [0, 0, 0]
    for e, role in e_to_role.items():
        m_role[role] = m_e[e]
    details.append(
        {"edge_convention": {"u_ccw_roles": u_order, "v_ccw_roles": v_order, "e_to_role": {str(e+1): r for e,r in e_to_role.items()}, "m_e": m_e, "m_by_role": m_role}}
    )
    return v.tolist(), m_role, details


base.crossing_data = convention_crossing_data
base.vertex_pairings = v3.twist_aware_vertex_pairings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
