from __future__ import annotations

import argparse
import math
from pathlib import Path

import certify_theta_associated_link_homfly as base


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def twist_aware_vertex_pairings(side_paths, center, vertex: str):
    """Glue actual ribbon endpoints through sectors of the vertex disk.

    Half twists can exchange the stored + and - ribbon labels between the two
    ends of an edge.  Therefore we determine physical CW/CCW sides directly
    from the endpoint geometry at each vertex rather than propagating labels.
    For each edge ray, exactly one endpoint lies on its CCW side and one on its
    CW side.  The boundary of the trivalent vertex disk connects the CCW side
    of each ray to the CW side of the next ray in cyclic order.
    """
    index = 0 if vertex == "u" else -1
    rays = []
    physical_side = {}
    for role in range(3):
        plus = side_paths[role][1][index]
        minus = side_paths[role][-1][index]
        midpoint = 0.5 * (plus + minus)
        ray_angle = math.atan2(midpoint[1] - center[1], midpoint[0] - center[0])
        endpoint_angles = {
            1: math.atan2(plus[1] - center[1], plus[0] - center[0]),
            -1: math.atan2(minus[1] - center[1], minus[0] - center[0]),
        }
        diffs = {side: angle_diff(angle, ray_angle) for side, angle in endpoint_angles.items()}
        ccw = max(diffs, key=diffs.get)
        cw = min(diffs, key=diffs.get)
        if diffs[ccw] <= 0 or diffs[cw] >= 0:
            raise AssertionError(
                f"vertex {vertex} role {role}: endpoints do not straddle ray: {diffs}"
            )
        physical_side[role] = {"ccw": ccw, "cw": cw}
        rays.append((ray_angle, role))

    rays.sort()
    order = [role for _, role in rays]
    mapping = {}
    for k, role in enumerate(order):
        nxt = order[(k + 1) % 3]
        a = (role, physical_side[role]["ccw"])
        b = (nxt, physical_side[nxt]["cw"])
        mapping[a] = b
        mapping[b] = a
    if len(mapping) != 6:
        raise AssertionError(f"vertex {vertex}: invalid sector pairing {mapping}")
    return mapping


base.vertex_pairings = twist_aware_vertex_pairings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
