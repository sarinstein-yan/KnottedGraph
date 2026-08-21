from __future__ import annotations

import argparse
import math
from pathlib import Path

import certify_theta_associated_link_homfly as base


def cyclic_vertex_pairings(side_paths, center, vertex: str):
    """Glue ribbon sides through the trivalent vertex disk by cyclic sectors.

    All theta edges are stored oriented u->v.  At u the + ribbon side is the
    left side of the outgoing edge ray, so the CCW sector from edge i to the
    next edge j connects (i,+) to (j,-).  At v the outward ray is opposite to
    the stored edge orientation, hence the same sector connects (i,-) to
    (j,+).  This is the boundary pairing of the blackboard vertex disk and is
    independent of small ribbon width/trim choices.
    """
    index = 0 if vertex == "u" else -1
    rays = []
    for role in range(3):
        plus = side_paths[role][1][index]
        minus = side_paths[role][-1][index]
        midpoint = 0.5 * (plus + minus)
        vec = midpoint[:2] - center[:2]
        rays.append((math.atan2(vec[1], vec[0]), role))
    rays.sort()
    order = [role for _, role in rays]

    mapping = {}
    for k, role in enumerate(order):
        nxt = order[(k + 1) % 3]
        if vertex == "u":
            a, b = (role, 1), (nxt, -1)
        else:
            a, b = (role, -1), (nxt, 1)
        mapping[a] = b
        mapping[b] = a
    if len(mapping) != 6:
        raise AssertionError(f"vertex {vertex}: invalid cyclic pairing {mapping}")
    return mapping


# Replace only the erroneous geometric endpoint matcher.  Every other piece of
# the certificate (Vesnin-Oshmarina twist formula, zero-linking validation,
# repeated geometric trials, Topoly Jones/HOMFLY evaluation) remains unchanged.
base.vertex_pairings = cyclic_vertex_pairings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
