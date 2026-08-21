from __future__ import annotations

import argparse
from pathlib import Path

import certify_theta_symmetric_quandle as sq

ORIGINAL = sq.diagram_constraints


def corrected(graph):
    constraints = ORIGINAL(graph)
    # Every theta role edge is stored and oriented u -> v.  Fix each quandle
    # arc's normal to the left normal of this orientation.  After a tiny local
    # isotopy standardizes the three vertex rays, the normal points
    # counterclockwise at u and clockwise at v.  Jang--Oshiro index the arcs
    # clockwise and use x_i itself precisely when the normal points from
    # e_{i-1} to e_{i+1}, i.e. clockwise.  Thus rho is applied at u and not v.
    constraints["vertices"] = [
        tuple((var, True) for var, _ in constraints["vertices"][0]),
        tuple((var, False) for var, _ in constraints["vertices"][1]),
    ]
    return constraints


sq.diagram_constraints = corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sq.run(args.plantri, args.output)


if __name__ == "__main__":
    main()
