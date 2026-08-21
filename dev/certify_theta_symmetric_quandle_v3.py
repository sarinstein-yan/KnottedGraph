from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import certify_theta_symmetric_quandle as sq
import certify_theta_symmetric_quandle_v2 as sq2


def standardized_vertex_constraints(graph):
    """Use Jang--Oshiro coloring conditions in local standard vertex form.

    Start from v2 for crossing constraints and quandle-arc construction, but
    replace the vertex rho flags by their exact local-isotopy interpretation.
    For arcs indexed clockwise around a vertex, 'the normal points from
    e_{i-1} to e_{i+1}' means that the normal points clockwise across e_i.
    A tiny disk isotopy can make the three rays evenly spaced without changing
    the diagram class, so this criterion is independent of their raw projected
    angular spacing.
    """
    constraints = sq2.corrected_diagram_constraints.__wrapped__(graph) if hasattr(sq2.corrected_diagram_constraints, "__wrapped__") else None
    # We cannot reuse v2 because it intentionally rejects non-separated raw
    # ray configurations. Rebuild from the original routine and then correct
    # the flags by the intrinsic edge orientation: every role edge is stored
    # u->v and its fixed normal is the left normal. Hence at u the outward ray
    # sees a counterclockwise normal (rho is required), while at v the u->v
    # tangent is inward and its left normal is clockwise (rho is not required).
    base = sq.diagram_constraints_original(graph) if hasattr(sq, "diagram_constraints_original") else None
    if base is None:
        # The original routine has correct arc merging and crossing equations;
        # only its vertex flag heuristic is replaced below.
        base = sq._original_diagram_constraints(graph) if hasattr(sq, "_original_diagram_constraints") else sq.diagram_constraints(graph)
    vertices = []
    for vertex_index, vertex in enumerate(base["vertices"]):
        use_rho = vertex_index == 0  # run() constructs vertices in order u,v
        vertices.append(tuple((var, use_rho) for var, _ in vertex))
    base["vertices"] = vertices
    return base


# Preserve the unpatched v1 function explicitly before monkey-patching.
if not hasattr(sq, "_original_diagram_constraints"):
    sq._original_diagram_constraints = sq.diagram_constraints


def corrected(graph):
    base = sq._original_diagram_constraints(graph)
    base["vertices"] = [
        tuple((var, True) for var, _ in base["vertices"][0]),
        tuple((var, False) for var, _ in base["vertices"][1]),
    ]
    return base


sq.diagram_constraints = corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sq.run(args.plantri, args.output)


if __name__ == "__main__":
    main()
