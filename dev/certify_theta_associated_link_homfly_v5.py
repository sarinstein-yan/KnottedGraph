from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import topoly

import certify_theta_associated_link_homfly as base
import certify_theta_associated_link_homfly_v3 as v3
import certify_theta_associated_link_homfly_v4 as v4


def gln_vector(components):
    return tuple(
        float(topoly.gln(components[i].tolist(), components[j].tolist()))
        for i, j in ((0, 1), (0, 2), (1, 2))
    )


def zero_linking_crossing_data(graph):
    """Determine the unique associated-surface framing from its invariant definition.

    The Kauffman--Wolcott--Zhao/Huh associated link is the boundary of the
    pair-of-pants band surface with zero Seifert form; equivalently its three
    oriented boundary components have pairwise linking number zero.  We use
    Vesnin--Oshmarina (4.1) only as the preferred search center, then determine
    the actual twist vector in our coordinate/sign convention directly from
    this defining zero-linking condition.  This removes all ambiguity from
    projection CCW conventions and half-twist sign conventions.
    """
    v, formula_by_role, details = v4.convention_crossing_data(graph)
    bound = max(8, max(abs(x) for x in formula_by_role) + 3)
    candidates = list(itertools.product(range(-bound, bound + 1), repeat=3))
    candidates.sort(
        key=lambda m: min(
            sum(abs(m[i] - formula_by_role[i]) for i in range(3)),
            sum(abs(m[i] + formula_by_role[i]) for i in range(3)),
        )
    )
    solutions = []
    for m in candidates:
        comps = base.build_boundary(graph, list(m), eps=0.005, trim=0.03)
        gln = gln_vector(comps)
        if all(abs(value) < 0.15 for value in gln):
            solutions.append((m, gln))
            # The band surface with zero Seifert form is unique up to ambient
            # isotopy.  A first solution is sufficient for the subsequent link
            # invariant, but record whether nearby framing solutions also occur.
            if len(solutions) >= 2:
                break
    if not solutions:
        raise AssertionError(
            f"no zero-linking associated framing in [-{bound},{bound}]^3; formula={formula_by_role}"
        )
    chosen, gln = solutions[0]
    details.append(
        {
            "zero_linking_framing": {
                "formula_by_role": formula_by_role,
                "chosen_by_role": list(chosen),
                "gln": list(gln),
                "second_solution_if_any": (
                    {"m": list(solutions[1][0]), "gln": list(solutions[1][1])}
                    if len(solutions) > 1
                    else None
                ),
                "search_bound": bound,
            }
        }
    )
    return v, list(chosen), details


base.vertex_pairings = v3.twist_aware_vertex_pairings
base.crossing_data = zero_linking_crossing_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
