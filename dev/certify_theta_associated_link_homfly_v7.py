from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import topoly

import certify_theta_associated_link_homfly as base
import certify_theta_associated_link_homfly_v3 as v3
import certify_theta_associated_link_homfly_v4 as v4
import certify_theta_associated_link_homfly_v6 as v6


def gln_vector(components):
    return tuple(
        float(topoly.gln(components[i].tolist(), components[j].tolist()))
        for i, j in ((0, 1), (0, 2), (1, 2))
    )


def admissible_zero_linking_crossing_data(graph):
    """Find the zero-Seifert orientable pair-of-pants framing.

    Odd/arbitrary half-twist assignments can change the number of boundary
    components, hence they are not admissible associated-surface framings.
    Such candidates are skipped rather than treated as fatal.  Among genuine
    three-component boundary links we impose the invariant defining condition
    lk(L_i,L_j)=0 for all pairs.
    """
    v, formula_by_role, details = v4.convention_crossing_data(graph)
    bound = max(10, max(abs(x) for x in formula_by_role) + 5)
    candidates = list(itertools.product(range(-bound, bound + 1), repeat=3))
    candidates.sort(
        key=lambda m: min(
            sum(abs(m[i] - formula_by_role[i]) for i in range(3)),
            sum(abs(m[i] + formula_by_role[i]) for i in range(3)),
        )
    )
    solutions = []
    admissible_count = 0
    rejected_boundary_count = 0
    for m in candidates:
        try:
            comps = v6.fixed_build_boundary(graph, list(m), eps=0.005, trim=0.03)
        except AssertionError:
            rejected_boundary_count += 1
            continue
        admissible_count += 1
        gln = gln_vector(comps)
        if all(abs(value) < 0.15 for value in gln):
            solutions.append((m, gln))
            if len(solutions) >= 4:
                break
    if not solutions:
        raise AssertionError(
            f"no admissible zero-linking framing in [-{bound},{bound}]^3; "
            f"formula={formula_by_role}; admissible={admissible_count}; "
            f"rejected_boundary={rejected_boundary_count}"
        )

    chosen, gln = solutions[0]
    details.append(
        {
            "zero_linking_framing": {
                "formula_by_role": formula_by_role,
                "chosen_by_role": list(chosen),
                "gln": list(gln),
                "all_first_solutions": [
                    {"m": list(m), "gln": list(g)} for m, g in solutions
                ],
                "search_bound": bound,
                "admissible_three_component_candidates_checked": admissible_count,
                "non_three_component_candidates_skipped": rejected_boundary_count,
            }
        }
    )
    return v, list(chosen), details


base.build_boundary = v6.fixed_build_boundary
base.vertex_pairings = v3.twist_aware_vertex_pairings
base.crossing_data = admissible_zero_linking_crossing_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
