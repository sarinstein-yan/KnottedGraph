from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import topoly

import certify_theta_associated_link_homfly as base
import certify_theta_associated_link_homfly_v3 as v3
import certify_theta_associated_link_homfly_v4 as v4
import certify_theta_associated_link_homfly_v6 as v6


def gln_vector(components):
    return np.asarray(
        [
            float(topoly.gln(components[i].tolist(), components[j].tolist()))
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
        dtype=float,
    )


def valid_boundary(graph, m):
    try:
        return v6.fixed_build_boundary(graph, list(map(int, m)), eps=0.005, trim=0.03)
    except AssertionError:
        return None


def linear_zero_linking_crossing_data(graph):
    """Solve the canonical zero-Seifert framing from linking-number response.

    Once an orientable three-boundary-component parity class is fixed, adding
    a full twist to one band (two half-twists) preserves the pair-of-pants
    topology.  Pairwise linking numbers change affinely and integrally with
    those full twists.  We measure the three response columns, solve the 3x3
    system exactly up to numerical rounding, and then verify the proposed
    framing geometrically by recomputing all three linking numbers.
    """
    v, formula, details = v4.convention_crossing_data(graph)
    formula = np.asarray(formula, dtype=int)

    # Find the closest admissible parity-class representative to the published
    # formula, allowing for our global sign/convention differences.
    offsets = list(itertools.product(range(-3, 4), repeat=3))
    offsets.sort(key=lambda d: sum(abs(x) for x in d))
    baseline = None
    baseline_components = None
    for sign in (1, -1):
        center = sign * formula
        for delta in offsets:
            m = center + np.asarray(delta, dtype=int)
            comps = valid_boundary(graph, m)
            if comps is not None:
                baseline = m
                baseline_components = comps
                break
        if baseline is not None:
            break
    if baseline is None or baseline_components is None:
        raise AssertionError(f"could not find orientable baseline near formula={formula.tolist()}")

    L0 = gln_vector(baseline_components)
    response = np.zeros((3, 3), dtype=float)
    for i in range(3):
        shifted = baseline.copy()
        shifted[i] += 2
        comps = valid_boundary(graph, shifted)
        if comps is None:
            raise AssertionError("full twist changed boundary-component count unexpectedly")
        response[:, i] = gln_vector(comps) - L0

    if abs(np.linalg.det(response)) < 1e-8:
        raise AssertionError(
            f"singular full-twist linking response: baseline={baseline.tolist()}, response={response.tolist()}"
        )
    k_real = np.linalg.solve(response, -L0)
    k = np.rint(k_real).astype(int)
    if np.max(np.abs(k_real - k)) > 0.15:
        raise AssertionError(
            f"zero-linking solution not integral in full-twist coordinates: {k_real.tolist()}"
        )
    chosen = baseline + 2 * k
    chosen_components = valid_boundary(graph, chosen)
    if chosen_components is None:
        raise AssertionError("solved framing is not an orientable pair of pants")
    L = gln_vector(chosen_components)
    if np.max(np.abs(L)) > 0.15:
        raise AssertionError(
            f"solved framing failed zero-linking verification: chosen={chosen.tolist()}, gln={L.tolist()}"
        )

    details.append(
        {
            "zero_linking_linear_solve": {
                "formula_by_role": formula.tolist(),
                "baseline_by_role": baseline.tolist(),
                "baseline_gln": L0.tolist(),
                "full_twist_response": response.tolist(),
                "full_twist_coordinates_real": k_real.tolist(),
                "full_twist_coordinates_integer": k.tolist(),
                "chosen_by_role": chosen.tolist(),
                "verified_gln": L.tolist(),
            }
        }
    )
    return v, chosen.tolist(), details


base.build_boundary = v6.fixed_build_boundary
base.vertex_pairings = v3.twist_aware_vertex_pairings
base.crossing_data = linear_zero_linking_crossing_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
