from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from topoly import homfly
from topoly.params import Closure

from knotted_graph.projection import compute_yamada_polynomial

import discover_yamada_theta_collisions as core

A = sp.Symbol("A")

CANDIDATE_PAIRS = [
    ((7, 57, 0.12), (13, 67, 0.08)),
    ((7, 198, 0.12), (13, 188, 0.08)),
    ((20, 57, 0.12), (23, 52, 0.12)),
    ((20, 89, 0.12), (23, 244, 0.12)),
    ((20, 166, 0.12), (23, 11, 0.12)),
    ((20, 198, 0.12), (23, 203, 0.12)),
    ((32, 58, 0.12), (39, 153, 0.05)),
    ((32, 117, 0.12), (39, 117, 0.05)),
    ((32, 138, 0.12), (39, 138, 0.05)),
    ((32, 197, 0.12), (39, 102, 0.05)),
]


def constituent_cycle(edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
    return np.vstack([edge_a, edge_b[-2:0:-1]])


def stable_topoly_value(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def constituent_signature(edge_points: list[np.ndarray]) -> tuple[str, ...]:
    signatures: list[str] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cycle = constituent_cycle(edge_points[i], edge_points[j])
        h = homfly(
            cycle.tolist(),
            closure=Closure.CLOSED,
            chiral=True,
            run_parallel=False,
            max_cross=30,
        )
        signatures.append(stable_topoly_value(h))
    return tuple(sorted(signatures))


def signature_resolved(signature: tuple[str, ...]) -> bool:
    return all("TMC" not in value for value in signature)


def reconstruct(
    shadows: list[core.Shadow], descriptor: tuple[int, int, float]
) -> dict:
    shadow_index, bits, fraction = descriptor
    shadow = {item.index: item for item in shadows}[shadow_index]
    graph, edge_points = core.spatial_theta(
        shadow, bits, approach_fraction=fraction
    )
    result = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=True,
        n_jobs=1,
        crossing_warning_threshold=None,
        return_result=True,
    )
    crossing_count = int(result.projection.num_crossings)
    if crossing_count != 8:
        raise AssertionError(
            f"candidate shadow={shadow_index} bits={bits} reconstructed with "
            f"{crossing_count}, not 8, crossings"
        )
    signature = constituent_signature(edge_points)
    return {
        "shadow": shadow_index,
        "bits": bits,
        "bitstring": format(bits, "08b"),
        "approach_fraction": fraction,
        "yamada": sp.expand(result.polynomial),
        "constituent_homfly_multiset": signature,
        "constituent_homfly_resolved": signature_resolved(signature),
    }


def run(plantri: str, output: Path) -> dict:
    shadows = core.generate_shadows(plantri, 8)
    certified: list[dict] = []
    unresolved: list[dict] = []

    for pair_index, (left_descriptor, right_descriptor) in enumerate(CANDIDATE_PAIRS):
        left = reconstruct(shadows, left_descriptor)
        right = reconstruct(shadows, right_descriptor)
        same_yamada = sp.simplify(sp.expand(left["yamada"] - right["yamada"])) == 0
        if not same_yamada:
            raise AssertionError(
                f"pair {pair_index}: discovery Yamada collision did not reproduce"
            )
        fully_resolved = bool(
            left["constituent_homfly_resolved"]
            and right["constituent_homfly_resolved"]
        )
        different_constituents = bool(
            fully_resolved
            and left["constituent_homfly_multiset"]
            != right["constituent_homfly_multiset"]
        )
        record = {
            "pair_index": pair_index,
            "left": {k: v for k, v in left.items() if k != "yamada"},
            "right": {k: v for k, v in right.items() if k != "yamada"},
            "normalized_yamada": str(left["yamada"]),
            "same_normalized_yamada": same_yamada,
            "constituent_certification_fully_resolved": fully_resolved,
            "different_constituent_homfly_multiset": different_constituents,
        }
        if different_constituents:
            certified.append(record)
            print(
                "CERTIFIED_NONISOTOPIC_COLLISION="
                + json.dumps(record, sort_keys=True)
            )
        else:
            unresolved.append(record)
            print("UNRESOLVED_COLLISION=" + json.dumps(record, sort_keys=True))

    result = {
        "candidate_pair_count": len(CANDIDATE_PAIRS),
        "certified_nonisotopic_pair_count": len(certified),
        "unresolved_pair_count": len(unresolved),
        "certified_pairs": certified,
        "unresolved_pairs": unresolved,
        "certificate_rule": (
            "Only different, fully resolved unordered constituent-HOMFLY multisets "
            "certify non-isotopy. Topoly labels such as TMC are recorded as "
            "inconclusive and never treated as invariant values."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        "SUMMARY="
        + json.dumps(
            {
                "candidate_pair_count": result["candidate_pair_count"],
                "certified_nonisotopic_pair_count": result[
                    "certified_nonisotopic_pair_count"
                ],
                "unresolved_pair_count": result["unresolved_pair_count"],
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output)


if __name__ == "__main__":
    main()
