from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import topoly
from shapely.geometry import LinearRing
from topoly.params import Closure

import certify_theta_branched_lifts as old
import certify_theta_branched_lifts_v2 as lift2
import discover_yamada_theta_collisions as core

# The six collision pairs independently separated by constituent HOMFLY data.
PAIRS = [
    ((7, 57, 0.12), (13, 67, 0.08)),
    ((7, 198, 0.12), (13, 188, 0.08)),
    ((20, 57, 0.12), (23, 52, 0.12)),
    ((20, 89, 0.12), (23, 244, 0.12)),
    ((20, 166, 0.12), (23, 11, 0.12)),
    ((20, 198, 0.12), (23, 203, 0.12)),
]


def cycle_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.vstack([a, b[-2:0:-1]])


def knot_name(points: np.ndarray) -> str:
    return str(
        topoly.homfly(
            points.tolist(),
            closure=Closure.CLOSED,
            chiral=True,
            run_parallel=False,
            max_cross=30,
        )
    )


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def deterministic_rotations(seed: int, count: int):
    yield np.eye(3), (0.0, 0.0, 0.0)
    # A deterministic low-discrepancy-like sweep, followed by seeded samples.
    golden = (1.0 + 5.0**0.5) / 2.0
    for i in range(1, min(count, 600) + 1):
        a = 2.0 * math.pi * ((i / golden) % 1.0)
        b = math.acos(1.0 - 2.0 * ((i * 0.61803398875) % 1.0))
        c = 2.0 * math.pi * ((i * 0.41421356237) % 1.0)
        yield rot_z(c) @ rot_y(b) @ rot_x(a), (a, b, c)
    rng = np.random.default_rng(seed)
    for _ in range(max(0, count - 600)):
        a, c = rng.uniform(0.0, 2.0 * math.pi, size=2)
        b = math.acos(rng.uniform(-1.0, 1.0))
        yield rot_z(c) @ rot_y(b) @ rot_x(a), (float(a), float(b), float(c))


def rotate_edges(edges: list[np.ndarray], matrix: np.ndarray) -> list[np.ndarray]:
    return [edge @ matrix.T for edge in edges]


def simple_star_branch(edges: list[np.ndarray], roles: tuple[int, int]) -> bool:
    polygon = lift2.branch_polygon_xyz(edges[roles[0]], edges[roles[1]])
    ring = LinearRing(polygon[:, :2])
    if not ring.is_simple or not ring.is_valid or abs(ring.area) < 1e-12:
        # LinearRing.area is normally zero; polygon-kernel test below is the
        # actual nondegeneracy certificate.  Keep only simplicity here.
        if not ring.is_simple or not ring.is_valid:
            return False
    try:
        kernel = old._kernel_polygon(polygon[:, :2])
    except Exception:
        return False
    return len(kernel) >= 3 and np.ptp(polygon[:, 0]) > 1e-8 and np.ptp(polygon[:, 1]) > 1e-8


def find_branch_projection(
    edges: list[np.ndarray],
    branch_roles: tuple[int, int],
    *,
    seed: int,
    rotations: int,
):
    for index, (matrix, angles) in enumerate(deterministic_rotations(seed, rotations)):
        rotated = rotate_edges(edges, matrix)
        if simple_star_branch(rotated, branch_roles):
            return rotated, {
                "rotation_index": index,
                "euler_xyz": list(angles),
                "rotation_matrix": matrix.tolist(),
            }
    return None, None


def ordered_constituents(edges: list[np.ndarray]) -> dict[str, str]:
    result = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        result[f"{i}-{j}"] = knot_name(cycle_points(edges[i], edges[j]))
    return result


def certify_candidate(
    shadows: dict[int, core.Shadow],
    descriptor: tuple[int, int, float],
    *,
    seed: int,
    rotations: int,
    xyz_dir: Path,
) -> dict:
    shadow_index, bits, fraction = descriptor
    _, raw = core.spatial_theta(shadows[shadow_index], bits, approach_fraction=fraction)
    edges = [np.asarray(points, dtype=float) for points in raw]
    constituents = ordered_constituents(edges)
    trivial_pairs = [
        tuple(map(int, key.split("-")))
        for key, value in constituents.items()
        if value == "0_1"
    ]
    attempts = []
    for branch_roles in trivial_pairs:
        lifted_role = next(iter({0, 1, 2} - set(branch_roles)))
        rotated, rotation_certificate = find_branch_projection(
            edges,
            branch_roles,
            seed=seed + 1000 * shadow_index + bits,
            rotations=rotations,
        )
        if rotated is None:
            attempts.append(
                {
                    "branch_roles": list(branch_roles),
                    "status": "no_simple_star_projection_found",
                }
            )
            continue

        trials = []
        for samples, offset in ((12, 0.0), (24, 0.0), (36, 0.0), (24, 0.07), (24, -0.07)):
            knot, metadata = lift2.construct_lift(
                rotated, branch_roles, lifted_role, samples, offset
            )
            invariants = old.knot_invariants(knot)
            trials.append({"metadata": metadata, "invariants": invariants})
            if samples == 36 and offset == 0.0:
                np.savetxt(
                    xyz_dir / f"shadow{shadow_index}_bits{bits}_branch{branch_roles[0]}{branch_roles[1]}_lift.xyz",
                    knot,
                    fmt="%.16g",
                )
        signatures = {json.dumps(t["invariants"], sort_keys=True) for t in trials}
        if len(signatures) != 1:
            attempts.append(
                {
                    "branch_roles": list(branch_roles),
                    "status": "unstable_lift_invariant",
                    "rotation": rotation_certificate,
                    "trials": trials,
                }
            )
            continue
        attempts.append(
            {
                "branch_roles": list(branch_roles),
                "lifted_role": lifted_role,
                "status": "certified_stable_lift",
                "rotation": rotation_certificate,
                "stable_invariants": trials[0]["invariants"],
                "trials": trials,
            }
        )

    return {
        "shadow": shadow_index,
        "bits": bits,
        "bitstring": format(bits, "08b"),
        "constituents_ordered": constituents,
        "trivial_constituent_pairs": [list(pair) for pair in trivial_pairs],
        "branch_attempts": attempts,
    }


def run(plantri: str, output: Path, xyz_dir: Path, rotations: int) -> dict:
    shadows_list = core.generate_shadows(plantri, 8)
    shadows = {shadow.index: shadow for shadow in shadows_list}
    xyz_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    pair_results = []
    for pair_index, (left_desc, right_desc) in enumerate(PAIRS):
        members = []
        for desc in (left_desc, right_desc):
            key = desc[:2]
            if key not in cache:
                cache[key] = certify_candidate(
                    shadows,
                    desc,
                    seed=20260821,
                    rotations=rotations,
                    xyz_dir=xyz_dir,
                )
            members.append(cache[key])
        record = {"pair_index": pair_index, "left": members[0], "right": members[1]}
        pair_results.append(record)
        print("PRIME_SEARCH_PAIR=" + json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "rotation_search_count_per_trivial_constituent": rotations,
        "pairs": pair_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--rotations", type=int, default=5000)
    args = parser.parse_args()
    run(args.plantri, args.output, args.xyz_dir, args.rotations)


if __name__ == "__main__":
    main()
