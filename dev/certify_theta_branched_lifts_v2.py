from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import certify_theta_branched_lifts as old
import discover_yamada_theta_collisions as core

# (label, shadow, bits, approach, two branch-constituent edge roles, lifted edge)
TARGETS = [
    ("pair6_left", 32, 58, 0.12, (1, 2), 0),
    ("pair6_right", 39, 153, 0.05, (0, 2), 1),
    ("pair9_left", 32, 197, 0.12, (1, 2), 0),
    ("pair9_right", 39, 102, 0.05, (0, 2), 1),
    # The clean-left theta also has a crossing-free XY projection of its
    # constituent e0 U e1, so its branched lift can be certified by the same
    # fully explicit construction.
    ("clean_left", 7, 57, 0.12, (0, 1), 2),
]


def branch_polygon_xyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.vstack([a, b[-2:0:-1]])
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def boundary_hit(
    polygon_xyz: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float]:
    """Return radial boundary distance and interpolated boundary height."""
    xy = polygon_xyz[:, :2]
    hits: list[tuple[float, float]] = []
    for i, a in enumerate(xy[:-1]):
        b = xy[i + 1]
        segment = b - a
        denominator = old._cross2(direction, segment)
        if abs(denominator) < 1e-13:
            continue
        ac = a - center
        t = old._cross2(ac, segment) / denominator
        u = old._cross2(ac, direction) / denominator
        if t > 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
            z = (1.0 - u) * polygon_xyz[i, 2] + u * polygon_xyz[i + 1, 2]
            hits.append((t, float(z)))
    if not hits:
        raise AssertionError("ray from kernel point missed branch polygon")
    return min(hits, key=lambda item: item[0])


def flatten_and_radialize(
    point: np.ndarray,
    polygon_xyz: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Apply one explicit ambient homeomorphism that flattens the branch.

    Along every ray from a kernel point, R(theta) is the distance to the branch
    polygon and h(theta) is its boundary z-height.  The continuous global
    height function

        f(r,theta) = h(theta)*(r/R)                  for r/R <= 1
                   = h(theta)*max(0, 2-r/R)         for r/R >= 1

    agrees exactly with the branch height at r=R and vanishes outside twice
    the radial boundary.  Hence (x,y,z)->(x,y,z-f(x,y)) is an ambient
    homeomorphism flattening the entire branch curve.  The simultaneous radial
    plane homeomorphism r->r/R(theta) sends its projection to the unit circle.
    """
    vector = point[:2] - center
    rho = float(np.linalg.norm(vector))
    if rho < 1e-14:
        return np.array([0.0, 0.0, point[2]], dtype=float)
    direction = vector / rho
    radius, boundary_z = boundary_hit(polygon_xyz, center, direction)
    scaled = rho / radius
    if scaled <= 1.0:
        correction = boundary_z * scaled
    else:
        correction = boundary_z * max(0.0, 2.0 - scaled)
    return np.array([direction[0] * scaled, direction[1] * scaled, point[2] - correction])


def construct_lift(
    edge_points: list[np.ndarray],
    branch_roles: tuple[int, int],
    lifted_role: int,
    samples_per_segment: int,
    alpha_offset: float,
) -> tuple[np.ndarray, dict]:
    polygon_xyz = branch_polygon_xyz(
        edge_points[branch_roles[0]], edge_points[branch_roles[1]]
    )
    polygon_xy = polygon_xyz[:, :2]
    kernel = old._kernel_polygon(polygon_xy)
    center = kernel.mean(axis=0)

    # Explicit sanity certificate: after applying the flattening to every
    # branch vertex, z must vanish and radius must be one.
    branch_mapped = np.asarray(
        [flatten_and_radialize(point, polygon_xyz, center) for point in polygon_xyz]
    )
    branch_radial_error = float(
        np.max(np.abs(np.hypot(branch_mapped[:, 0], branch_mapped[:, 1]) - 1.0))
    )
    branch_height_error = float(np.max(np.abs(branch_mapped[:, 2])))
    if branch_radial_error > 1e-8 or branch_height_error > 1e-8:
        raise AssertionError(
            f"branch flattening failed: radial={branch_radial_error}, z={branch_height_error}"
        )

    sampled = old._densify(edge_points[lifted_role], samples_per_segment)
    mapped = np.asarray(
        [flatten_and_radialize(point, polygon_xyz, center) for point in sampled]
    )

    # From here use the already-audited circle->axis inversion and angular
    # square-root lift.  We reproduce the few lines here so the corrected z
    # coordinate is passed through unchanged.
    endpoint_angles = [
        np.arctan2(mapped[i, 1], mapped[i, 0]) % (2.0 * np.pi) for i in (0, -1)
    ]
    candidates = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    alpha = float(
        max(
            candidates,
            key=lambda candidate: min(
                old._angular_distance(float(candidate), float(endpoint))
                for endpoint in endpoint_angles
            ),
        )
        + alpha_offset
    )
    ca, sa = np.cos(alpha), np.sin(alpha)
    rotation = np.array([[ca, sa], [-sa, ca]])
    transformed = mapped.copy()
    transformed[:, :2] = transformed[:, :2] @ rotation.T

    delta = transformed - np.array([1.0, 0.0, 0.0])
    norm_sq = np.sum(delta * delta, axis=1)
    if np.min(norm_sq) < 1e-12:
        raise AssertionError("lifted arc approached inversion point")
    inverted = delta / norm_sq[:, None]
    base = np.column_stack([inverted[:, 2], inverted[:, 0] + 0.5, inverted[:, 1]])
    radii = np.hypot(base[:, 0], base[:, 1])
    if max(radii[0], radii[-1]) > 1e-8:
        raise AssertionError("third-edge endpoints missed branch axis")

    non_axis = np.flatnonzero(radii > 1e-8)
    angles = np.zeros(len(base))
    angles[non_axis] = np.unwrap(np.arctan2(base[non_axis, 1], base[non_axis, 0]))
    angles[: non_axis[0]] = angles[non_axis[0]]
    angles[non_axis[-1] + 1 :] = angles[non_axis[-1]]
    first = np.column_stack(
        [radii * np.cos(angles / 2.0), radii * np.sin(angles / 2.0), base[:, 2]]
    )
    first[radii <= 1e-8, :2] = 0.0
    second = first.copy()
    second[:, :2] *= -1.0
    knot = np.vstack([first, second[-2:0:-1], first[0]])
    return knot, {
        "branch_roles": list(branch_roles),
        "lifted_role": lifted_role,
        "kernel_center": center.tolist(),
        "kernel_vertices": kernel.tolist(),
        "branch_radial_error": branch_radial_error,
        "branch_height_error": branch_height_error,
        "samples_per_segment": samples_per_segment,
        "alpha_offset": alpha_offset,
        "point_count": len(knot),
    }


def run(plantri: str, output: Path, xyz_dir: Path) -> dict:
    shadows = core.generate_shadows(plantri, 8)
    by_index = {shadow.index: shadow for shadow in shadows}
    xyz_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, shadow_index, bits, fraction, branch_roles, lifted_role in TARGETS:
        _, raw = core.spatial_theta(by_index[shadow_index], bits, approach_fraction=fraction)
        edges = [np.asarray(points, dtype=float) for points in raw]
        trials = []
        for samples, offset in ((12, 0.0), (24, 0.0), (36, 0.0), (24, 0.07), (24, -0.07)):
            knot, metadata = construct_lift(edges, branch_roles, lifted_role, samples, offset)
            invariants = old.knot_invariants(knot)
            trials.append({"metadata": metadata, "invariants": invariants})
            if samples == 36 and offset == 0.0:
                np.savetxt(xyz_dir / f"{label}_corrected_lift.xyz", knot, fmt="%.16g")
        signatures = {json.dumps(t["invariants"], sort_keys=True) for t in trials}
        if len(signatures) != 1:
            raise AssertionError(f"{label}: corrected lift is not stable across trials")
        record = {
            "label": label,
            "shadow": shadow_index,
            "bits": bits,
            "stable_invariants": trials[0]["invariants"],
            "trials": trials,
        }
        records.append(record)
        print("CORRECTED_BRANCHED_LIFT=" + json.dumps(record, sort_keys=True), flush=True)

    by_label = {r["label"]: r for r in records}
    pairs = []
    for name in ("pair6", "pair9"):
        left = by_label[f"{name}_left"]["stable_invariants"]
        right = by_label[f"{name}_right"]["stable_invariants"]
        pairs.append({"pair": name, "distinguished": left != right, "left": left, "right": right})
    payload = {
        "method": "corrected explicit ambient flattening + double-branched-cover lift",
        "targets": records,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
