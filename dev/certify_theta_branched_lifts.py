from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import topoly
from topoly.params import Closure

import discover_yamada_theta_collisions as core

TARGETS = [
    ("pair6_left", 32, 58, 0.12, 1, 0),
    ("pair6_right", 39, 153, 0.05, 0, 1),
    ("pair9_left", 32, 197, 0.12, 1, 0),
    ("pair9_right", 39, 102, 0.05, 0, 1),
]


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _closed_branch_polygon(edge: np.ndarray, direct: np.ndarray) -> np.ndarray:
    pts = np.vstack([edge[:, :2], direct[-2::-1, :2]])
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def _kernel_polygon(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords[:-1], dtype=float)
    signed_area = 0.5 * np.sum(
        pts[:, 0] * np.roll(pts[:, 1], -1)
        - pts[:, 1] * np.roll(pts[:, 0], -1)
    )
    if signed_area < 0:
        pts = pts[::-1]

    radius = 1000.0
    clipped = [
        np.array([-radius, -radius]),
        np.array([radius, -radius]),
        np.array([radius, radius]),
        np.array([-radius, radius]),
    ]

    def clip_half_plane(poly, a, b):
        out = []
        direction = b - a

        def inside(p):
            return _cross2(direction, p - a) >= -1e-12

        def intersection(p, q):
            segment = q - p
            denominator = _cross2(direction, segment)
            if abs(denominator) < 1e-15:
                return q
            t = -_cross2(direction, p - a) / denominator
            return p + t * segment

        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            p_in, q_in = inside(p), inside(q)
            if p_in and q_in:
                out.append(q)
            elif p_in and not q_in:
                out.append(intersection(p, q))
            elif not p_in and q_in:
                out.extend([intersection(p, q), q])
        return out

    for i, a in enumerate(pts):
        clipped = clip_half_plane(clipped, a, pts[(i + 1) % len(pts)])
        if not clipped:
            break
    if not clipped:
        raise AssertionError("branch polygon is not star-shaped")
    return np.asarray(clipped, dtype=float)


def _ray_boundary_radius(coords: np.ndarray, center: np.ndarray, direction: np.ndarray) -> float:
    pts = np.asarray(coords[:-1], dtype=float)
    hits = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        segment = b - a
        denominator = _cross2(direction, segment)
        if abs(denominator) < 1e-13:
            continue
        ac = a - center
        t = _cross2(ac, segment) / denominator
        u = _cross2(ac, direction) / denominator
        if t > 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
            hits.append(t)
    if not hits:
        raise AssertionError("ray from kernel point missed branch polygon")
    return min(hits)


def _radial_disk_map(xy: np.ndarray, coords: np.ndarray, center: np.ndarray) -> np.ndarray:
    vector = np.asarray(xy, dtype=float) - center
    rho = float(np.linalg.norm(vector))
    if rho < 1e-14:
        return np.zeros(2)
    direction = vector / rho
    boundary_radius = _ray_boundary_radius(coords, center, direction)
    return (rho / boundary_radius) * direction


def _densify(polyline: np.ndarray, samples_per_segment: int) -> np.ndarray:
    points = []
    for a, b in zip(polyline[:-1], polyline[1:]):
        for i in range(samples_per_segment):
            points.append(a + (b - a) * (i / samples_per_segment))
    points.append(polyline[-1])
    return np.asarray(points, dtype=float)


def _angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def construct_lift(
    edge_points: list[np.ndarray],
    *,
    branch_role: int,
    lifted_role: int,
    samples_per_segment: int,
    alpha_offset: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Explicitly construct the knot in the double cover branched over U.

    Here U is the unique unknotted constituent.  Its XY projection is a simple
    polygon.  A point in the polygon kernel gives an explicit radial disk
    homeomorphism to the unit disk.  Inversion at a boundary point sends the
    branch circle to a line, which is placed on the z-axis.  The standard
    branched quotient is then (r,theta,z) -> (r,2*theta,z).  The preimage of
    the third theta edge is the closed knot used in the Thurston primeness
    criterion for theta-curves.
    """
    branch = edge_points[branch_role]
    direct = edge_points[2]
    lifted_edge = edge_points[lifted_role]
    polygon = _closed_branch_polygon(branch, direct)
    kernel = _kernel_polygon(polygon)
    center = kernel.mean(axis=0)

    sampled = _densify(lifted_edge, samples_per_segment)
    mapped = np.asarray(
        [[*_radial_disk_map(point[:2], polygon, center), point[2]] for point in sampled],
        dtype=float,
    )

    endpoint_angles = [
        math.atan2(mapped[i, 1], mapped[i, 0]) % (2.0 * math.pi)
        for i in (0, -1)
    ]
    candidates = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    alpha = float(
        max(
            candidates,
            key=lambda candidate: min(
                _angular_distance(candidate, endpoint) for endpoint in endpoint_angles
            ),
        )
        + alpha_offset
    )

    ca, sa = math.cos(alpha), math.sin(alpha)
    rotation = np.array([[ca, sa], [-sa, ca]])
    transformed = mapped.copy()
    transformed[:, :2] = transformed[:, :2] @ rotation.T

    inversion_center = np.array([1.0, 0.0, 0.0])
    delta = transformed - inversion_center
    norm_sq = np.sum(delta * delta, axis=1)
    if np.min(norm_sq) < 1e-12:
        raise AssertionError("lifted arc approached the inversion point too closely")
    inverted = delta / norm_sq[:, None]

    base = np.column_stack([inverted[:, 2], inverted[:, 0] + 0.5, inverted[:, 1]])
    radii = np.hypot(base[:, 0], base[:, 1])
    if max(radii[0], radii[-1]) > 1e-8:
        raise AssertionError("third-edge endpoints did not land on branch axis")

    non_axis = np.flatnonzero(radii > 1e-8)
    if not len(non_axis):
        raise AssertionError("degenerate lifted arc")
    angles = np.zeros(len(base))
    angles[non_axis] = np.unwrap(np.arctan2(base[non_axis, 1], base[non_axis, 0]))
    angles[: non_axis[0]] = angles[non_axis[0]]
    angles[non_axis[-1] + 1 :] = angles[non_axis[-1]]

    first_sheet = np.column_stack(
        [
            radii * np.cos(angles / 2.0),
            radii * np.sin(angles / 2.0),
            base[:, 2],
        ]
    )
    first_sheet[radii <= 1e-8, :2] = 0.0
    second_sheet = first_sheet.copy()
    second_sheet[:, :2] *= -1.0
    knot = np.vstack([first_sheet, second_sheet[-2:0:-1], first_sheet[0]])
    return knot, {
        "kernel_center": center.tolist(),
        "kernel_vertices": kernel.tolist(),
        "inversion_angle": alpha,
        "samples_per_segment": samples_per_segment,
        "branch_role": branch_role,
        "lifted_role": lifted_role,
        "point_count": len(knot),
    }


def _stable(value):
    if isinstance(value, dict):
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    return str(value)


def knot_invariants(knot: np.ndarray) -> dict:
    kwargs = dict(
        closure=Closure.CLOSED,
        chiral=True,
        run_parallel=False,
        max_cross=30,
    )
    return {
        "homfly": _stable(topoly.homfly(knot.tolist(), **kwargs)),
        "jones": _stable(topoly.jones(knot.tolist(), **kwargs)),
    }


def run(plantri: str, output: Path, xyz_dir: Path) -> dict:
    shadows = core.generate_shadows(plantri, 8)
    by_index = {shadow.index: shadow for shadow in shadows}
    xyz_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for label, shadow_index, bits, fraction, branch_role, lifted_role in TARGETS:
        _, edge_points_raw = core.spatial_theta(
            by_index[shadow_index], bits, approach_fraction=fraction
        )
        edge_points = [np.asarray(points, dtype=float) for points in edge_points_raw]

        trials = []
        for samples, offset in ((12, 0.0), (24, 0.0), (36, 0.0), (24, 0.07), (24, -0.07)):
            knot, metadata = construct_lift(
                edge_points,
                branch_role=branch_role,
                lifted_role=lifted_role,
                samples_per_segment=samples,
                alpha_offset=offset,
            )
            invariants = knot_invariants(knot)
            trials.append({"metadata": metadata, "invariants": invariants})
            if samples == 36 and offset == 0.0:
                np.savetxt(xyz_dir / f"{label}_lift.xyz", knot, fmt="%.16g")

        signatures = {
            json.dumps(trial["invariants"], sort_keys=True, default=str)
            for trial in trials
        }
        if len(signatures) != 1:
            raise AssertionError(f"{label}: branched-lift invariant unstable across trials")
        record = {
            "label": label,
            "shadow": shadow_index,
            "bits": bits,
            "bitstring": format(bits, "08b"),
            "branch_constituent": "0_1",
            "stable_invariants": trials[0]["invariants"],
            "trials": trials,
        }
        records.append(record)
        print("BRANCHED_LIFT=" + json.dumps(record, sort_keys=True), flush=True)

    by_label = {record["label"]: record for record in records}
    pair_results = []
    for pair_name in ("pair6", "pair9"):
        left = by_label[f"{pair_name}_left"]["stable_invariants"]
        right = by_label[f"{pair_name}_right"]["stable_invariants"]
        pair_results.append(
            {
                "pair": pair_name,
                "lift_invariants_distinguish": left != right,
                "left": left,
                "right": right,
            }
        )
        print("BRANCHED_LIFT_PAIR=" + json.dumps(pair_results[-1], sort_keys=True), flush=True)

    payload = {
        "method": (
            "explicit double branched cover over the unique crossing-free unknotted "
            "constituent using a star-shaped disk homeomorphism, inversion to the "
            "branch axis, and the standard angular-doubling quotient"
        ),
        "targets": records,
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
    args = parser.parse_args()
    run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
