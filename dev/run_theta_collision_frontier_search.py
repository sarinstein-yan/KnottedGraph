from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from knotted_graph.invariants.yamada.native import native_available

import discover_yamada_theta_collisions as core


def admissible_theta_shadows(
    plantri: str,
    crossings: int,
) -> tuple[list[core.Shadow], list[dict]]:
    candidates = core.generate_shadows(plantri, crossings)
    if crossings == 8 and len(candidates) != 39:
        raise AssertionError(
            f"Moriuchi reports 39 degree-sequence candidates at eight crossings; "
            f"plantri produced {len(candidates)}"
        )

    accepted: list[core.Shadow] = []
    rejected: list[dict] = []
    for shadow in candidates:
        try:
            traces = core.trace_theta_edges(shadow)
        except ValueError as exc:
            rejected.append({"shadow": shadow.index, "reason": str(exc)})
            continue
        if len(traces) != 3:
            rejected.append(
                {"shadow": shadow.index, "reason": f"expected 3 strands, got {len(traces)}"}
            )
            continue
        accepted.append(shadow)
    return accepted, rejected


def run(
    plantri: str,
    crossings: int,
    output: Path,
    *,
    limit_shadows: int | None = None,
    limit_assignments: int | None = None,
) -> dict:
    if not native_available():
        raise RuntimeError("The production native Yamada backend is required")

    all_candidates = core.generate_shadows(plantri, crossings)
    accepted, rejected = admissible_theta_shadows(plantri, crossings)
    if limit_shadows is not None:
        accepted = accepted[:limit_shadows]

    assignment_count = 1 << crossings
    if limit_assignments is not None:
        assignment_count = min(assignment_count, limit_assignments)

    records: list[dict] = []
    geometry_fractions: dict[int, float] = {}
    for position, shadow in enumerate(accepted, start=1):
        fraction = core.choose_safe_approach_fraction(shadow, crossings)
        geometry_fractions[shadow.index] = fraction
        print(
            f"theta shadow {shadow.index}: geometry PASS at fraction={fraction}",
            flush=True,
        )
        for bits in range(assignment_count):
            record, _ = core.assignment_record(
                shadow,
                bits,
                crossings,
                fraction,
            )
            records.append(record)
        print(
            f"theta shadow {position}/{len(accepted)} complete; records={len(records)}",
            flush=True,
        )

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[record["yamada_key"]].append(record)
    collision_buckets = [
        sorted(bucket, key=lambda row: (row["shadow"], row["bits"]))
        for bucket in buckets.values()
        if len(bucket) > 1
    ]
    collision_buckets.sort(
        key=lambda bucket: (-len(bucket), bucket[0]["shadow"], bucket[0]["bits"])
    )

    result = {
        "crossings": crossings,
        "plantri_degree_sequence_candidate_count": len(all_candidates),
        "theta_admissible_shadow_count": len(accepted),
        "rejected_non_theta_shadow_count": len(rejected),
        "rejected_non_theta_shadows": rejected,
        "assignments_per_shadow": assignment_count,
        "diagram_count": len(records),
        "distinct_yamada_count": len(buckets),
        "collision_bucket_count": len(collision_buckets),
        "largest_collision_bucket": max((len(bucket) for bucket in collision_buckets), default=1),
        "geometry_fractions": geometry_fractions,
        "collision_buckets": collision_buckets,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))

    compact = {
        key: value
        for key, value in result.items()
        if key not in {"records", "collision_buckets", "rejected_non_theta_shadows"}
    }
    print("SUMMARY=" + json.dumps(compact, sort_keys=True), flush=True)
    if collision_buckets:
        first = collision_buckets[0]
        print(
            "FIRST_COLLISION_BUCKET="
            + json.dumps(
                {
                    "size": len(first),
                    "yamada": first[0]["yamada"],
                    "members": [
                        {
                            "shadow": row["shadow"],
                            "bits": row["bits"],
                            "bitstring": row["bitstring"],
                        }
                        for row in first[:12]
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--crossings", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-shadows", type=int)
    parser.add_argument("--limit-assignments", type=int)
    args = parser.parse_args()
    run(
        args.plantri,
        args.crossings,
        args.output,
        limit_shadows=args.limit_shadows,
        limit_assignments=args.limit_assignments,
    )


if __name__ == "__main__":
    main()
