from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_theta_collision_frontier_search as frontier
import search_prime_certified_yamada_pairs as prime_search
import discover_yamada_theta_collisions as core


def constituent_record(shadows, descriptor):
    shadow_index = int(descriptor["shadow"])
    bits = int(descriptor["bits"])
    fraction = float(descriptor["approach_fraction"])
    _, raw = core.spatial_theta(shadows[shadow_index], bits, approach_fraction=fraction)
    edges = [np.asarray(points, dtype=float) for points in raw]
    ordered = prime_search.ordered_constituents(edges)
    values = list(ordered.values())
    resolved = all("TMC" not in value for value in values)
    signature = tuple(sorted(values)) if resolved else None
    return {
        "shadow": shadow_index,
        "bits": bits,
        "bitstring": format(bits, "08b"),
        "approach_fraction": fraction,
        "constituents_ordered": ordered,
        "constituent_multiset": list(signature) if signature is not None else None,
        "fully_resolved": resolved,
        "has_unknot_constituent": "0_1" in values,
    }


def run(plantri: str, output: Path, frontier_output: Path):
    corpus = frontier.run(plantri, 8, frontier_output)
    shadows = {s.index: s for s in core.generate_shadows(plantri, 8)}
    geometry_fractions = {int(k): float(v) for k, v in corpus["geometry_fractions"].items()}

    cache = {}
    bucket_results = []
    for bucket_index, bucket in enumerate(corpus["collision_buckets"]):
        members = []
        for row in bucket:
            key = (int(row["shadow"]), int(row["bits"]))
            if key not in cache:
                descriptor = dict(row)
                descriptor["approach_fraction"] = geometry_fractions[key[0]]
                cache[key] = constituent_record(shadows, descriptor)
            members.append(cache[key])

        resolved = [m for m in members if m["fully_resolved"]]
        signatures = {}
        for member in resolved:
            key = tuple(member["constituent_multiset"])
            signatures.setdefault(key, []).append(member)
        distinct = len(signatures) >= 2
        # A pair from two different constituent signatures is immediately
        # non-isotopic as an unlabeled theta-curve.  We prioritize buckets in
        # which both sides also contain an unknot, enabling the independent
        # Calcut--Metcalf-Burton/Thurston prime-lift criterion.
        signature_groups = [
            {
                "signature": list(sig),
                "members": group,
                "has_unknot_member": any(m["has_unknot_constituent"] for m in group),
            }
            for sig, group in sorted(signatures.items(), key=lambda kv: repr(kv[0]))
        ]
        record = {
            "bucket_index": bucket_index,
            "bucket_size": len(bucket),
            "normalized_yamada": bucket[0]["yamada"],
            "resolved_member_count": len(resolved),
            "unresolved_member_count": len(members) - len(resolved),
            "distinct_resolved_constituent_multiset_count": len(signatures),
            "constituent_distinct_nonisotopy_available": distinct,
            "two_unknot_signature_groups_available": sum(
                1 for group in signature_groups if group["has_unknot_member"]
            ) >= 2,
            "signature_groups": signature_groups,
        }
        bucket_results.append(record)
        if distinct:
            print("CONSTITUENT_DISTINCT_BUCKET=" + json.dumps(record, sort_keys=True), flush=True)

    promising = [
        b for b in bucket_results
        if b["constituent_distinct_nonisotopy_available"]
    ]
    promising.sort(
        key=lambda b: (
            not b["two_unknot_signature_groups_available"],
            b["bucket_size"],
            b["bucket_index"],
        )
    )
    payload = {
        "collision_bucket_count": len(bucket_results),
        "unique_collision_member_count": len(cache),
        "constituent_distinct_bucket_count": len(promising),
        "prime_lift_ready_bucket_count": sum(
            bool(b["two_unknot_signature_groups_available"]) for b in promising
        ),
        "promising_buckets": promising,
        "all_buckets": bucket_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("ALL_BUCKET_CONSTITUENT_SUMMARY=" + json.dumps({
        "collision_bucket_count": payload["collision_bucket_count"],
        "unique_collision_member_count": payload["unique_collision_member_count"],
        "constituent_distinct_bucket_count": payload["constituent_distinct_bucket_count"],
        "prime_lift_ready_bucket_count": payload["prime_lift_ready_bucket_count"],
    }, sort_keys=True), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output, args.frontier_output)


if __name__ == "__main__":
    main()
