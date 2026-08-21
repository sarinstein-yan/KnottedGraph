from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import discover_yamada_theta_collisions as core
import search_prime_certified_yamada_pairs as search

TARGETS = [
    ("clean_right", 13, 67, 0.08),
    ("clean_right_mirror", 13, 188, 0.08),
]


def run(plantri: str, output: Path, xyz_dir: Path, rotations: int) -> dict:
    shadows = {s.index: s for s in core.generate_shadows(plantri, 8)}
    xyz_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, shadow, bits, fraction in TARGETS:
        record = search.certify_candidate(
            shadows,
            (shadow, bits, fraction),
            seed=20260821 + 99991,
            rotations=rotations,
            xyz_dir=xyz_dir,
        )
        record["label"] = label
        records.append(record)
        print("CLEAN_RIGHT_PRIME_SEARCH=" + json.dumps(record, sort_keys=True), flush=True)
    payload = {"rotations": rotations, "targets": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--rotations", type=int, default=100000)
    args = parser.parse_args()
    run(args.plantri, args.output, args.xyz_dir, args.rotations)


if __name__ == "__main__":
    main()
