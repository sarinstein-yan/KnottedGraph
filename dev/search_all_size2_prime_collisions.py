from __future__ import annotations

import argparse
from pathlib import Path

import search_prime_certified_yamada_pairs as search

# Complete list of the 17 size-two collision buckets in the audited restricted
# eight-crossing corpus.  The descriptors are (shadow, bits, safe fraction).
ALL_SIZE_TWO_PAIRS = [
    ((7, 57, 0.12), (13, 67, 0.08)),
    ((7, 198, 0.12), (13, 188, 0.08)),
    ((9, 7, 0.05), (9, 121, 0.05)),
    ((9, 134, 0.05), (9, 248, 0.05)),
    ((15, 102, 0.12), (15, 153, 0.12)),
    ((20, 57, 0.12), (23, 52, 0.12)),
    ((20, 89, 0.12), (23, 244, 0.12)),
    ((20, 166, 0.12), (23, 11, 0.12)),
    ((20, 198, 0.12), (23, 203, 0.12)),
    ((25, 27, 0.12), (25, 89, 0.12)),
    ((25, 59, 0.12), (25, 93, 0.12)),
    ((25, 162, 0.12), (25, 196, 0.12)),
    ((25, 166, 0.12), (25, 228, 0.12)),
    ((32, 58, 0.12), (39, 153, 0.05)),
    ((32, 117, 0.12), (39, 117, 0.05)),
    ((32, 138, 0.12), (39, 138, 0.05)),
    ((32, 197, 0.12), (39, 102, 0.05)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--rotations", type=int, default=2000)
    args = parser.parse_args()
    search.PAIRS = ALL_SIZE_TWO_PAIRS
    search.run(args.plantri, args.output, args.xyz_dir, args.rotations)


if __name__ == "__main__":
    main()
