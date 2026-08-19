"""Focused correctness/performance gate for generalized self-adjacent smoothing."""
from __future__ import annotations

import benchmark_theta_full_recursion_candidate as candidate


def main():
    # Compare the new exact smoother against the independent legacy state-sum
    # result at tractable sizes, then verify the high-crossing emergency fallback
    # has disappeared at n=17.
    for n in (9, 13):
        candidate.run(n, "legacy_smoothing", candidate.ORIGINAL_SMOOTH)
        candidate.run(n, "general_self_adjacent_smoothing", candidate.general_smooth)
    candidate.run(17, "general_self_adjacent_smoothing", candidate.general_smooth)
    candidate.sh._smooth_crossing = candidate.ORIGINAL_SMOOTH


if __name__ == "__main__":
    main()
