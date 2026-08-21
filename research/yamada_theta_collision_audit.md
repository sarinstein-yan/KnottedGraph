# Audited scope of the theta-curve Yamada collision search

This branch is a mathematical-discovery branch. The current computation is intentionally treated as a **restricted collision search**, not as a complete eight-crossing theta-curve census.

## What has been verified

- The production KnottedGraph Yamada backend is used for the spatial-graph invariant.
- The plantri stage enumerates simple 3-connected plane graphs with two 3-valent vertices and the remaining vertices 4-valent at the requested edge/vertex count.
- Degree sequence alone is not accepted as a theta shadow. Opposite half-edges at each 4-valent crossing vertex are traced and a candidate is retained only when the transition system produces exactly three edge-disjoint strands joining the two trivalent vertices and consuming every shadow edge exactly once.
- A geometric preflight requires the reconstructed spatial theta graph to project back to the intended number of crossings before the Yamada value is accepted.
- Equal Yamada values are compared symbolically, not numerically.
- Candidate non-isotopy is tested independently from Yamada via the multiset of HOMFLY-classified constituent knots. A difference in constituent-knot multisets is sufficient to prove the theta curves are not ambient isotopic.
- The strongest currently clean candidate pair is `(shadow 7, bits 57)` versus `(shadow 13, bits 67)`. Both reconstruct with eight crossings and have identical normalized Yamada polynomial in the current convention, while their constituent-knot multisets are `{-3_1,0_1,0_1}` and `{-3_1,-3_1,0_1}` respectively.

## What has NOT been proved

The current search does **not** establish that the first Yamada collision among prime theta-curves occurs at crossing eight.

Reasons:

1. Moriuchi's enumeration of prime theta-curves uses prime basic theta-polyhedra together with substitutions of algebraic tangles at 4-valent vertices. Eight-crossing diagrams are not exhausted by placing a single crossing at every 4-valent slot of an eight-slot plane graph. Lower-order polyhedra with multi-crossing tangle substitutions also contribute.
2. The current plantri 5.8 command returns 41 degree-sequence candidates, whereas Moriuchi's older research note reports 39. This discrepancy must not be hidden or interpreted as a reproduction of Moriuchi's complete enumeration.
3. The current candidate pairs have not yet been proved prime as theta-curves.
4. An eight-crossing diagram gives only an upper bound on crossing number. A proof of exact crossing number eight requires an independent lower bound or identification in a complete table.
5. Moriuchi explicitly reports that non-prime theta-curves are not completely classified by Yamada/Alexander at low crossing because vertex-connected sums introduce collisions. Therefore the statement "non-isotopic theta-curves can share a Yamada polynomial" is not, by itself, a breakthrough.

## Correct research claim at the present stage

The branch has found reproducible **candidate classical theta-curve Yamada collisions** inside a restricted one-crossing-per-slot family, including pairs independently separated by constituent-knot data.

The next theorem-level target is:

> Find a pair of **prime** non-isotopic theta-curves with the same normalized Yamada polynomial, prove their exact crossing numbers, and determine the minimal crossing number at which such a prime collision occurs.

Only after prime certification plus an exact crossing-number argument should a statement such as `c_Y = 8` be made.

## Recommended execution path

For reproducibility use `dev/run_theta_collision_frontier_search.py` for the restricted sweep and `dev/certify_theta_yamada_collision_candidates.py` for the current independently separated candidate pairs. Do not interpret a zero-collision result from this restricted search as completeness through eight crossings.
