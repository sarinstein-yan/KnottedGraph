# Protein-topology manifests

These manifests freeze every protein identifier, chain, crosslink policy, and
selection caveat used by the protein-topology analyses. They are analytic
cohorts, not estimates of topology prevalence across the entire PDB.

## Manifest roles

- `cohort_v1.csv`: original 20-entry engineering/exploratory cohort, including
  1AOC, 3ULK, and 5OSQ.
- `pattern_validation_v1.csv`: 21 exact-evaluable disulfide proteins selected
  to test local-lasso and exact abstract-connectivity hypotheses.
- `higher_order_validation_v1.csv`: targeted 5OSQ conditioned subset analysis
  through triples.
- `population_conditioned_recovered_v1.csv`: final 114-protein recovered
  exact-evaluable cohort for the sequence-redundancy-controlled population
  analysis.
- `population_conditioned_v1.csv`: nested 82-protein no-natural-fallback
  sensitivity cohort.
- `complexity_recovery_v1.csv`: 32 candidates that exceeded the original
  24-crossing cap and were reanalyzed with certified Repulsor fallback.

The population and recovery manifests derive from the first 200 results of the
frozen query in `rcsb_discovery_query_v2.json`. The query is
sorted by increasing experimental resolution and requests representatives
grouped at 30% sequence identity. It requires 2–8 reported disulfides, 40–300
deposited polymer monomers, one protein polymer entity, experimental structure
determination, and reported resolution at most 2.5 Å. The population inference
is conditional on the recovered exact-evaluable subset of this declared query
result.

## Final population profile

The fold-preserving null uses exact unique non-native perfect matchings of the
same disulfide endpoints. Eligible ensembles are exhaustive when they contain
at most 20 matchings and otherwise use a seeded sample without replacement.
Certificate-checked Repulsor fallback is applied only to natural or null states
that exceed the exact crossing cap.

```bash
export REPULSOR_ROOT="$PWD/external/Repulsor"
uv run kg-protein-topology \
  examples/protein_topology/population_conditioned_recovered_v1.csv \
  results/protein_topology/population_conditioned_recovered_v1 \
  --rotation-samples 32 --max-crossings 40 --n-jobs -1 \
  --no-pairs --exact-subsets none \
  --null-replicates 20 --null-seed 2026 \
  --null-embedding-mode coordinate_preserving \
  --null-sampling-mode unique_disulfide_matchings \
  --repulsion-steps 100 --repulsion-max-time 10 \
  --repulsion-free-special-vertices \
  --repulsion-decimation-passes 16 \
  --repulsion-max-points-per-edge 32 --repulsion-fallback-only \
  --null-repulsion-fallback-steps 100 \
  --null-repulsion-fallback-max-time 10 \
  --null-repulsion-fallback-free-special-vertices \
  --null-repulsion-fallback-decimation-passes 16 \
  --null-repulsion-fallback-max-points-per-edge 32 \
  --repulsor-root external/Repulsor \
  --allow-repulsor-certificate-only \
  --conditioned-robustness --no-resume
```

The verified local result contains 114/114 successful natural analyses and
398/398 successful selected unique nulls. The abstract-conditioned
natural-minus-null topology-carrying-edge fraction is -0.078484 with bootstrap
95% CI [-0.122403, -0.036190] and paired sign-flip `p=0.0002699973`. The nested
82-protein sensitivity analysis gives -0.063919 with bootstrap 95% CI
[-0.101866, -0.028066] and `p=0.00072999`.

## Disulfide pair-validation profile

The no-natural-fallback 82-protein manifest contains 207 distinct order-two
crosslink subsets. This separate run preserves the pair table rather than
overwriting the Direction 6 sensitivity result:

```bash
uv run kg-protein-topology \
  examples/protein_topology/population_conditioned_v1.csv \
  results/protein_topology/disulfide_pair_validation_v1 \
  --rotation-samples 32 --max-crossings 40 --n-jobs -1 \
  --no-pairs --exact-subsets none --null-replicates 0 \
  --conditioned-robustness --conditioned-max-subset-order 2 \
  --no-resume
```

The verified result contains 82/82 successful natural and conditioned analyses,
207/207 unique order-two subsets, six information-carrying pairs, and zero
strictly cooperative pairs. Together with the 85 recovered high-complexity
pairs, the disulfide-only survey contains 292 exact pairs and zero strict pairs.

## High-complexity recovery profile

```bash
uv run kg-protein-topology \
  examples/protein_topology/complexity_recovery_v1.csv \
  results/protein_topology/complexity_recovery_v1 \
  --rotation-samples 32 --max-crossings 40 --n-jobs -1 \
  --exact-subsets none --null-replicates 0 \
  --repulsion-steps 100 --repulsion-max-time 10 \
  --repulsion-free-special-vertices \
  --repulsion-decimation-passes 16 \
  --repulsion-max-points-per-edge 32 --repulsion-fallback-only \
  --repulsor-root external/Repulsor \
  --allow-repulsor-certificate-only \
  --conditioned-robustness --conditioned-max-subset-order 2 \
  --no-resume
```

All 32 candidates were recovered. Safe pre-decimation checks every shortcut's
swept triangle against non-adjacent segments before Repulsor is invoked.

The legacy `canonical_low_crossing` null remains useful for engineering tests
of rewired abstract connectivity, but it is not fold preserving and is not the
primary biological null.
