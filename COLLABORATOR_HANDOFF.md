# KnottedGraph Collaborator Handoff

This repository contains the current KnottedGraph library, the reorganized user-guide notebooks, and the Sphinx website source. Development branches should preserve the public interfaces and validate changes against the repository test and notebook workflows before integration.

## Where to Continue

- Main notebook entry point: `User_guide/00_user_guide.ipynb`
- Core workflows: `User_guide/01_getting_started.ipynb`, `User_guide/02_core_workflows.ipynb`, `User_guide/03_advanced_and_reproduction.ipynb`
- Applications:
  - `User_guide/applications/01_physics_applications.ipynb`
  - `User_guide/applications/02_mathematics_applications.ipynb`
  - `User_guide/applications/03_protein_applications.ipynb`
  - `User_guide/applications/04_analytic_knot_fields.ipynb`
- Correctness/performance notebooks:
  - `User_guide/benchmarks/01_yamada_sanity_checks.ipynb`
  - `User_guide/benchmarks/02_application_regression_checks.ipynb`
  - `User_guide/benchmarks/03_knottedgraph_vs_topoly_scaling.ipynb`
  - `User_guide/benchmarks/04_thick_handlebody_validation.ipynb`
  - `User_guide/benchmarks/05_arbitrary_knot_fields.ipynb`
  - `User_guide/benchmarks/06_subcubic_theta_formula_discovery.ipynb`
  - `User_guide/benchmarks/07_protein_crosslink_topology.ipynb`
- Website source: `doc/`

## Protein Crosslink Workflow

The protein implementation lives in `knotted_graph.inputs.crosslinks` and
`knotted_graph.applications.protein`. Dataset runs use `kg-protein-topology` and
write local resumable artifacts under `results/`, which is intentionally ignored.
The exact workflow reduces each perturbation to a bridgeless cyclic core, records
crossing-cap failures explicitly, and never treats equal Yamada fingerprints as
proof of isotopy.

Repulsor is optional. `scripts/bootstrap_repulsion.py` prepares the pinned public
checkout using HTTPS. Native build failures and before/after topology validation
status are retained in the protein batch JSON. Safe pre/post-decimation checks
every shortcut against non-adjacent segments. On macOS the driver builds against
the system Accelerate framework.

The six proposed scientific directions now have auditable outcomes:

- Direction 1: exact state lattices and abstract-conditioned topology-carrying
  edges are implemented. The complete 13-edge 5OSQ lattice has 8192 successful
  states; 5OSQ has 3/13 carrying edges.
- Direction 2: arbitrary-order strict cooperativity is implemented. 5OSQ has
  19 strict pairs and 3 strict triples. Across 292 exact disulfide-only pairs,
  no strict cooperative pair was observed. The independent
  `disulfide_pair_validation_v1` artifact contains all 207 primary-cohort pairs;
  `complexity_recovery_v1` contains the remaining 85.
- Direction 3: exhaustive and bounded/proven minimum-generator searches are
  implemented. The complete 8192-state 5OSQ run proves `m_top=13`, with only the
  full edge set generating the full fingerprint.
- Direction 4: a Topoly-backed, density-audited local-lasso detector gives six
  same-lasso-multiset/different-global-fingerprint groups. The conclusion is
  scoped to lasso decomposition; no joint knot/theta/handcuff taxonomy is
  claimed.
- Direction 5: exact attributed multigraph isomorphism gives two
  same-connectivity/different-spatial-fingerprint groups.
- Direction 6: the final 114-protein declared recovered exact-evaluable cohort
  and all 398 selected unique disulfide-rewiring nulls succeed. The conditioned
  natural-minus-null carrying-edge fraction is -0.078484, bootstrap 95% CI
  [-0.122403, -0.036190], paired sign-flip `p=0.0002699973`; every conditioned
  inference gate is true. The nested 82-protein no-natural-fallback sensitivity
  analysis preserves the sign and significance (-0.063919, `p=0.00072999`).

Raw `R1` remains zero for every natural and null graph and is retained as an
explicit saturation result. The population claim uses the
abstract-conditioned observable, which holds abstract connectivity fixed. Its
inference is conditional on the first-200, resolution-sorted, 30%-identity
representative query and its exact-evaluable subset—not the whole PDB.

Tracked manifests and the frozen RCSB query are under
`examples/protein_topology/`. The authoritative local numerical summary is
`results/protein_topology/LOCAL_RUN_REPORT.md`; primary machine-readable evidence
is in `pattern_validation_v1/`, `higher_order_validation_v1/`,
`disulfide_pair_validation_v1/`, `minimum_generator_5osq_v1/`,
`population_conditioned_recovered_v1/`, `population_conditioned_v1/`, and
`complexity_recovery_v1/` below that result directory.

## Generated Website Output

`doc/_build/` and `site_preview/` are generated Sphinx build directories and are ignored by Git. This avoids committing duplicate HTML, static assets, and Sphinx caches. The reproducible website source is `doc/` together with the tracked figure assets under `doc/assets/`.

## Rebuild the Local Website Preview

From the repository root:

```bash
uv sync --all-extras --group docs
uv run --group docs python -m sphinx -b html -W --keep-going doc site_preview
open site_preview/index.html
```

If an existing virtual environment is already installed:

```bash
.venv/bin/python -m sphinx -b html -W --keep-going doc site_preview
open site_preview/index.html
```

The built preview is local. Publication to GitHub Pages is handled by `.github/workflows/docs.yml` on the configured deployment branch.

## Validation Before Integration

Run the complete test suite and the repository consistency audit before accepting a branch:

```bash
uv sync --all-extras --group dev --group docs
uv run --no-sync pytest -q
uv run --no-sync python dev/check_repository_consistency.py
```

For changes affecting application outputs, also execute `User_guide/benchmarks/02_application_regression_checks.ipynb` through `dev/execute_notebook.py`.

When the local protein result directories are present, verify every frozen
Directions 1–6 count and inference gate with:

```bash
uv run python dev/audit_protein_topology_science.py
```
