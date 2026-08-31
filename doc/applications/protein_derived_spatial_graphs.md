# Protein-Derived Spatial Graphs

<div class="kg-hero">
  <p class="kg-lead">Parse physical PDB/mmCIF crosslinks, extract the cyclic spatial-graph core, scan edge and subset perturbations, compare rewired null models, and optionally validate Repulsor relaxation before exact Yamada evaluation.</p>
  <div class="kg-link-row">
    <a href="../../User_guide/applications/03_protein_applications.ipynb">Open 03_protein_applications.ipynb</a>
  </div>
</div>

## Coordinate-to-graph policy

`load_crosslinked_protein` reads `SSBOND`/`LINK` records from PDB files and
`_struct_conn` rows from mmCIF. The normalized record distinguishes disulfide,
non-backbone covalent, metal coordination, solvent coordination, hydrogen bond,
and ordinary adjacent peptide links. The default graph includes disulfide and
non-backbone covalent crosslinks; metal coordination is opt-in.

Backbone segments are stored as embedded polylines. Crosslinks are separate
multiedges with stable IDs and original endpoint metadata. Metal centers are
represented as nodes when their coordinate atom is present.
When several `LINK` records reuse one ligand atom, small deterministic
general-position lanes separate the residue-level multiedges. Their offsets are
recorded in graph metadata.

```python
from knotted_graph.inputs import load_crosslinked_protein

protein = load_crosslinked_protein(
    "1AOC",
    chain_ids=["A"],
    allowed_crosslink_types={"disulfide"},
    data_dir="pdb-cache",
)
```

## Cyclic core and perturbations

Open protein termini are graph bridges, and one bridge makes the Yamada
polynomial vanish. Protein fingerprint analysis therefore extracts the embedded
2-core, removes remaining single bridges between cyclic blocks, and repeats the
same reduction after every deletion.
If a deletion removes every cycle, the workflow uses the explicit convention
`Y(empty)=1`.

```python
from knotted_graph.applications.protein import (
    FingerprintComputer,
    FingerprintSettings,
    analyze_crosslink_perturbations,
)

computer = FingerprintComputer(
    "results/protein_topology/cache",
    settings=FingerprintSettings(max_crossings=16),
)
analysis = analyze_crosslink_perturbations(
    protein.graph,
    fingerprinter=computer,
    include_pairs=True,
    enumerate_all_subsets=True,
    max_exact_crosslinks=8,
)
```

The result includes single-edge indicators, `f_top`, `R1`, cooperative pairs,
all evaluated subsets, inclusion-minimal changed deletion subsets, and strict
minimum retained fingerprint-generating sets. For crosslink set `E`, the latter
minimizes `|S|` subject to `Y(G_backbone + S) == Y(G_full)`. The incremental
search reports either a proven minimum or a rigorous lower bound. It never
presents a bounded negative search as a minimum. Exact evaluation stops before
the state sum when the selected diagram exceeds `max_crossings`; that outcome
is an explicit error, not an unchanged fingerprint.

Raw deletion fingerprints also see the change in abstract connectivity. The
optional abstract-conditioned analysis compares each observed embedding with a
deterministic low-crossing reference of the same abstract graph. It reports
topology-carrying edges and strict cooperative subsets of arbitrary requested
order. A subset is strictly cooperative only if it removes the baseline excess
embedding topology and no non-empty proper subset does.

## Batch and null models

```bash
uv run kg-protein-topology proteins.csv results/protein_topology/run_01 \
  --rotation-samples 32 --max-crossings 40 \
  --conditioned-robustness --conditioned-max-subset-order 3 \
  --minimum-generator-max-retained-crosslinks 5 \
  --null-replicates 20 --null-seed 2026 \
  --null-embedding-mode coordinate_preserving \
  --null-sampling-mode unique_disulfide_matchings
```

The batch is resumable and writes per-protein JSON, summary/edge/pair CSV files,
the full evaluated state table (`subset_states.csv`), bounded/proven generating
sets, arbitrary-order conditioned subsets, pattern candidates, population
statistics, fingerprint cache records, and figures. Frozen exploratory,
pattern, high-order, population, and complexity-recovery manifests are in
`examples/protein_topology/`.

Two null-sampling policies are explicit. `random_replicates` is retained for
general chemistry. `unique_disulfide_matchings` exactly enumerates eligible
non-native perfect matchings of one intrachain disulfide endpoint set and uses a
seeded sample without replacement only when the eligible ensemble exceeds the
requested cap. This prevents duplicate rewires from masquerading as
replication.

Two null-embedding policies are also explicit. `coordinate_preserving` retains
the folded-protein coordinates and is the primary biological null.
`canonical_low_crossing` tests rewired abstract connectivity in a deterministic
planar/seeded-spring embedding with height-layered edges and positive static
clearance; it is not fold preserving.

Dense coordinate-preserving rewires may exceed the exact crossing cap. The
null-fallback options apply Repulsor only to failed nulls. Safe pre-decimation
and post-decimation test every shortcut against non-adjacent segments, while
accepted Repulsor steps carry a swept-topology certificate. Freeing special
vertices is an explicit option rather than an implicit change in the layout
model.

The optional `crosslink_ids` manifest column accepts `|`-separated stable IDs for
targeted motif scans. Resume files are reused only when both manifest entry and
analysis configuration match.

`crosslink_content_signature` counts chemistry and intra/inter-chain scope and
is only a screening proxy. A separate optional Topoly-backed minimal-surface
adapter computes complete local disulfide-lasso signatures and audits them
across mesh densities. Pattern groups require successful stable lasso analyses.
Exact abstract-connectivity candidates are verified with attributed multigraph
isomorphism rather than a hash match alone. The current motif conclusion is
limited to lasso decomposition; no complete joint knot/theta/handcuff taxonomy
is claimed.

`dataset_statistics.json` refuses a population-inference-ready status when the
sampling design is undeclared, a selected null state fails, the cohort/null
replication requirement is missed, or the statistic has no informative
variation. The verified final 114-protein conditioned run analyzed all 398
selected unique nulls and found a natural-minus-matched-null carrying-edge
fraction of -0.078484 (bootstrap 95% CI [-0.122403, -0.036190], paired
sign-flip `p=0.0002699973`). The nested 82-protein no-natural-fallback
sensitivity analysis gives -0.063919 (bootstrap 95% CI
[-0.101866, -0.028066], `p=0.00072999`). This inference is conditional on the
declared recovered exact-evaluable RCSB-query cohort, not the whole PDB.

An unequal Yamada fingerprint proves a topological distinction under the chosen
convention. Equality is only an equal fingerprint and does not prove isotopy.

<div class="kg-wide-figure">
  <img src="../site_figures/repulsive_curves.png" alt="Repulsive curves workflow for embedded spatial graphs">
</div>
