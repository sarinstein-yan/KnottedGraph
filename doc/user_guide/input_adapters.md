# Input Adapters

<div class="kg-hero">
  <p class="kg-lead">This page orients users to the supported input routes and points to the notebooks where each route is demonstrated. Adapter-specific details should explain what object the user starts from, how it becomes an embedded graph, and which diagnostics should be checked before computing topology.</p>
  <div class="kg-link-row">
    <a href="../../User_guide/01_getting_started.ipynb">Open 01_getting_started.ipynb</a>
  </div>
</div>

## Protein crosslinks

Use `load_crosslinked_protein` for PDB `SSBOND`/`LINK` records or mmCIF
`_struct_conn` rows. It returns normalized crosslink records, parsed atom
coordinates, excluded records, diagnostics, and an embedded `MultiGraph` whose
backbone and crosslink edges retain their polylines.

```python
from knotted_graph.inputs import load_crosslinked_protein

protein = load_crosslinked_protein(
    "5OSQ",
    chain_ids=["A"],
    allowed_crosslink_types={"disulfide", "metal_coordination"},
    data_dir="pdb-cache",
)
```

For a targeted physical motif, pass stable IDs with `crosslink_ids`. Adjacent
peptide links and solvent coordination are classified but excluded by default.
Inspect `protein.issues`, `protein.excluded_crosslinks`, and
`validate_embedding(protein.graph)` before topology analysis.
