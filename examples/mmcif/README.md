# mmCIF Backbone Inputs

This Task 2 smoke test uses the public RCSB mmCIF input adapter without adding
Biopython or changing downstream topology algorithms.

Supported inputs:

- RCSB `.cif` / mmCIF files.
- Ordered atom traces from the `_atom_site` loop.
- Protein C-alpha curves with `atom_name="CA"`.
- RNA/DNA phosphate curves with `atom_name="P"`.

Outputs:

- downloaded mmCIF files in `data/`,
- extracted coordinate arrays in `data/*.npy`,
- Matplotlib PNG previews,
- Plotly `MultiGraph(pos/pts)` HTML previews.

Run from the repository root:

```bash
PYTHONPATH=src python examples/mmcif/plot_mmcif_backbone_examples.py
```
