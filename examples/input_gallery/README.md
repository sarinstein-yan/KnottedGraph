# Additional Supported-Input Gallery

This folder stress-tests the Task 2 input adapters with additional datasets.
It now uses the formal `knotted_graph.inputs` API where available, while
volume-specific examples remain prototype-level.

The gallery currently renders:

- 1UBQ ubiquitin from RCSB PDB as a protein C-alpha curve,
- 1BNA chain B from RCSB PDB as a DNA phosphate curve,
- 1UBQ from RCSB mmCIF as a protein C-alpha curve,
- 1EHZ from RCSB mmCIF as an RNA phosphate curve,
- additional CSV and XYZ coordinate-chain examples,
- LAMMPS dump and GROMACS `.gro` polymer snapshots,
- a node/edge CSV spatial graph,
- a trefoil tube PLY surface mesh,
- a gyroid-like scalar field `.npz` converted to an isosurface mesh.

Run from the repository root:

```bash
PYTHONPATH=src python examples/input_gallery/plot_additional_supported_inputs.py
```

Outputs are written under `data/` and `figures/`. A machine-readable summary is
written to `data/gallery_summary.json`.

## Publication-Style Summary Figure

To make a 3x3 report figure in the same visual style as the archived
surface-skeletonization figure, run:

```bash
PYTHONPATH=src python examples/input_gallery/plot_publication_style_gallery.py
```

This writes:

- `figures/task2_input_gallery_publication_style.png`
- `figures/task2_input_gallery_publication_style.svg`
- `figures/task2_input_gallery_publication_style.pdf`
- individual panel screenshots under `figures/publication_style_panels/`
- `data/publication_style_gallery_summary.json`
