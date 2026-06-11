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
- JSON, SWC, and GraphML 3D node-graph examples for the revised main figure,
- a genus-2 PLY surface mesh,
- gyroid-like scalar fields and vector-flow `.npz` volumes.

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

To make the comparison version with a raw-input thumbnail inset in each panel,
run:

```bash
TASK2_GALLERY_RAW_INSETS=1 PYTHONPATH=src python examples/input_gallery/plot_publication_style_gallery.py
```

To make the preferred source-domain comparison version with a separate
right-hand source panel for each example, run:

```bash
TASK2_GALLERY_SOURCE_INSETS=1 PYTHONPATH=src python examples/input_gallery/plot_publication_style_gallery.py
```

This writes:

- `figures/task2_input_gallery_publication_style.png`
- `figures/task2_input_gallery_publication_style.svg`
- `figures/task2_input_gallery_publication_style.pdf`
- `figures/task2_input_gallery_publication_style_with_raw_inputs.png`
- `figures/task2_input_gallery_publication_style_with_raw_inputs.svg`
- `figures/task2_input_gallery_publication_style_with_raw_inputs.pdf`
- `figures/task2_input_gallery_publication_style_with_source_insets.png`
- `figures/task2_input_gallery_publication_style_with_source_insets.svg`
- `figures/task2_input_gallery_publication_style_with_source_insets.pdf`
- individual panel screenshots under `figures/publication_style_panels/`
- raw-inset panel screenshots under `figures/publication_style_panels_with_raw_inputs/`
- source-inset panel screenshots under `figures/publication_style_panels_with_source_insets/`
- `data/publication_style_gallery_summary.json`
- `data/publication_style_gallery_with_raw_inputs_summary.json`
- `data/publication_style_gallery_with_source_insets_summary.json`

## Main-Text Figure Selection

The current advisor-feedback pass separates the figure plan into one concise
main-text gallery and grouped appendix galleries.  The main-text selection is
recorded in:

- `MAIN_TEXT_SELECTION.md`
- `main_text_selection.json`

The main figure should show each input type once and should use explicit 3D
node/edge outputs.  Repeated PDB, mmCIF, CSV, polymer, surface, volume, and
Fermi examples should move to appendix/workflow figures.  Surface, volume,
vector-flow, and Fermi examples are still valid Task 2 input types;
surface-like examples should be shown with their resulting graph before
returning to the main figure.

## Main-Text Figure Rendering

The main-text rendering script uses the selection above and writes a new figure
without overwriting the older publication-style gallery:

```bash
PYTHONPATH=src python examples/input_gallery/plot_main_text_input_figure.py
```

On Vanda, submit the PBS wrapper instead of rendering on the login node:

```bash
qsub examples/input_gallery/run_main_text_input_figure.pbs
```

This writes:

- `figures/main_text_input_gallery.png`
- `figures/main_text_input_gallery.svg`
- `figures/main_text_input_gallery.pdf`
- individual panel screenshots under `figures/main_text_panels/`
- `data/main_text_input_gallery_summary.json`

## Appendix Biology Figure

Appendix S1 groups multiple biological examples.  Unlike the main-text figure,
this appendix figure emphasizes broader domain diversity within PDB and mmCIF
inputs.  Each panel shows a source-domain molecular view above the converted
backbone graph, with protein/RNA traces color-coded by residue/index segment and
multi-chain examples color-coded by chain.

```bash
PYTHONPATH=src python examples/input_gallery/plot_appendix_biology_inputs.py
```

On Vanda, submit the PBS wrapper:

```bash
qsub examples/input_gallery/run_appendix_biology_inputs.pbs
```

This writes:

- `figures/appendix_biology_inputs.png`
- `figures/appendix_biology_inputs.svg`
- `figures/appendix_biology_inputs.pdf`
- individual source/converted screenshots under `figures/appendix_biology_panels/`
- `data/appendix_biology_inputs_summary.json`

## Appendix Polymer And Coordinate-Chain Figure

Appendix S2 groups polymer-simulation snapshots and lightweight coordinate-chain
formats.  Each panel shows a source bead/chain view above the converted
graph-compatible curve.  Closed examples are shown as closed loops, while open
coordinate chains show their endpoints as red nodes.  The figure covers the
core coordinate-chain suffixes `.csv`, `.json`, `.npy`, `.tsv`, `.txt`, `.dat`,
and `.xyz`, plus GROMACS `.gro` and LAMMPS dump polymer inputs.

```bash
PYTHONPATH=src python examples/input_gallery/plot_appendix_polymer_coordinate_inputs.py
```

On Vanda, submit the PBS wrapper:

```bash
qsub examples/input_gallery/run_appendix_polymer_coordinate_inputs.pbs
```

This writes:

- `figures/appendix_polymer_coordinate_inputs.png`
- `figures/appendix_polymer_coordinate_inputs.svg`
- `figures/appendix_polymer_coordinate_inputs.pdf`
- individual source/converted screenshots under `figures/appendix_polymer_coordinate_panels/`
- `data/appendix_polymer_coordinate_inputs_summary.json`

## Appendix Spatial Graph Figure

Appendix S3 groups node/edge CSV spatial-graph examples for engineering and
abstract network settings.  Each panel shows a source CSV graph schematic above
the converted `MultiGraph(pos/pts)` graph.  All examples use curved embedded
edge paths through `points_json` so spatial routing is visible.

```bash
PYTHONPATH=src python examples/input_gallery/plot_appendix_spatial_graph_inputs.py
```

On Vanda, submit the PBS wrapper:

```bash
qsub examples/input_gallery/run_appendix_spatial_graph_inputs.pbs
```

This writes:

- `figures/appendix_spatial_graph_inputs.png`
- `figures/appendix_spatial_graph_inputs.svg`
- `figures/appendix_spatial_graph_inputs.pdf`
- individual source/converted screenshots under `figures/appendix_spatial_graph_panels/`
- `data/appendix_spatial_graph_inputs_summary.json`

## Appendix Surface, Volume, Flow, And Fermi Figure

Appendix S4 groups surface mesh, scalar-volume, vector-flow, and
Fermi-surface-style inputs.  Each panel shows the source geometry above a
prototype skeleton/spatial graph or oriented spatial graph result.  Surface and
scalar-volume panels overlay the result on a translucent copy of the source
geometry; the vector-flow panel uses arrows to show orientation.  This addresses
the advisor note that surface-only panels are not informative enough for the
framework.  Surface, volume, flow, and Fermi examples remain organized in this
appendix.

```bash
PYTHONPATH=src python examples/input_gallery/plot_appendix_surface_volume_fermi_inputs.py
```

On Vanda, submit the PBS wrapper:

```bash
qsub examples/input_gallery/run_appendix_surface_volume_fermi_inputs.pbs
```

This writes:

- `figures/appendix_surface_volume_fermi_inputs.png`
- `figures/appendix_surface_volume_fermi_inputs.svg`
- `figures/appendix_surface_volume_fermi_inputs.pdf`
- individual source/skeleton screenshots under `figures/appendix_surface_volume_fermi_panels/`
- `data/appendix_surface_volume_fermi_inputs_summary.json`
