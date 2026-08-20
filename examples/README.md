# Task 2 Input Format Prototypes

This directory contains examples-level prototypes for making KnottedGraph input
formats more user-friendly across different scientific domains.

For a report-ready overview of the current Task 2 status, see
`TASK2_SUMMARY.md`. For the publication-style input-gallery caption, see
`input_gallery/GALLERY_CAPTION.md`.

The current Main, S1, and S2 publication compositors have a small, stable entry
point in [`input_gallery/task2_figures/`](input_gallery/task2_figures/README.md).
That package is separate from historical gallery experiments and validates all
accepted panel inputs before rendering.

The stable adapters have been migrated into `knotted_graph.inputs`; the broader
examples show how domain data can be adapted into the internal shapes the
downstream library already understands:

- ordered 3D coordinate curves,
- `networkx.MultiGraph` objects with node `pos` and edge `pts`,
- PyVista `PolyData` surface meshes.

## Current Coverage

| Domain | Example folder | Input formats | Prototype output |
| --- | --- | --- | --- |
| Proteins | `proteins/` | RCSB PDB `.pdb` | C-alpha coordinate curve and `MultiGraph(pos/pts)` |
| DNA | `dna/` | RCSB PDB `.pdb` | phosphate coordinate curve and `MultiGraph(pos/pts)` |
| Protein/RNA mmCIF | `mmcif/` | RCSB mmCIF `.cif` | atom trace curve and `MultiGraph(pos/pts)` |
| Polymers | `polymers/` | LAMMPS dump, GROMACS `.gro` | open/closed polymer coordinate curves |
| Generic coordinate chains | `coordinate_chains/` | `.csv`, `.json`, `.tsv`, `.dat`, `.xyz`, `.npy` | open/closed coordinate curves |
| Spatial graphs | `spatial_graphs/` | JSON graph, node/edge CSV | abstract spatial `MultiGraph(pos/pts)` |
| Electric circuits | `electric_circuits/` | JSON spatial network | circuit-like embedded spatial graph |
| Mesh surfaces | `surfaces/` | `.obj`, `.off`, `.ply`, `.stl`, `.vtk`, `.vtp` | PyVista `PolyData` surface |
| Volumetric fields | `volumetric_fields/` | scalar `.npy`, `.npz` | extracted isosurface mesh |
| Fermi surfaces | `fermi_surfaces/` | generated scalar-field surface mesh | PyVista `PolyData` Fermi surface |
| Cross-domain gallery | `input_gallery/` | additional PDB/mmCIF/CSV/XYZ/LAMMPS/GRO/PLY/NPZ examples | validation plots across adapters |

## Spatial Graph CSV Status

Spatial Graph CSV is already supported at the examples/prototype level through
`spatial_graphs/`. This covers engineering-style component/interconnect systems
such as electric circuits, pipe networks, cooling networks, and other embedded
spatial networks.

The prototype converts node/edge CSV inputs into the library's existing internal
spatial-graph representation:

- `networkx.MultiGraph`;
- node attribute `pos = [x, y, z]`;
- edge attribute `pts = [[x0, y0, z0], ..., [x1, y1, z1]]`.

For the later public API, the recommended user-facing schema should be:

- `nodes.csv`: required `node_id,x,y,z`; optional `label,type`;
- `edges.csv`: required `edge_id,source,target`; optional `label,type,points_json`.

The optional `points_json` column is important for spatial topology examples:
without it, an edge is only a straight segment between its endpoint nodes; with
it, a cable, pipe, or wire can be represented by a full embedded 3D polyline.

## Running Smoke Tests

Run scripts from the repository root with the package on `PYTHONPATH`, for
example:

```bash
PYTHONPATH=src python examples/mmcif/plot_mmcif_backbone_examples.py
PYTHONPATH=src python examples/polymers/plot_polymer_snapshot_examples.py
PYTHONPATH=src python examples/spatial_graphs/plot_spatial_graph_csv_examples.py
PYTHONPATH=src python examples/volumetric_fields/plot_volumetric_field_examples.py
PYTHONPATH=src python examples/input_gallery/plot_additional_supported_inputs.py
```

## Current Intent

The current goal is breadth and validation. The stable core API now includes:

```python
from knotted_graph.inputs import from_coordinate_chain
from knotted_graph.inputs import from_spatial_graph_csv
from knotted_graph.inputs import from_surface_mesh
from knotted_graph.inputs import from_protein_ca_backbone
from knotted_graph.inputs import from_nucleic_acid_backbone
from knotted_graph.inputs import from_mmcif_backbone
from knotted_graph.inputs import from_lammps_dump
from knotted_graph.inputs import from_gromacs_gro
```
