# Task 2 Summary: User-Friendly Input Formats

Task 2 focuses on making KnottedGraph easier to use with data from different
fields. The current work combines examples-level prototypes with a core input
API. It demonstrates how domain-specific files can be adapted into the internal
geometric objects already used by the library, without changing the downstream
PD-code or Yamada-polynomial algorithms.

## Current Status

The Task 2 prototypes now cover inputs from proteins, DNA, RNA, polymers,
coordinate chains, spatial engineering networks, surface meshes, volumetric
fields, and Fermi-surface-style examples.

The central idea is:

```text
domain file -> lightweight input adapter -> internal geometric object -> common plotting / later topology pipeline
```

The examples currently validate that these inputs can be loaded, converted, and
visualized. They do not claim that every input directly produces a PD code.

## Internal Objects

The prototypes convert external data into three main internal forms:

| Internal object | Used for | Required data |
| --- | --- | --- |
| Ordered 3D coordinate curve | protein backbones, DNA/RNA traces, polymer chains, generic coordinate chains | an `(N, 3)` ordered point array |
| `networkx.MultiGraph(pos/pts)` | spatial graphs, skeleton-like graphs, circuits, pipe/cooling networks | node `pos` and edge `pts` |
| PyVista `PolyData` | surface meshes, extracted isosurfaces, Fermi surfaces | mesh vertices/faces or an extracted surface |

For open curves, the endpoints are interpreted as graph nodes for visualization.
In the publication-style gallery, biological open traces are displayed with a
direct endpoint closure so the rendered curve is a closed loop suitable for
knot-style visual inspection. For genuinely closed curves, no separate endpoint
nodes are added.

## Supported Prototype Inputs

| Domain | Current formats | Example folder | Prototype output |
| --- | --- | --- | --- |
| Proteins | RCSB PDB `.pdb` | `proteins/` | core API: C-alpha coordinate curve and `MultiGraph(pos/pts)` |
| DNA | RCSB PDB `.pdb` | `dna/` | core API: phosphate coordinate curve and `MultiGraph(pos/pts)` |
| Protein/RNA mmCIF | RCSB mmCIF `.cif` | `mmcif/` | core API: atom-trace coordinate curve and `MultiGraph(pos/pts)` |
| Polymers | LAMMPS dump, GROMACS `.gro` | `polymers/` | core API: open/closed polymer coordinate curves |
| Generic coordinate chains | `.csv`, `.json`, `.tsv`, `.dat`, `.xyz`, `.npy` | `coordinate_chains/` | core API: ordered 3D coordinate curves |
| Spatial graphs | JSON graph, node/edge CSV | `spatial_graphs/` | core CSV API / JSON prototype: abstract spatial `MultiGraph(pos/pts)` |
| Electric circuits | JSON spatial network | `electric_circuits/` | circuit-like embedded spatial graph |
| Mesh surfaces | `.obj`, `.off`, `.ply`, `.stl`, `.vtk`, `.vtp` | `surfaces/` | PyVista `PolyData` surface |
| Volumetric fields | scalar `.npy`, `.npz` | `volumetric_fields/` | extracted isosurface mesh |
| Fermi surfaces | generated `.vtp` surface mesh | `fermi_surfaces/` | PyVista `PolyData` surface |
| Cross-domain gallery | PDB/mmCIF/CSV/XYZ/LAMMPS/GRO/PLY/NPZ/VTP | `input_gallery/` | report-style validation figure |

## Spatial Graph CSV

Spatial Graph CSV is supported in the core API and at the
examples/prototype level. It is
intended for engineering-style systems such as electric circuits, pipe networks,
cooling systems, mechanical component networks, and other embedded spatial
networks.

The core API accepts a node/edge CSV pair and converts it into:

```text
networkx.MultiGraph
node attribute: pos = [x, y, z]
edge attribute: pts = [[x0, y0, z0], ..., [x1, y1, z1]]
```

Public schema:

```csv
node_id,x,y,z,label,type
1,0,0,0,Component 1,component
2,1,0,0,Component 2,component
```

```csv
edge_id,source,target,label,type,points_json
e1,1,2,Wire 1,wire,
e2,2,1,Curved Wire,wire,"[[1,0,0],[0.7,0.2,0.8],[0.3,0.2,0.8],[0,0,0]]"
```

If `points_json` is empty, the edge is a straight segment. If `points_json` is
present, it preserves the full embedded 3D path of a wire, cable, or pipe.

## Report Gallery

The current publication-style gallery is available at:

- `examples/input_gallery/figures/task2_input_gallery_publication_style.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.pdf`

The gallery contains 9 validated panels:

1. Knotted Protein PDB
2. DNA Double Helix PDB
3. tRNA mmCIF
4. Trefoil Polymer LAMMPS
5. Cinquefoil XYZ
6. K4 Spatial Graph CSV
7. Trefoil Tube PLY
8. Gyroid Volume NPZ
9. Rashba Fermi VTP

All panels currently report success with no recorded parsing issues in
`examples/input_gallery/data/publication_style_gallery_summary.json`.

## Current Limits

- The stable Task 2 adapters are now migrated into the core
  `src/knotted_graph/inputs` package.
- The formal API covers coordinate chains, spatial graph CSV files, surface
  meshes, protein PDB backbones, DNA/RNA PDB backbones, mmCIF atom traces,
  LAMMPS dump polymer chains, and GROMACS GRO polymer chains.
- The core adapters have lightweight unit tests. The broader examples are still
  smoke tests and gallery validations.
- Protein, DNA, RNA, and polymer examples currently focus on geometric traces,
  not biochemical semantics.
- Surface and volume examples currently demonstrate input conversion and
  visualization, not spatial-graph extraction.
- Inputs do not all directly return PD code. PD-code construction remains a
  downstream step for suitable `MultiGraph(pos/pts)` objects.

## Recommended Next Phase

The public input API is described in `examples/TASK2_API_DESIGN.md`.

The formal input API now exposes:

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

Scalar fields and Fermi surfaces can remain as examples until their schemas and
expected user workflows are clearer.

## Testing

The core adapters are covered by lightweight unit tests under `tests/`.
The most important tested behaviors are:

- valid input conversion;
- missing required columns or fields;
- non-numeric coordinates;
- duplicate node or edge identifiers;
- invalid edge endpoints;
- preservation of labels and type metadata;
- expected internal object shape, especially node `pos` and edge `pts`;
- plotting smoke tests for small representative inputs.
