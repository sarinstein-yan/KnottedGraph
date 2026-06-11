# Task 2 Summary: User-Friendly Input Formats

Task 2 focuses on making KnottedGraph easier to use with data from different
fields. The current work combines examples-level prototypes with a core input
API. It demonstrates how domain-specific files can be adapted into the internal
geometric objects already used by the library, without changing the downstream
PD-code or Yamada-polynomial algorithms.

## Current Status

The Task 2 prototypes now cover inputs from proteins, DNA, RNA, polymers,
coordinate chains, spatial engineering networks, surface meshes, volumetric
fields, vector-flow volumes, and Fermi-surface-style examples.

Advisor follow-up: the gallery should show as many possible user-facing input
types as practical.  Surface meshes, scalar volumes, and Fermi-surface-style
geometries remain valid Task 2 input types and should stay in appendix/workflow
figures even if they are not selected for the graph-only main text figure.  The
final Appendix S4 plan uses a genus-2 surface mesh together with the other
surface, volume, vector-flow, and Fermi examples.

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
| Mesh surfaces | `.obj`, `.off`, `.ply`, `.stl`, `.vtk`, `.vtp` | `surfaces/` | PyVista `PolyData`; S4 prototype skeleton graph |
| Volumetric fields | scalar `.npy`, `.npz` | `volumetric_fields/` | extracted isosurface mesh; S4 prototype skeleton graph |
| Vector-flow volumes | vector-field `.npz` | `input_gallery/` | prototype oriented spatial graph |
| Fermi surfaces | generated `.vtp` surface mesh | `fermi_surfaces/` | PyVista `PolyData`; S4 prototype skeleton graph |
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

The advisor-feedback main-text gallery is now available at:

- `examples/input_gallery/figures/main_text_input_gallery.png`
- `examples/input_gallery/figures/main_text_input_gallery.svg`
- `examples/input_gallery/figures/main_text_input_gallery.pdf`

This main-text gallery prioritizes input-format diversity and shows each input
type once:

1. Protein Backbone (\texttt{PDB})
2. tRNA (\texttt{mmCIF})
3. Ring Polymer (\texttt{GRO})
4. Polymer (\texttt{LAMMPS})
5. Cinquefoil Knot (\texttt{XYZ})
6. Engineering Network (\texttt{CSV})
7. Spatial Graph (\texttt{JSON})
8. Neuron Morphology (\texttt{SWC})
9. Spatial Network (\texttt{GraphML})

Its machine-readable validation summary is:

- `examples/input_gallery/data/main_text_input_gallery_summary.json`

Appendix S1, the grouped biological-input figure, is available at:

- `examples/input_gallery/figures/appendix_biology_inputs.png`
- `examples/input_gallery/figures/appendix_biology_inputs.svg`
- `examples/input_gallery/figures/appendix_biology_inputs.pdf`

This appendix figure shows multiple PDB and mmCIF examples:

1. Crambin PDB
2. Ubiquitin PDB
3. Protein Backbone PDB
4. Hemoglobin PDB
5. B-DNA Duplex PDB
6. tRNA mmCIF
7. Ubiquitin mmCIF

Each panel displays a source-domain molecular view above the converted backbone
graph.  Protein/RNA traces are color-coded by residue/index segment, while
multi-chain examples are color-coded by chain.  Yamada-polynomial values are not
printed inside each compact panel; their status remains pending downstream
audit.

Its machine-readable validation summary is:

- `examples/input_gallery/data/appendix_biology_inputs_summary.json`

Appendix S2, the grouped polymer and coordinate-chain figure, is available at:

- `examples/input_gallery/figures/appendix_polymer_coordinate_inputs.png`
- `examples/input_gallery/figures/appendix_polymer_coordinate_inputs.svg`
- `examples/input_gallery/figures/appendix_polymer_coordinate_inputs.pdf`

This appendix figure shows multiple polymer and coordinate-chain examples:

1. Ring Polymer (\texttt{GRO})
2. Polymer (\texttt{LAMMPS})
3. Trefoil Polymer (\texttt{LAMMPS})
4. Bottlebrush Polymer (\texttt{XYZ})
5. Coiled Cable (\texttt{DAT})
6. Cinquefoil Knot (\texttt{XYZ})
7. Figure-Eight Knot (\texttt{XYZ})
8. Sensor Trace (\texttt{CSV})
9. Lissajous Loop (\texttt{JSON})
10. Ribbon Loop (\texttt{NPY})
11. Meander Path (\texttt{TSV})
12. Plain Text Cable (\texttt{TXT})

Each panel displays a source bead/chain view above the converted
graph-compatible curve.  Closed examples are rendered as closed loops, while
open coordinate chains mark endpoints as red graph nodes.  This appendix now
covers the lightweight coordinate-chain suffixes supported by the core API:
`.csv`, `.json`, `.npy`, `.tsv`, `.txt`, `.dat`, and `.xyz`, alongside polymer
simulation formats.  Yamada-polynomial values are omitted from the compact
panels until downstream audit is complete.

Its machine-readable validation summary is:

- `examples/input_gallery/data/appendix_polymer_coordinate_inputs_summary.json`

Appendix S3, the grouped spatial-graph CSV figure, is available at:

- `examples/input_gallery/figures/appendix_spatial_graph_inputs.png`
- `examples/input_gallery/figures/appendix_spatial_graph_inputs.svg`
- `examples/input_gallery/figures/appendix_spatial_graph_inputs.pdf`

This appendix figure shows multiple node/edge CSV examples:

1. Engineering Network CSV
2. Pipe Manifold CSV
3. Circuit Harness CSV
4. Cooling Network CSV
5. Vascular Branch CSV
6. Lattice Truss CSV
7. Hopf Link CSV
8. Three-Ring Link CSV

Each panel displays a source CSV graph schematic above the converted
`MultiGraph(pos/pts)` graph.  All examples use curved `points_json` edge paths,
not only straight source-target segments, so embedded spatial routing is visible.
Yamada-polynomial values are omitted from the compact panels until downstream
audit is complete.

Its machine-readable validation summary is:

- `examples/input_gallery/data/appendix_spatial_graph_inputs_summary.json`

Appendix S4, the grouped surface, volume, and Fermi-surface-style figure, is
available at:

- `examples/input_gallery/figures/appendix_surface_volume_fermi_inputs.png`
- `examples/input_gallery/figures/appendix_surface_volume_fermi_inputs.svg`
- `examples/input_gallery/figures/appendix_surface_volume_fermi_inputs.pdf`

This appendix figure shows surface-like examples together with a prototype
skeleton/spatial graph result:

1. Genus-2 Surface Mesh (\texttt{PLY})
2. Torus Surface Mesh (\texttt{PLY})
3. Vector Flow Volume (\texttt{NPZ})
4. Gyroid Volume (\texttt{NPZ})
5. Schwarz-P Volume (\texttt{NPZ})
6. Nodal-Line Fermi (\texttt{VTP})

Each panel displays the source surface, scalar-volume, vector-flow, or Fermi
geometry above the graph-compatible skeleton visualization.  Surface and scalar
volume panels overlay the skeleton/spatial graph on a translucent copy of the
source geometry; the vector-flow panel displays an oriented spatial graph with
direction arrows.  These skeletons are prototype workflow visualizations;
robust automatic extraction and Yamada-polynomial evaluation remain pending
downstream audit.

Its machine-readable validation summary is:

- `examples/input_gallery/data/appendix_surface_volume_fermi_inputs_summary.json`

The current publication-style gallery is available at:

- `examples/input_gallery/figures/task2_input_gallery_publication_style.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.pdf`

A comparison version with raw-input thumbnails is available at:

- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_raw_inputs.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_raw_inputs.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_raw_inputs.pdf`

The preferred comparison version with separate source-domain panels is available
at:

- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_source_insets.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_source_insets.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_source_insets.pdf`

The gallery contains 9 validated panels:

1. Protein Backbone (\texttt{PDB})
2. tRNA (\texttt{mmCIF})
3. Ring Polymer (\texttt{GRO})
4. Polymer (\texttt{LAMMPS})
5. Cinquefoil Knot (\texttt{XYZ})
6. Engineering Network (\texttt{CSV})
7. Spatial Graph (\texttt{JSON})
8. Neuron Morphology (\texttt{SWC})
9. Spatial Network (\texttt{GraphML})

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
- Surface, volume, vector-flow, and Fermi examples now have Appendix S4
  prototype skeleton/spatial-graph visualizations, but robust extraction and
  downstream Yamada evaluation are still pending audit.
- Appendix S4 keeps surface, volume, vector-flow, and Fermi examples in scope
  for Task 2 by pairing each input geometry with a prototype skeleton/spatial
  graph or oriented graph result.
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

Scalar fields, vector-flow volumes, and Fermi surfaces can remain examples-level
adapters until their schemas, skeleton-extraction assumptions, and expected user
workflows are clearer.

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
