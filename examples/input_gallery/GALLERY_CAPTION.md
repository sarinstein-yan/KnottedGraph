# Task 2 Input Gallery Caption

## Short Caption

Task 2 input-format gallery. Representative data from different application
domains are converted into common KnottedGraph-compatible geometric objects.
Blue geometry shows the imported curve, graph, surface, or volume-derived
surface. Red points mark graph nodes or original open-curve endpoints. Biological
open traces are shown with direct endpoint closure for knot-style visualization.
Gray silhouettes show orthogonal projections to emphasize the embedded 3D
geometry.

## Full Caption

Task 2 input-format gallery demonstrating user-friendly adapters for distinct
scientific and engineering domains. Each panel shows a representative external
input converted into an internal object already understood by the KnottedGraph
workflow: ordered coordinate curves, embedded `networkx.MultiGraph(pos/pts)`
objects, or PyVista `PolyData` surfaces. (a) A knotted protein backbone loaded
from a PDB file and represented as a directly closed C-alpha coordinate trace.
(b) A DNA double helix loaded from a PDB file and represented by directly closed
phosphate-backbone traces. (c) A tRNA structure loaded from mmCIF and represented
as a directly closed RNA phosphate trace.
(d) A trefoil polymer ring loaded from a LAMMPS-style snapshot. (e) A closed
cinquefoil coordinate chain loaded from an XYZ file. (f) A K4 embedded spatial
graph loaded from node/edge CSV files, representing the type of component-
interconnect input useful for engineering networks such as circuits, pipe
systems, and cooling networks. (g) A trefoil tube surface loaded from a PLY mesh
file. (h) A gyroid-like volumetric scalar field loaded from NPZ and converted to
an isosurface. (i) A Rashba-inspired split Fermi-surface mesh loaded from VTP.
The gallery validates the input layer only: these examples are converted and
plotted, while PD-code and Yamada-polynomial calculations remain downstream
steps for suitable spatial graphs.

## Panel Key

| Panel | Title | Input format | Domain | Internal object |
| --- | --- | --- | --- | --- |
| (a) | Knotted Protein PDB | `.pdb` | protein | coordinate curve / `MultiGraph(pos/pts)` |
| (b) | DNA Double Helix PDB | `.pdb` | DNA | coordinate curves / `MultiGraph(pos/pts)` |
| (c) | tRNA mmCIF | `.cif` | RNA | coordinate curve / `MultiGraph(pos/pts)` |
| (d) | Trefoil Polymer LAMMPS | LAMMPS dump | polymer | closed coordinate curve |
| (e) | Cinquefoil XYZ | `.xyz` | coordinate chain | closed coordinate curve |
| (f) | K4 Spatial Graph CSV | node/edge CSV | spatial network | `MultiGraph(pos/pts)` |
| (g) | Trefoil Tube PLY | `.ply` | surface mesh | PyVista `PolyData` |
| (h) | Gyroid Volume NPZ | `.npz` | volumetric field | extracted isosurface |
| (i) | Rashba Fermi VTP | `.vtp` | Fermi surface | PyVista `PolyData` |

## Figure Files

- `examples/input_gallery/figures/task2_input_gallery_publication_style.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.pdf`
