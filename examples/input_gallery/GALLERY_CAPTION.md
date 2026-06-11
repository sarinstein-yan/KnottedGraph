# Task 2 Input Gallery Caption

## Main Text Figure Caption

Main-text Task 2 input-format gallery.  The 3x3 figure prioritizes
input-format diversity: each panel shows one representative user-facing input
format converted into the geometric object used by the KnottedGraph input
workflow.  The main converted geometry is shown on the left, with a compact
source-domain view on the right.  Protein and RNA backbone traces are
color-coded by residue/index segments for biological readability.  Spatial graph
CSV edges are rendered with curved embedded edge paths (`points_json`) rather
than only straight source-target segments.  After the graph-only revision, all
main panels show curve or spatial-graph outputs rather than visualization-only
surfaces.  Surface mesh, scalar-volume, vector-flow, and Fermi-surface geometry
examples are reserved for appendix/workflow figures where the resulting
skeleton/spatial graph can also be displayed.  These surface/volume/flow/Fermi
examples remain valid Task 2 input types even though they are not selected for
the graph-only main 3x3 figure.  Yamada-polynomial values are omitted from the
compact figures until the downstream pipeline audit is complete.

Main-text panels: (a) Protein Backbone (\texttt{PDB}), using 1J85 as a
conservative protein-backbone example.  (b) tRNA (\texttt{mmCIF}).  (c) Ring
Polymer (\texttt{GRO}).  (d) Polymer (\texttt{LAMMPS}).  (e)
Cinquefoil Knot (\texttt{XYZ}).  (f) Engineering Network (\texttt{CSV}).
(g) Spatial Graph (\texttt{JSON}).  (h) Neuron Morphology (\texttt{SWC}).
(i) Spatial Network (\texttt{GraphML}).

Main-text figure files:

- `examples/input_gallery/figures/main_text_input_gallery.png`
- `examples/input_gallery/figures/main_text_input_gallery.svg`
- `examples/input_gallery/figures/main_text_input_gallery.pdf`

## Appendix S1 Biology Caption

Appendix S1 biological input examples.  This grouped appendix figure emphasizes
domain diversity within biological structure inputs rather than input-format
uniqueness.  Each panel shows the source-domain molecular structure above the
converted backbone graph used by the input workflow.  Protein and RNA traces are
color-coded by residue/index segment; multi-chain structures are color-coded by
chain.  Red nodes mark open-trace endpoints after conversion.  Yamada-polynomial
values are not printed inside each compact panel; their status remains pending
downstream audit because these biological examples have not yet been checked
through the graph-to-PD/Yamada pipeline.

Appendix S1 panels: (a) Crambin PDB.  (b) Ubiquitin PDB.  (c) Protein Backbone
PDB.  (d) Hemoglobin PDB, shown as a multi-chain color-coded protein complex.
(e) B-DNA Duplex PDB, shown with chain/strand coloring.  (f) tRNA mmCIF.  (g)
Ubiquitin mmCIF.

Appendix S1 figure files:

- `examples/input_gallery/figures/appendix_biology_inputs.png`
- `examples/input_gallery/figures/appendix_biology_inputs.svg`
- `examples/input_gallery/figures/appendix_biology_inputs.pdf`

## Appendix S2 Polymer And Coordinate-Chain Caption

Appendix S2 polymer and coordinate-chain input examples.  This grouped
appendix figure emphasizes polymer-simulation and lightweight coordinate-file
inputs.  Each panel shows a source-domain bead/chain view above the converted
graph-compatible curve used by the input workflow.  Closed polymer and knot
examples are rendered as closed loops; open coordinate chains mark their
endpoints as red graph nodes.  Yamada-polynomial status is tracked in the
machine-readable summary rather than repeated inside each compact panel, because
these examples have not yet been checked through the graph-to-PD/Yamada
pipeline.

Appendix S2 panels: (a) Ring Polymer (\texttt{GRO}).  (b) Polymer
(\texttt{LAMMPS}).  (c) Trefoil Polymer (\texttt{LAMMPS}).  (d) Bottlebrush
Polymer (\texttt{XYZ}).  (e) Coiled Cable (\texttt{DAT}).  (f) Cinquefoil Knot
(\texttt{XYZ}).  (g) Figure-Eight Knot (\texttt{XYZ}).  (h) Sensor Trace
(\texttt{CSV}).  (i) Lissajous Loop (\texttt{JSON}).  (j) Ribbon Loop
(\texttt{NPY}).  (k) Meander Path (\texttt{TSV}).  (l) Plain Text Cable
(\texttt{TXT}).

Appendix S2 figure files:

- `examples/input_gallery/figures/appendix_polymer_coordinate_inputs.png`
- `examples/input_gallery/figures/appendix_polymer_coordinate_inputs.svg`
- `examples/input_gallery/figures/appendix_polymer_coordinate_inputs.pdf`

## Appendix S3 Spatial Graph Caption

Appendix S3 spatial graph CSV examples.  This grouped appendix figure shows
multiple node/edge CSV inputs for engineering, transport, mechanical, and
abstract spatial-network settings.  Each panel displays the source CSV graph
schematic above the converted embedded `MultiGraph(pos/pts)` representation.
All examples use curved edge paths through `points_json`, rather than only
straight source-target edges, so the embedded spatial routing is visible.  Red
nodes mark graph vertices and blue tubes show embedded edges.  Yamada-polynomial
values are omitted from the compact panels until downstream audit is complete.

Appendix S3 panels: (a) Engineering Network CSV.  (b) Pipe Manifold CSV.  (c)
Circuit Harness CSV.  (d) Cooling Network CSV.  (e) Vascular Branch CSV.  (f)
Lattice Truss CSV.  (g) Hopf Link CSV.  (h) Three-Ring Link CSV.

Appendix S3 figure files:

- `examples/input_gallery/figures/appendix_spatial_graph_inputs.png`
- `examples/input_gallery/figures/appendix_spatial_graph_inputs.svg`
- `examples/input_gallery/figures/appendix_spatial_graph_inputs.pdf`

## Appendix S4 Surface, Volume, Flow, And Fermi Caption

Appendix S4 surface-like input examples.  This grouped appendix figure responds
to the point that a surface alone is not enough to demonstrate the framework:
each panel shows the source surface, scalar volume, vector-flow volume, or
Fermi-surface geometry above a prototype skeleton/spatial graph result.  The
surface and scalar-volume panels overlay the graph-compatible skeleton on a
translucent copy of the source geometry; the vector-flow panel displays an
oriented spatial graph with arrows.  These skeletons are used for input-format
and workflow visualization, not claimed as a completed automatic
surface-to-skeleton extraction algorithm; Yamada polynomial values are omitted
from the compact panels until downstream audit is complete.

Appendix S4 panels: (a) Genus-2 Surface Mesh (\texttt{PLY}).  (b) Torus
Surface Mesh (\texttt{PLY}).  (c) Vector Flow Volume (\texttt{NPZ}).  (d)
Gyroid Volume (\texttt{NPZ}).  (e) Schwarz-P Volume (\texttt{NPZ}).  (f)
Nodal-Line Fermi (\texttt{VTP}).

Appendix S4 figure files:

- `examples/input_gallery/figures/appendix_surface_volume_fermi_inputs.png`
- `examples/input_gallery/figures/appendix_surface_volume_fermi_inputs.svg`
- `examples/input_gallery/figures/appendix_surface_volume_fermi_inputs.pdf`

## Short Caption

Task 2 input-format gallery. Representative data from different application
domains are converted into common KnottedGraph-compatible geometric objects.
Blue geometry shows the imported curve, graph, surface, or volume-derived
surface. Red points mark graph nodes or original open-curve endpoints. Biological
open traces are shown with direct endpoint closure for knot-style visualization.
In the source-inset comparison figure, the small right-hand panel
shows a compact source-domain view before conversion: molecular all-atom views
with backbone traces, bead-and-bond polymer snapshots, component-network
schematics, or network-file schematics.

## Full Caption

Task 2 main-text input-format gallery demonstrating user-friendly adapters for
distinct scientific and engineering domains. Each panel shows a representative
external input converted into a graph-compatible internal object already
understood by the KnottedGraph input workflow: ordered coordinate curves or
embedded `networkx.MultiGraph(pos/pts)` objects. (a) A protein backbone loaded
from a PDB file and represented as a directly closed C-alpha coordinate trace.
(b) A tRNA structure loaded from mmCIF and represented as a directly closed RNA
phosphate trace. (c) A ring polymer loaded from a GROMACS GRO file. (d) A
polymer conformation loaded from a LAMMPS-style snapshot. (e) A closed cinquefoil
coordinate chain loaded from an XYZ file. (f) A 3D engineering
component-interconnect network loaded from node/edge CSV files. (g) An abstract
spatial graph loaded from JSON. (h) A neuron morphology loaded from an SWC file
and converted to a 3D node tree. (i) A spatial network loaded from GraphML using
3D node coordinates and embedded edge paths. The gallery validates the input
layer only: these examples are converted and plotted, while PD-code and
Yamada-polynomial calculations remain downstream steps for suitable spatial
graphs.

## Panel Key

| Panel | Title | Input format | Domain | Internal object |
| --- | --- | --- | --- | --- |
| (a) | Protein Backbone (\texttt{PDB}) | `.pdb` | protein | coordinate curve / `MultiGraph(pos/pts)` |
| (b) | tRNA (\texttt{mmCIF}) | `.cif` | RNA | coordinate curve / `MultiGraph(pos/pts)` |
| (c) | Ring Polymer (\texttt{GRO}) | `.gro` | polymer | closed coordinate curve |
| (d) | Polymer (\texttt{LAMMPS}) | LAMMPS dump | polymer | closed coordinate curve |
| (e) | Cinquefoil Knot (\texttt{XYZ}) | `.xyz` | coordinate chain | closed coordinate curve |
| (f) | Engineering Network (\texttt{CSV}) | node/edge CSV | engineering network | `MultiGraph(pos/pts)` |
| (g) | Spatial Graph (\texttt{JSON}) | `.json` | abstract spatial graph | `MultiGraph(pos/pts)` |
| (h) | Neuron Morphology (\texttt{SWC}) | `.swc` | neuroscience morphology | `MultiGraph(pos/pts)` |
| (i) | Spatial Network (\texttt{GraphML}) | `.graphml` | graph/network exchange | `MultiGraph(pos/pts)` |

## Figure Files

- `examples/input_gallery/figures/task2_input_gallery_publication_style.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style.pdf`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_raw_inputs.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_raw_inputs.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_raw_inputs.pdf`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_source_insets.png`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_source_insets.svg`
- `examples/input_gallery/figures/task2_input_gallery_publication_style_with_source_insets.pdf`
