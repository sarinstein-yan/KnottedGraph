# Main Text Input-Figure Selection

This note records Step 1 and Step 2 for reorganizing the Task 2 figures after
advisor feedback.

The main-text figure should show each input type once, while also making sure
each main panel displays a 3D node/edge object rather than visualization-only
surface geometry.  It prioritizes input-format diversity among graph-producing
inputs: the reader should immediately see that PDB, mmCIF, GRO, LAMMPS, XYZ,
CSV, JSON, SWC, and GraphML-style inputs can all enter the same graph-compatible
workflow.  The appendix should emphasize broader domain diversity by showing
multiple examples within each input family, including surface/volume/physics
inputs where their resulting spatial graphs can be displayed.

Advisor follow-up: the overall Task 2 gallery should show as many possible
input types as practical for users.  Surface meshes, scalar volumes,
vector-flow volumes, and Fermi-surface-style geometry are valid input types and
should stay in appendix or workflow figures, even though they are not in the
graph-only main 3x3 figure.

## Selection Rules

- Use one aesthetically strong and domain-representative example per input
  format.
- Avoid repeated file formats in the main text.  For example, do not show both a
  protein PDB and a DNA PDB in the same main figure.
- Keep the main figure visually simple: source/original data on the side,
  converted KnottedGraph-compatible object as the main panel.
- For spatial graph CSV inputs, render edges using embedded curved paths
  (`points_json` / edge point lists), not only straight source-target segments.
  Curved paths are important for communicating spatial topology.
- Move breadth, repeated examples, and failure/edge-case demonstrations to the
  appendix.
- Surface, volume, vector-flow, and Fermi examples are moved out of the main
  3x3 figure for now because their current gallery panels show visualization
  geometry rather than explicit 3D node/edge graphs.  They are still valid Task
  2 input types and should appear in appendix/workflow figures where the
  resulting skeleton/spatial graph is shown.
- Protein, DNA, and RNA appendix examples should use chain or residue-segment
  color coding where possible.
- Yamada polynomials should be displayed in appendix panels only after the
  downstream graph-to-PD/Yamada pipeline is confirmed for that example.

## Current Candidate Pool

These candidates already render successfully in the existing Task 2 gallery
summaries.

### Main Gallery Candidates

| Candidate | Input format | Domain | Current status | Main-text role |
| --- | --- | --- | --- | --- |
| Protein Backbone PDB (1J85) | `.pdb` | protein | success, no issues | selected PDB representative; use conservative title unless the knotted-protein claim is documented |
| DNA Double Helix PDB | `.pdb` | DNA | success, no issues | move to biology appendix because PDB would be repeated |
| tRNA mmCIF | `.cif` | RNA | success, no issues | selected mmCIF representative |
| Trefoil Polymer LAMMPS | LAMMPS dump | polymer | success, no issues | backup LAMMPS representative |
| Cinquefoil XYZ | `.xyz` | coordinate chain | success, no issues | selected XYZ representative |
| Engineering Network CSV | node/edge CSV | spatial graph | success, no issues | selected CSV representative |
| Spatial Graph JSON | `.json` | abstract spatial graph | success, no issues | selected JSON node-graph representative |
| Neuron Morphology SWC | `.swc` | neuroscience morphology | success, no issues | selected SWC node-tree representative |
| Spatial Network GraphML | `.graphml` | graph/network exchange | success, no issues | selected GraphML node-network representative |

### Biology Appendix Candidates

| Candidate | Input format | Domain | Current status | Appendix use |
| --- | --- | --- | --- | --- |
| Crambin PDB | `.pdb` | protein | success, no issues | compact protein example |
| Ubiquitin PDB | `.pdb` | protein | success, no issues | standard protein example |
| Protein Backbone PDB (1J85) | `.pdb` | protein | success, no issues | topology-relevant protein candidate; document source before calling it knotted |
| Hemoglobin PDB | `.pdb` | protein complex | success, no issues | multi-chain color-coding example |
| B-DNA Duplex PDB | `.pdb` | DNA | success, no issues | strand-colored DNA example |
| tRNA mmCIF | `.cif` | RNA | success, no issues | classic RNA example |
| Ubiquitin mmCIF | `.cif` | protein | success, no issues | mmCIF protein backup |
| Alpha Helix XYZ | `.xyz` | peptide coordinate chain | success, no issues | small peptide/coordinate-chain bridge |
| Viral Capsid PLY | `.ply` | biomolecular surface | success, no issues | biology surface appendix candidate |

### Spatial-Graph and Polymer Appendix Candidates

| Candidate | Input format | Domain | Current status | Appendix use |
| --- | --- | --- | --- | --- |
| Pipe Manifold CSV | node/edge CSV | spatial network | success, no issues | engineering network |
| Circuit Harness CSV | node/edge CSV | spatial network | success, no issues | electric/circuit network |
| Cooling Network CSV | node/edge CSV | spatial network | success, no issues | thermal/cooling network |
| Vascular Branch CSV | node/edge CSV | spatial network | success, no issues | branching biological/transport network |
| Lattice Truss CSV | node/edge CSV | spatial network | success, no issues | mechanical truss network |
| Ring Polymer GRO | `.gro` | polymer | success, no issues | selected GRO representative |
| Polymer LAMMPS | LAMMPS dump | polymer | success, no issues | selected LAMMPS representative |
| Bottlebrush Polymer XYZ | `.xyz` | polymer coordinate chain | success, no issues | polymer appendix candidate |
| Coiled Cable DAT | `.dat` / `.xyz` | cable harness | success, no issues | coordinate-chain/cable appendix candidate |
| Sensor Trace CSV | `.csv` | coordinate chain | success, no issues | coordinate-chain format coverage |
| Lissajous Loop JSON | `.json` | coordinate chain | success, no issues | coordinate-chain format coverage |
| Ribbon Loop NPY | `.npy` | coordinate chain | success, no issues | coordinate-chain format coverage |
| Meander Path TSV | `.tsv` | coordinate chain | success, no issues | coordinate-chain format coverage |
| Plain Text Cable TXT | `.txt` | coordinate chain | success, no issues | coordinate-chain format coverage |

### Surface, Volume, and Physics Appendix Candidates

| Candidate | Input format | Domain | Current status | Appendix use |
| --- | --- | --- | --- | --- |
| Genus-2 Surface PLY | `.ply` | surface mesh | success, no issues | selected S4 surface example |
| Torus Surface PLY | `.ply` | surface mesh | success, no issues | surface appendix |
| Figure-Eight XYZ | `.xyz` | coordinate knot | success, no issues | XYZ backup / appendix knot |
| Hopf Link CSV | node/edge CSV | spatial graph | success, no issues | link/spatial graph appendix |
| Three-Ring Link CSV | node/edge CSV | spatial graph | success, no issues | link/spatial graph appendix |
| Gyroid Volume NPZ | `.npz` | volumetric field | success, no issues | appendix/workflow candidate; should be paired with extracted spatial graph or skeleton |
| Schwarz-P Volume NPZ | `.npz` | volumetric field | success, no issues | volume appendix |
| Vector Flow Volume NPZ | `.npz` | vector-flow volume | success, no issues | flow appendix; displays oriented spatial graph |
| Nodal-Line Fermi VTP | `.vtp` | Fermi surface | success, no issues | appendix/workflow candidate; should be paired with extracted graph/skeleton if used |

## Main Text Figure: Recommended 9-Panel Selection

| Panel | Input type | Selected example | Why this one |
| --- | --- | --- | --- |
| (a) | PDB | Protein Backbone (\texttt{PDB}) | 1J85 is the current topology-relevant and visually complex protein-backbone candidate.  Use the conservative title unless the knotted-protein provenance is documented from a reliable database or paper. |
| (b) | mmCIF | tRNA (\texttt{mmCIF}) | Classic RNA structure and a clear non-PDB biomolecular file type. |
| (c) | GRO | Ring Polymer (\texttt{GRO}) | Represents GROMACS-style polymer/molecular snapshots and complements the LAMMPS example. |
| (d) | LAMMPS dump | Polymer (\texttt{LAMMPS}) | Simulation-derived polymer conformation, now rendered with a shape distinct from the cinquefoil coordinate-knot panel. |
| (e) | XYZ / coordinate chain | Cinquefoil Knot (\texttt{XYZ}) | Clean, recognizable closed coordinate knot; good generic coordinate-chain representative. |
| (f) | Spatial Graph CSV | Engineering Network (\texttt{CSV}) | Best main-text representative for circuits, pipe systems, cooling systems, and embedded component networks.  Render using curved edge paths from `points_json` / edge point lists to show spatial topology. |
| (g) | Spatial Graph JSON | Spatial Graph (\texttt{JSON}) | Friendly JSON schema for an abstract embedded graph; shows explicit red nodes and routed 3D edges. |
| (h) | Neuron Morphology SWC | Neuron Morphology (\texttt{SWC}) | Classic neuroscience morphology format; naturally represents a 3D node tree with parent-child edges. |
| (i) | Spatial Network GraphML | Spatial Network (\texttt{GraphML}) | Common graph/network exchange format extended with 3D node coordinates and embedded edge paths. |

## Hamiltonian Input Note

This gallery is currently organized around user-facing file/input formats.  A
Hamiltonian-derived examples are not selected for the main 3x3 gallery after
the graph-only revision, because the current Fermi/VTP panel is a surface mesh
rather than an explicit node graph.  If the paper's main workflow figure already
shows Hamiltonian-to-geometry conversion, this gallery can leave Hamiltonian
input out to avoid duplication.  If Hakan wants an explicit Hamiltonian input in
the Task 2 gallery, it should appear in a workflow/appendix physics panel that
shows:

```text
Hamiltonian parameters -> computed Fermi/nodal geometry -> KnottedGraph object
```

## Main-Text Examples to Move Out

| Existing example | Reason to move to appendix |
| --- | --- |
| DNA Double Helix PDB | PDB would otherwise appear twice in the main figure.  Keep it in the biology appendix, where broader domain diversity matters more than input-format uniqueness. |
| Trefoil Polymer LAMMPS | Good example, but the selected Polymer LAMMPS conformation is more visually expressive for the single LAMMPS slot. |
| Ubiquitin PDB / Crambin PDB / Hemoglobin PDB | Useful biology breadth examples; not needed in a one-PDB main figure. |
| Ubiquitin mmCIF | mmCIF duplicate; tRNA is more visually recognizable. |
| Pipe / circuit / cooling / vascular / truss CSV variants | Good appendix breadth examples; main figure should keep one CSV representative. |
| Surface / gyroid / vector-flow / Fermi-surface geometry examples | Useful appendix/workflow examples, but not main-text panels until their extracted or prototype spatial graph is shown. |
| Torus / Schwarz-P / Hopf link / three-ring link | Useful appendix examples; graph-like examples such as Hopf link and three-ring link can remain in graph appendices. |

## Appendix Grouping Plan

| Appendix figure | Group | Examples |
| --- | --- | --- |
| Figure S1 | Biological structure inputs | Crambin PDB, Ubiquitin PDB, Protein Backbone PDB (1J85), Hemoglobin PDB, B-DNA Duplex PDB, tRNA mmCIF, Ubiquitin mmCIF |
| Figure S2 | Polymer and coordinate-chain inputs | Ring Polymer GRO, Polymer LAMMPS, Trefoil Polymer LAMMPS, Bottlebrush Polymer XYZ, Coiled Cable DAT, Cinquefoil XYZ, Figure-Eight XYZ, Sensor Trace CSV, Lissajous Loop JSON, Ribbon Loop NPY, Meander Path TSV, Plain Text Cable TXT |
| Figure S3 | Spatial graph / engineering CSV inputs | Engineering Network CSV, Pipe Manifold CSV, Circuit Harness CSV, Cooling Network CSV, Vascular Branch CSV, Lattice Truss CSV, Hopf Link CSV, Three-Ring Link CSV |
| Figure S4 | Surface, volume, flow, and Fermi inputs | Genus-2 Surface PLY, Torus Surface PLY, Vector Flow Volume NPZ, Gyroid Volume NPZ, Schwarz-P Volume NPZ, Nodal-Line Fermi VTP |

## Required Follow-Up Before Redrawing

1. Add or reuse a plotting layout for the new main-text figure using the selected
   nine examples above.
2. For biology panels, add color coding by chain or residue segment.
3. Appendix S4 has been regenerated with surface, volume, vector-flow, and
   Fermi input examples organized in appendix.
4. Surface/volume/flow/Fermi appendix panels now show both the original input
   and a prototype skeleton/spatial graph or oriented graph result.  Robust
   extraction and downstream Yamada evaluation still need a separate audit.
5. For spatial graph CSV panels, ensure the selected example uses curved edge
   paths (`points_json`) so over/under spatial routing is visible.
6. Before adding Yamada polynomials to appendix panels, audit which selected
   examples can reliably pass through the downstream PD/Yamada pipeline.
7. Keep generated figures, raw datasets, PBS scripts, and logs out of commits
   unless explicitly approved.
