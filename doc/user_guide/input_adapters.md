# Input Adapters

<div class="kg-hero">
  <p class="kg-lead"><strong>Input adapters turn external geometry into one of two explicit in-memory objects:</strong> an embedded <code>networkx.MultiGraph</code> for curve and graph data, or a <code>pyvista.PolyData</code> surface mesh. Start with the table below, inspect the returned validation issues, and only then continue to projection, skeletonization, or an application workflow.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/Latest_Workplace/User_guide/04_input_adapters.ipynb">Open The Runnable Input Tutorial</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/Latest_Workplace/User_guide/02_core_workflows.ipynb">Open Core Workflows</a>
    <a href="../api/inputs.html">Open Input API Reference</a>
  </div>
</div>

<div class="kg-wide-figure">
  <a href="../site_figures/input_formats_overview.png">
    <img src="../site_figures/input_formats_overview.png" alt="Eight-panel gallery of input families converted to graph-compatible representations" decoding="async">
  </a>
  <p class="kg-caption">Dark teal marks graph edges or curves, red spheres mark true endpoints or vertices, translucent geometry provides physical context, and purple arrows show orientation. The public adapters listed below form the supported ingestion boundary; other panels represent application-level or in-memory workflows.</p>
</div>

## Choose a public adapter

Import adapters from `knotted_graph.inputs`, not from the package root.

```{list-table}
:header-rows: 1
:class: kg-route-table
:widths: 20 24 27 29

* - Starting data
  - Public entry point
  - Accepted input
  - Returned object
* - Ordered coordinate chain
  - `from_coordinate_chain`
  - An `(N, 3)` array, or CSV, DAT, JSON, NPY, TSV, TXT, or XYZ
  - `CoordinateInputResult`; use `.graph`
* - Protein or nucleic-acid PDB trace
  - `from_protein_ca_backbone`, `from_nucleic_acid_backbone`, or `from_pdb_backbone`
  - Local PDB path or four-character RCSB PDB identifier
  - `PDBBackboneInputResult`; use `.graph`
* - mmCIF atom trace
  - `from_mmcif_backbone`
  - Local CIF/mmCIF path or four-character RCSB PDB identifier
  - `MMCIFBackboneInputResult`; use `.graph`
* - GROMACS polymer snapshot
  - `from_gromacs_gro`
  - GRO file, optionally filtered by atom and residue name
  - `PolymerInputResult`; use `.graph`
* - LAMMPS polymer snapshot
  - `from_lammps_dump`
  - First frame of a LAMMPS dump containing unscaled `x`, `y`, and `z`
  - `PolymerInputResult`; use `.graph`
* - Embedded spatial graph
  - `from_spatial_graph_csv`
  - A node CSV and a separate edge CSV
  - `SpatialGraphInputResult`; use `.graph`
* - Surface mesh
  - `from_surface_mesh`
  - OBJ, OFF, PLY, STL, VTK, or VTP surface geometry
  - `SurfaceInputResult`; use `.mesh`
```

The coordinate JSON adapter accepts a top-level array or an object containing
`points` or `coords`. It is not a generic graph-JSON reader. Likewise, the NPY
entry means one `(N, 3)` coordinate array; NPZ field and volume archives are not
handled by this adapter.

## One graph contract

All graph-producing adapters return an undirected `networkx.MultiGraph` with a
common embedded-geometry convention:

- every node has a finite three-vector in its `pos` attribute;
- every edge has an `(M, 3)` polyline in its `pts` attribute;
- the first and last polyline points match the incident node positions, in
  either orientation; and
- parallel edges remain distinct through their multigraph keys.

Ordered chains, PDB/mmCIF atom traces, and polymer snapshots become **one
geometric edge**. Their coordinate samples live in `edge["pts"]`; the samples do
not become one graph vertex each. An open chain has `start` and `end` nodes. A
geometrically closed chain is represented by one `loop_anchor` node and a
self-loop. The paired-CSV adapter, by contrast, preserves the supplied graph
topology.

Every adapter also returns an `issues` list. An empty list means that the result
passed that adapter's structural checks; it does not prove that the geometry is
suitable for a particular invariant or scientific interpretation.

## Coordinate chains and explicit closure

```python
import numpy as np

from knotted_graph.inputs import from_coordinate_chain

coords = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.0, 1.0, 0.0]]
)
result = from_coordinate_chain(coords, input_id="open_trace")
graph = result.graph

if result.issues:
    raise ValueError(result.issues)
```

File input uses the same function:

```python
result = from_coordinate_chain("trace.csv", columns=("x", "y", "z"))
```

Closure is never inferred merely because a user intends the curve to be closed.
Choose one of the following behaviours deliberately:

```python
# Add the final segment back to the first point.
closed_result = from_coordinate_chain(coords, closure="direct")

# Accept coordinates that already repeat the first point at the end.
closed_coords = np.vstack([coords, coords[0]])
already_closed = from_coordinate_chain(closed_coords, closed=True)

# Record the intended closure without changing the open geometry.
record_only = from_coordinate_chain(
    coords,
    closed=True,
    closure="metadata_only",
)
```

With `closure="direct"`, `result.coords` remains the loaded coordinate array,
whereas the graph edge's `pts` array includes the added closing point.

## Biomolecular traces

The biomolecular adapters extract matching atom records and preserve file
order. They produce a backbone trace, not a molecular bond graph.

```python
from pathlib import Path

from knotted_graph.inputs import (
    from_mmcif_backbone,
    from_nucleic_acid_backbone,
    from_protein_ca_backbone,
)

protein = from_protein_ca_backbone(
    Path("protein.pdb"),
    chain_id="A",
)
dna = from_nucleic_acid_backbone(
    Path("dna.pdb"),
    chain_id="A",
)
phosphate_trace = from_mmcif_backbone(
    Path("rna.cif"),
    chain_id="A",
    atom_name="P",
)
```

A four-character string with no file suffix is treated as an RCSB identifier.
The default `download=True` then caches the downloaded file in `data_dir`, or in
the current directory when `data_dir` is omitted:

```python
protein = from_protein_ca_backbone(
    "1CRN",
    chain_id="A",
    data_dir="data/rcsb",
)
```

For reproducible or offline work, pass a local `Path`. Also keep these selection
rules in mind:

- PDB input with several matching chains requires an explicit `chain_id`.
- mmCIF input without `chain_id` selects the chain with the most matching atoms;
  pass `chain_id` to avoid an unintended choice.
- the current mmCIF reader targets RCSB-style `_atom_site` loops with one
  complete atom row on each physical line; reformat other valid CIF layouts
  before loading them;
- only `ATOM` records, the requested model and atom name, and blank or `A`
  alternate locations are used;
- the convenience functions default to protein C-alpha (`CA`) and nucleic-acid
  phosphate (`P`) traces, respectively; and
- at least two matching atoms are required.

## Polymer snapshots and units

```python
from knotted_graph.inputs import from_gromacs_gro, from_lammps_dump

gro_result = from_gromacs_gro(
    "ring.gro",
    atom_name="BB",
    residue_name="RNG",
    output_unit_scale=10.0,
    closure="direct",
)

lammps_result = from_lammps_dump(
    "chain.dump",
    molecule_id=7,
    sort_column="id",
)
```

The GRO reader multiplies loaded coordinates by `output_unit_scale`; its default
of `10.0` converts the usual nanometre coordinates to ångströms. Set the scale
explicitly when the downstream workflow uses another unit convention. The
reader ignores the simulation box and does not unwrap periodic trajectories.

The LAMMPS reader loads only the first frame, requires unscaled `x`, `y`, and `z`
columns, and orders the selected rows by `sort_column`. It does not reconstruct
bonds, unwrap periodic coordinates, or read scaled `xs`, `ys`, and `zs` columns.
If the dump has a `mol` column, `molecule_id` filters it; without a `mol` column,
no molecule filter can be applied.

## Spatial-graph CSV

The graph adapter uses two files. A minimal node table is:

```text
node_id,x,y,z,label
u,0.0,0.0,0.0,inlet
v,1.0,0.0,0.0,outlet
```

and a minimal edge table is:

```text
edge_id,source,target,points_json
upper,u,v,"[[0,0,0],[0.5,0.3,0],[1,0,0]]"
lower,u,v,"[[0,0,0],[0.5,-0.3,0],[1,0,0]]"
```

```python
from knotted_graph.inputs import from_spatial_graph_csv

result = from_spatial_graph_csv(
    "nodes.csv",
    "edges.csv",
    graph_id="two_routes",
)
graph = result.graph
```

Node IDs may use `node_id` or the legacy name `id`. Edge IDs may use `edge_id`,
`id`, or `key`; when omitted they are generated. Additional columns are
preserved as string attributes. `points_json` is optional: without it, the edge
is a straight segment. When present, it must contain at least two 3D points and
its endpoints must match the source and target node positions.

## Surface meshes

Surface support requires the optional PyVista dependency:

```bash
pip install "knotted_graph[surface]"
```

```python
from knotted_graph.inputs import from_surface_mesh

result = from_surface_mesh("surface.ply")
mesh = result.mesh

for issue in result.issues:
    print(f"surface warning: {issue}")
```

By default the adapter cleans and triangulates the loaded geometry. Set
`clean=False` or `triangulate=False` only when the downstream method expects the
original representation. VTK and VTP input is converted to surface geometry;
this adapter is not a general volume reader. Open boundary edges are reported
in `issues` rather than rejected, and loading a surface does not skeletonize it
or create a graph.

## What is not a public file adapter?

Some input families in the overview figure enter through application workflows
or through user-prepared in-memory objects. They should not be described as
`knotted_graph.inputs` file support.

```{list-table}
:header-rows: 1
:class: kg-route-table
:widths: 27 35 38

* - Data family
  - Current route
  - Important boundary
* - Hamiltonian-derived surfaces and graphs
  - Use the physics/application APIs with an in-memory Hamiltonian.
  - There is no Hamiltonian file adapter in `knotted_graph.inputs`.
* - Surface or volume skeletonization
  - Load or construct the required array/mesh, then use the extraction workflow.
  - `from_surface_mesh` only loads surface geometry; it does not skeletonize.
* - Oriented vector fields and integrated flows
  - Load arrays with NumPy and pass them to the relevant workflow code.
  - There is no NPZ vector-field adapter in `knotted_graph.inputs`.
* - Mathematical graphs in GraphML, generic JSON, or edge-list form
  - Load with NetworkX or another parser, add `pos`/`pts`, and validate the embedding.
  - Only the paired spatial-graph CSV schema has a dedicated graph adapter.
* - SWC neural or vascular traces
  - Parse externally and convert to the embedded multigraph contract.
  - There is currently no public SWC adapter.
```

For a manually constructed graph, normalize and validate before continuing:

```python
import networkx as nx
import numpy as np

from knotted_graph.core import ensure_embedding

graph = nx.MultiGraph()
graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
graph.add_node("v", pos=np.array([1.0, 0.0, 0.0]))
graph.add_edge(
    "u",
    "v",
    key="curve",
    pts=np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.0], [1.0, 0.0, 0.0]]),
)
graph = ensure_embedding(graph)
```

## Validation and failure modes

Use the result's `issues` field for non-fatal structural findings. The public
validators are also useful when data is prepared in memory:

```python
from knotted_graph.inputs import (
    validate_coords,
    validate_spatial_graph,
)

coords = validate_coords(coords)
graph_issues = validate_spatial_graph(graph)
```

Surface validation imports the optional PyVista dependency and should stay in
surface-enabled environments:

```python
# Requires: pip install "knotted_graph[surface]"
from knotted_graph.inputs import validate_surface_mesh

mesh_issues = validate_surface_mesh(mesh)
```

Fatal schema failures--including missing required columns, unknown graph
endpoints, duplicate IDs, unsupported suffixes, insufficient usable geometry,
and invalid closure requests--raise exceptions instead of inventing data.
Recoverable row-level PDB or mmCIF coordinate problems can instead be skipped
and reported in `result.issues` when at least two valid selected atoms remain.
Inspect that list before accepting any trace. Catch a specific exception only
when the surrounding workflow can recover from it; otherwise let the error
identify the offending file or schema.

## Continue the workflow

- See [Workflow Overview](workflow_overview.md) for surface-to-graph and
  skeletonization context.
- See [Projection, PD Codes, And Yamada Polynomials](projection_yamada.md) after
  obtaining a suitable embedded graph.
- See the [complete input API reference](../api/inputs.md) for exact signatures
  and result classes.
