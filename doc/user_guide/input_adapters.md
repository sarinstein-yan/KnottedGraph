# Input Handling

Use an input adapter when your starting point is external data rather than an
already constructed embedded graph. The public adapters normalize supported
sources into either:

- a result object containing an embedded `networkx.MultiGraph`; or
- for surface files, a result object containing `PyVista.PolyData`.

This page distinguishes direct public file support from the wider set of
application-level representations shown in research figures.

<div class="kg-link-row">
  <a href="../feature_status.html">Feature-status matrix</a>
  <a href="workflow_overview.html">Continue through the workflow</a>
  <a href="../api/inputs.html">Inputs API</a>
  <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/01_getting_started.ipynb">Open Getting Started</a>
</div>

## Supported public routes

Import the public functions from `knotted_graph.inputs`:

| Starting data | Public call | Result | Important boundary |
| --- | --- | --- | --- |
| Array or CSV/DAT/JSON/NPY/TSV/TXT/XYZ ordered coordinates | `from_coordinate_chain` | `CoordinateInputResult` | JSON/NPY mean coordinates, not arbitrary graphs |
| PDB file or four-character RCSB ID | `from_pdb_backbone` | `PDBBackboneInputResult` | Select chain/model/atom explicitly |
| CIF/mmCIF file or RCSB ID | `from_mmcif_backbone` | `MMCIFBackboneInputResult` | Supports the documented RCSB-style `_atom_site` subset |
| GROMACS GRO snapshot | `from_gromacs_gro` | `PolymerInputResult` | Default coordinate conversion is nm to Å |
| LAMMPS dump | `from_lammps_dump` | `PolymerInputResult` | Reads the first frame and unscaled `x/y/z` columns |
| Paired node/edge CSV files | `from_spatial_graph_csv` | `SpatialGraphInputResult` | Two files; preserves parallel edges and optional polylines |
| OBJ/OFF/PLY/STL/VTK/VTP surface | `from_surface_mesh` | `SurfaceInputResult` | Requires `surface`; returns a mesh, not a graph |

Named knots, torus types, and Artin braid words are handled by
`KnotFunction`; see {doc}`../applications/analytic_knot_fields`. They are
analytic constructors rather than generic file parsers.

## The embedded-graph contract

High-level graph adapters return a dataclass with `.graph` and `.issues`.
The graph is an undirected `networkx.MultiGraph`:

- every node has `pos`, a finite NumPy array of shape `(3,)`;
- every edge has `pts`, a finite array of shape `(N, 3)`;
- the first and last `pts` rows match the positions of the edge endpoints;
- parallel geometric edges remain separate MultiGraph edges; and
- source identifiers, selections, and caller metadata remain available on the
  result or graph metadata.

Always inspect issues before continuing:

```python
result = from_coordinate_chain(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
    input_id="demo-curve",
)

if result.issues:
    for issue in result.issues:
        print("input issue:", issue)

graph = result.graph
print(graph.number_of_nodes(), graph.number_of_edges())
```

Fatal schema or selection errors raise an exception. Recoverable row-level or
post-load validation concerns may be returned in `.issues`; do not assume an
object is publication-ready merely because parsing completed.

## 1. Ordered coordinate chains

```python
import numpy as np
from knotted_graph.inputs import from_coordinate_chain

coordinates = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [1.0, 1.0, 0.0],
    ]
)

result = from_coordinate_chain(coordinates, input_id="open-curve")
graph = result.graph
```

For file input, pass a path. CSV uses named `x`, `y`, and `z` columns by
default; TSV/DAT/TXT read the first three fields; conventional XYZ files may
contain the atom count and a blank or nonblank comment line.

### Closure is explicit

- The default is an open chain.
- `closure="direct"` appends a straight final-to-first segment and produces a
  geometrically closed edge.
- `closed=True` without a closure method is valid only when the supplied first
  and last samples already agree.
- `closure="metadata_only"` records intent but does not close the geometry.

The result's `.coords` retain the original source samples. Inspect edge `pts`
when you need the geometry after direct closure.

## 2. PDB and mmCIF backbones

```python
from knotted_graph.inputs import (
    from_mmcif_backbone,
    from_protein_ca_backbone,
)

pdb_result = from_protein_ca_backbone(
    "protein.pdb",
    chain_id="A",
    model_id=1,
)

cif_result = from_mmcif_backbone(
    "structure.cif",
    chain_id="A",
    atom_name="CA",
    model_id=1,
)
```

A four-character PDB ID may be used instead of a local path when downloading
is enabled. For reproducible offline work, keep a local file and record its
provenance.

When multiple chains match, choose `chain_id` explicitly. The mmCIF reader
currently targets RCSB-style atom-site loops with one complete data row per
physical line; it is not a fully general CIF grammar implementation. PDB and
mmCIF backbone extraction creates an ordered curve from selected atoms—it does
not infer a domain-specific protein interaction or repulsive-layout graph.

## 3. Polymer snapshots

```python
from knotted_graph.inputs import from_gromacs_gro, from_lammps_dump

gro = from_gromacs_gro(
    "chain.gro",
    atom_name="BB",
    output_unit_scale=10.0,
)

lammps = from_lammps_dump(
    "chain.dump",
    molecule_id=7,
)
```

The GRO adapter treats the selected atoms as one ordered coordinate curve and
ignores the box/PBC record. Its default scale converts nanometres to ångströms.
The LAMMPS adapter reads only the first frame, expects unscaled `x/y/z`, and
does not reconstruct bonds, unwrap periodic images, or process `xs/ys/zs`.

## 4. Paired spatial-graph CSV

The nodes file requires an identifier and coordinates:

```text
node_id,x,y,z
0,0,0,0
1,1,0,0
```

The edges file requires source and target identifiers:

```text
edge_id,source,target,points_json
e0,0,1,"[[0,0,0],[0.5,0.2,0],[1,0,0]]"
```

Load both files together:

```python
from knotted_graph.inputs import from_spatial_graph_csv

result = from_spatial_graph_csv("nodes.csv", "edges.csv")
graph = result.graph
```

`points_json` is optional; without it the edge is straight. When present, the
polyline endpoints must match the corresponding node positions. Extra CSV
columns are preserved as string attributes. This route is not a GraphML,
single edge-list, or arbitrary graph-JSON reader.

## 5. Surface meshes

```python
from knotted_graph.inputs import from_surface_mesh

result = from_surface_mesh("surface.ply")
mesh = result.mesh
```

Install the optional dependency first:

```bash
uv sync --extra surface
```

Cleaning and triangulation are enabled by default and may change mesh
connectivity. Open boundaries are reported as issues rather than silently
filled. Converting a physical surface to a scientifically meaningful graph is
application-dependent; this adapter intentionally stops at `PolyData`.

## Formats shown in figures but not exposed as generic adapters

The following may appear after an external or application-specific conversion,
but are not public generic readers in `knotted_graph.inputs`:

- GraphML and generic edge lists;
- SWC neural morphology;
- arbitrary spatial-graph JSON;
- NPZ scalar/vector volumes and oriented flows;
- Hamiltonian files; and
- generic mesh-to-skeleton conversion.

See {doc}`../feature_status` before assuming that a figure subtitle is an
installation promise.

## Continue through the pipeline

After loading a graph:

```python
from knotted_graph.core import ensure_embedding
from knotted_graph.projection import select_projection

graph = ensure_embedding(result.graph)
projection = select_projection(graph)
print("crossings:", projection.num_crossings)
```

Continue with {doc}`workflow_overview` for cleanup and extraction decisions or
{doc}`projection_yamada` for projection, PD-code, and invariant interpretation.
For exact signatures and return fields, use {doc}`../api/inputs`.
