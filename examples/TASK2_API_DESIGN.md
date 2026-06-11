# Task 2 API Design: User-Friendly Input Formats

This document describes a small public input API for Task 2. The goal is to let
users from different fields bring their own data into the existing KnottedGraph
workflow without needing to understand the low-level internal conventions first.

Phase 1 of this design implemented the most general adapters: coordinate
chains, spatial graph CSV files, and surface meshes. Phase 2 migrated the
protein, DNA/RNA, mmCIF, and polymer snapshot adapters into the core package.
The remaining scalar-volume, vector-flow, and Fermi-surface examples are still
examples-level prototypes.

## Goal

Task 2 should provide lightweight adapters that convert domain-specific input
files into the geometric objects already used by KnottedGraph:

```text
domain file -> input adapter -> internal geometric object -> plotting / downstream topology
```

The input layer should be useful across domains such as proteins, DNA/RNA,
polymers, coordinate chains, spatial engineering networks, surface meshes,
volumetric scalar fields, vector-flow volumes, and Fermi-surface-style data.

## Internal Object Targets

The Phase 1 API should standardize around three internal object types.

| Input class | Internal object | Existing convention |
| --- | --- | --- |
| Ordered curve or chain | `networkx.MultiGraph` with one edge carrying `pts` | node attribute `pos`, edge attribute `pts` |
| Embedded spatial graph | `networkx.MultiGraph` | node attribute `pos`, edge attribute `pts` |
| Surface or mesh | `pyvista.PolyData` | mesh points and cells |

For coordinate curves, the raw coordinate array should remain accessible because
many scientific users will want to inspect or save the original trace.

## Package Location

Implemented module layout:

```text
src/knotted_graph/
  inputs/
    __init__.py
    coordinate_chain.py
    mmcif.py
    pdb.py
    polymer.py
    spatial_graph_csv.py
    surface_mesh.py
```

Possible later modules:

```text
src/knotted_graph/
  inputs/
    volume.py
```

This keeps Task 2 separated from the existing skeleton, PD-code, and Yamada
calculation modules.

## Phase 1 Public API

Phase 1 exposes the most stable and general adapters:

```python
from knotted_graph.inputs import from_coordinate_chain
from knotted_graph.inputs import from_spatial_graph_csv
from knotted_graph.inputs import from_surface_mesh
```

The Phase 2 domain adapters are:

```python
from knotted_graph.inputs import from_pdb_backbone
from knotted_graph.inputs import from_protein_ca_backbone
from knotted_graph.inputs import from_nucleic_acid_backbone
from knotted_graph.inputs import from_mmcif_backbone
from knotted_graph.inputs import from_lammps_dump
from knotted_graph.inputs import from_gromacs_gro
```

Scalar-volume, vector-flow, and Fermi-surface examples remain in `examples/`
until their user-facing schemas are clearer.

## Result Containers

Each adapter should return a small result object rather than only a bare graph.
This makes examples and debugging easier.

```python
@dataclass
class CoordinateInputResult:
    input_id: str
    source_path: Path | None
    source_format: str
    coords: np.ndarray
    graph: nx.MultiGraph
    closed: bool
    closure_method: str | None
    metadata: dict
    issues: list[str]
```

```python
@dataclass
class SpatialGraphInputResult:
    graph_id: str
    nodes_path: Path
    edges_path: Path
    graph: nx.MultiGraph
    metadata: dict
    issues: list[str]
```

```python
@dataclass
class SurfaceInputResult:
    mesh_id: str
    source_path: Path
    source_format: str
    mesh: pv.PolyData
    metadata: dict
    issues: list[str]
```

The downstream code can use `result.graph` or `result.mesh`, while users can
still inspect parsing details.

Additional Phase 2 domain result containers include:

```python
PDBBackboneInputResult
MMCIFBackboneInputResult
PolymerInputResult
```

## Coordinate Chain API

Purpose: generic input for polymers, mathematical knots, DNA/RNA traces already
converted to points, and any ordered 3D chain.

```python
def from_coordinate_chain(
    source,
    *,
    input_id: str | None = None,
    source_format: str | None = None,
    columns: tuple[str, str, str] = ("x", "y", "z"),
    delimiter: str | None = None,
    closed: bool = False,
    closure: str | None = None,
    metadata: dict | None = None,
) -> CoordinateInputResult:
    ...
```

Expected sources:

- `numpy.ndarray` with shape `(N, 3)`;
- `.npy` storing an `(N, 3)` array;
- `.csv`, `.tsv`, `.dat` tables with x/y/z columns or bare coordinate rows;
- `.json` containing `points` or `coords`;
- `.xyz` coordinate files.

Validation:

- coordinates must have shape `(N, 3)`;
- at least two points are required;
- values must be finite numeric values;
- no silent row dropping;
- if `closed=True`, the coordinates must already be closed unless an explicit
  closure method is provided;
- if `closure="direct"`, append the first point when needed;
- if `closure=None`, preserve the input as given.

Returned internal graph:

```text
nx.MultiGraph
graph.graph["input_kind"] = "coordinate_curve"
node["pos"] = np.ndarray shape (3,)
edge["pts"] = np.ndarray shape (N, 3)
```

Open curves should have `start` and `end` nodes. Closed curves can be represented
as one self-loop edge attached to a loop anchor node, matching the current
prototype convention.

Example:

```python
from knotted_graph.inputs import from_coordinate_chain

result = from_coordinate_chain(
    "chain.xyz",
    closed=True,
    closure="direct",
    input_id="cinquefoil_xyz",
)
graph = result.graph
```

## Spatial Graph CSV API

Purpose: simple embedded graph input for engineering systems, electric circuits,
pipe networks, cooling systems, mechanical component networks, and other spatial
network systems.

```python
def from_spatial_graph_csv(
    nodes_csv,
    edges_csv,
    *,
    graph_id: str | None = None,
    node_id_col: str = "node_id",
    edge_id_col: str = "edge_id",
    source_col: str = "source",
    target_col: str = "target",
    coord_cols: tuple[str, str, str] = ("x", "y", "z"),
    points_col: str | None = "points_json",
    metadata: dict | None = None,
) -> SpatialGraphInputResult:
    ...
```

Minimal `nodes.csv`:

```csv
node_id,x,y,z,label,type
1,0,0,0,Component 1,component
2,1,0,0,Component 2,component
3,1,1,0,Component 3,component
4,0,1,0,Component 4,component
```

Minimal `edges.csv`:

```csv
edge_id,source,target,label,type
e1,1,2,Wire 1,wire
e2,2,3,Pipe 1,pipe
e3,3,4,Wire 2,wire
e4,4,1,Pipe 2,pipe
```

Optional curved edge column:

```csv
edge_id,source,target,label,type,points_json
e1,1,2,Curved Wire,wire,"[[0,0,0],[0.5,0.2,0.4],[1,0,0]]"
```

Validation:

- required node columns exist;
- required edge columns exist;
- coordinates are finite numeric values;
- node IDs are unique;
- edge IDs are unique when provided;
- every edge source and target exists in `nodes.csv`;
- `points_json`, if present, is a valid `(N, 3)` point list;
- the first and last points of `points_json` match source and target positions;
- optional columns such as `label` and `type` are preserved as attributes;
- invalid rows raise errors instead of being silently dropped.

Returned internal graph:

```text
nx.MultiGraph
graph.graph["input_kind"] = "spatial_graph_csv"
node["pos"] = np.ndarray shape (3,)
edge["pts"] = np.ndarray shape (N, 3)
edge["label"], edge["type"], etc. preserved when available
```

Example:

```python
from knotted_graph.inputs import from_spatial_graph_csv

result = from_spatial_graph_csv(
    "nodes.csv",
    "edges.csv",
    graph_id="cooling_network_demo",
)
graph = result.graph
```

## Surface Mesh API

Purpose: accept general surface-like inputs from geometry processing, Fermi
surface calculations, volume extraction, or scientific visualization tools.

```python
def from_surface_mesh(
    path,
    *,
    mesh_id: str | None = None,
    triangulate: bool = True,
    clean: bool = True,
    metadata: dict | None = None,
) -> SurfaceInputResult:
    ...
```

Supported Phase 1 formats:

- `.ply`
- `.obj`
- `.off`
- `.stl`
- `.vtk`
- `.vtp`

Validation:

- file suffix is supported;
- loaded object can be converted to `pyvista.PolyData`;
- mesh has points and cells;
- coordinates are finite;
- report open boundary edges as issues rather than silently ignoring them.

Returned object:

```text
pyvista.PolyData
```

Example:

```python
from knotted_graph.inputs import from_surface_mesh

result = from_surface_mesh("trefoil_tube.ply", mesh_id="trefoil_tube")
mesh = result.mesh
```

## PDB Backbone API

Purpose: load protein C-alpha backbones and DNA/RNA phosphate traces from PDB
files or RCSB PDB IDs.

```python
def from_pdb_backbone(
    source,
    *,
    pdb_id: str | None = None,
    chain_id: str | None = None,
    atom_name: str = "CA",
    model_id: int = 1,
    residue_names: set[str] | None = None,
    data_dir=None,
    download: bool = True,
    save_coords: bool = False,
    closed: bool = False,
    closure: str | None = None,
    metadata: dict | None = None,
) -> PDBBackboneInputResult:
    ...
```

Convenience wrappers:

```python
from_protein_ca_backbone("1CRN", chain_id="A", data_dir="data", save_coords=True)
from_nucleic_acid_backbone("1BNA", chain_id="A", atom_name="P", data_dir="data")
```

Returned internal object: `networkx.MultiGraph(pos/pts)`.

## mmCIF Backbone API

Purpose: load an ordered atom trace from an RCSB mmCIF file without adding a
heavy Biopython dependency.

```python
def from_mmcif_backbone(
    source,
    *,
    pdb_id: str | None = None,
    chain_id: str | None = None,
    atom_name: str = "CA",
    model_id: int = 1,
    data_dir=None,
    download: bool = True,
    save_coords: bool = False,
    closed: bool = False,
    closure: str | None = None,
    metadata: dict | None = None,
) -> MMCIFBackboneInputResult:
    ...
```

Example:

```python
result = from_mmcif_backbone("1EHZ", chain_id="A", atom_name="P")
graph = result.graph
```

## Polymer Snapshot API

Purpose: load ordered polymer chains from common simulation snapshot formats.

```python
def from_lammps_dump(
    path,
    *,
    molecule_id: int | None = 1,
    sort_column: str = "id",
    closed: bool = False,
    closure: str | None = None,
    polymer_id: str | None = None,
    metadata: dict | None = None,
) -> PolymerInputResult:
    ...
```

```python
def from_gromacs_gro(
    path,
    *,
    atom_name: str | None = None,
    residue_name: str | None = None,
    output_unit_scale: float = 10.0,
    closed: bool = False,
    closure: str | None = None,
    polymer_id: str | None = None,
    metadata: dict | None = None,
) -> PolymerInputResult:
    ...
```

For ring polymers whose first and last points are not duplicated in the file,
use `closure="direct"` explicitly.

## Closure Policy

Closure should be explicit. The API should not hide topological choices.

Recommended values:

| Value | Meaning |
| --- | --- |
| `closure=None` | preserve input exactly |
| `closure="direct"` | append a straight segment from end to start |
| `closure="metadata_only"` | mark intended closedness without changing coordinates |

For Task 2 gallery figures, direct endpoint closure is acceptable for visual
inspection. For rigorous knot classification, closure strategy should be chosen
and documented by the user.

## Plotting Verification

The input API should not need a new plotting system. Examples can verify results
using existing plotting helpers or small Matplotlib/PyVista smoke plots.

Minimum verification for Phase 1:

- coordinate chain: draw blue curve, mark start/end or graph node points in red;
- spatial graph CSV: draw blue embedded edges, mark all graph nodes in red;
- surface mesh: draw blue surface with light gray projections if desired;
- save a PNG under the matching `examples/.../figures/` folder.

The current report-style gallery already follows this visual convention.

## Implementation Status

Completed in Phase 1:

1. Added `src/knotted_graph/inputs/` with three adapters:
   `coordinate_chain.py`, `spatial_graph_csv.py`, and `surface_mesh.py`.
2. Migrated stable logic from the current examples into those modules.
3. Added lightweight unit tests for the three public adapters.

Completed in Phase 2:

1. Added `pdb.py`, `mmcif.py`, and `polymer.py` under
   `src/knotted_graph/inputs/`.
2. Migrated protein C-alpha, DNA/RNA phosphate, mmCIF atom-trace, LAMMPS dump,
   and GROMACS GRO adapters into the core input API.
3. Updated the corresponding examples to call the public API.
4. Kept thin compatibility wrappers in the old example adapter files.
5. Added lightweight unit tests for the new domain adapters.
6. Kept PD-code and Yamada workflows unchanged.

Still to do:

1. Decide whether volumetric scalar fields should become a public API.
2. Decide whether Fermi-surface mesh examples need a separate convenience API
   or can simply use `from_surface_mesh`.

## Testing Plan

Implemented test files:

```text
tests/
  test_inputs_coordinate_chain.py
  test_inputs_spatial_graph_csv.py
  test_inputs_surface_mesh.py
```

Coordinate chain tests:

- valid array input;
- valid CSV/JSON/XYZ input;
- missing coordinate columns;
- non-numeric coordinates;
- wrong shape;
- NaN or infinite coordinates;
- open curve keeps start/end nodes;
- direct closure appends the first point.

Spatial graph CSV tests:

- valid node/edge CSV pair;
- missing node columns;
- missing edge columns;
- non-numeric node coordinates;
- duplicate node IDs;
- duplicate edge IDs;
- unknown source or target;
- valid `points_json`;
- invalid `points_json`;
- metadata preservation for `label` and `type`.

Surface mesh tests:

- valid small PLY/OFF mesh;
- unsupported suffix;
- empty mesh rejection;
- non-finite point rejection;
- open boundary reported in `issues`.

## Current Prototype Coverage

The examples currently cover:

| Domain | Formats | Status |
| --- | --- | --- |
| Proteins | `.pdb` | core API |
| DNA | `.pdb` | core API |
| RNA / mmCIF | `.cif` | core API |
| Polymers | LAMMPS dump, GROMACS `.gro` | core API |
| Coordinate chains | `.csv`, `.json`, `.tsv`, `.dat`, `.xyz`, `.npy` | core API |
| Spatial graphs | JSON, node/edge CSV | CSV core API, JSON example prototype |
| Electric circuits | JSON spatial network | example prototype |
| Surface meshes | `.obj`, `.off`, `.ply`, `.stl`, `.vtk`, `.vtp` | strong Phase 1 candidate |
| Volumetric fields | `.npy`, `.npz` scalar fields | example prototype |
| Vector-flow volumes | `.npz` vector fields | example prototype |
| Fermi surfaces | `.vtp` meshes | example prototype |

## What This API Does Not Promise Yet

- It does not guarantee that every input can produce a PD code.
- It does not change Yamada-polynomial calculation.
- It does not simplify PD codes.
- It does not perform automatic spatial-graph extraction from every surface.
- It does not choose a mathematically rigorous closure strategy for every open
  biological curve.

The current promise is narrow and clear: common external data formats can be
converted into KnottedGraph-compatible geometric objects in a documented,
validated, and reproducible way.
