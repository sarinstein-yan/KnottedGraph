# Inputs

The public input API lives in `knotted_graph.inputs`. Its high-level `from_*`
loaders normalize external geometry into either an embedded
`networkx.MultiGraph` or a PyVista surface mesh and return a result dataclass
containing the converted object, source information, and an `issues` list.
The same module also exposes lower-level converters, validators, and writers
with the return types documented in their individual signatures.

```python
from knotted_graph.inputs import from_coordinate_chain
```

Input adapters are not re-exported from the top-level `knotted_graph` namespace.
For task-oriented guidance, schemas, units, and limitations, start with the
[Input Adapters user guide](../user_guide/input_adapters.md).

## Support summary

```{list-table}
:header-rows: 1
:class: kg-route-table
:widths: 23 33 22 22

* - Family
  - Accepted input
  - Primary call
  - Result payload
* - Coordinate chain
  - Array, CSV, DAT, JSON, NPY, TSV, TXT, XYZ
  - `from_coordinate_chain`
  - `.graph`
* - Biomolecular trace
  - PDB or mmCIF path/RCSB identifier
  - `from_pdb_backbone`, `from_mmcif_backbone`
  - `.graph`
* - Polymer snapshot
  - GROMACS GRO, first-frame LAMMPS dump
  - `from_gromacs_gro`, `from_lammps_dump`
  - `.graph`
* - Spatial graph
  - Paired node and edge CSV files
  - `from_spatial_graph_csv`
  - `.graph`
* - Surface mesh
  - OBJ, OFF, PLY, STL, VTK, VTP
  - `from_surface_mesh`
  - `.mesh`
```

Graph-producing adapters follow the `MultiGraph(pos/pts)` contract: node `pos`
values are finite 3D points and edge `pts` values are finite 3D polylines whose
endpoints match their incident nodes. Surface loading is lazy and requires the
`surface` optional dependency.

Generic GraphML, generic graph JSON, edge-list, SWC, NPZ field/volume, and
Hamiltonian files do not currently have public adapters in this module.

## Coordinate chains

```{eval-rst}
.. autoclass:: knotted_graph.inputs.CoordinateInputResult

.. autofunction:: knotted_graph.inputs.validate_coords

.. autofunction:: knotted_graph.inputs.coordinates_to_multigraph

.. autofunction:: knotted_graph.inputs.from_coordinate_chain
```

## PDB and mmCIF traces

```{eval-rst}
.. autoclass:: knotted_graph.inputs.PDBBackboneInputResult

.. autofunction:: knotted_graph.inputs.from_pdb_backbone

.. autofunction:: knotted_graph.inputs.from_protein_ca_backbone

.. autofunction:: knotted_graph.inputs.from_nucleic_acid_backbone

.. autoclass:: knotted_graph.inputs.MMCIFBackboneInputResult

.. autofunction:: knotted_graph.inputs.from_mmcif_backbone
```

## Polymer snapshots

```{eval-rst}
.. autoclass:: knotted_graph.inputs.PolymerInputResult

.. autofunction:: knotted_graph.inputs.from_gromacs_gro

.. autofunction:: knotted_graph.inputs.from_lammps_dump

.. autofunction:: knotted_graph.inputs.write_gro_coords

.. autofunction:: knotted_graph.inputs.write_lammps_dump
```

## Spatial-graph CSV

```{eval-rst}
.. autoclass:: knotted_graph.inputs.SpatialGraphInputResult

.. autofunction:: knotted_graph.inputs.from_spatial_graph_csv

.. autofunction:: knotted_graph.inputs.validate_spatial_graph
```

## Surface meshes

Install `knotted_graph[surface]` before using this group.

```{eval-rst}
.. autoclass:: knotted_graph.inputs.SurfaceInputResult

.. autofunction:: knotted_graph.inputs.from_surface_mesh

.. autofunction:: knotted_graph.inputs.validate_surface_mesh
```

## Loader result and error contract

- A successful high-level `from_*` load returns a family-specific result
  dataclass. Access its `.graph` or `.mesh` payload explicitly.
- Lower-level helpers have different contracts: for example,
  `coordinates_to_multigraph` returns a bare `MultiGraph`, validators return
  normalized data or issue lists, and writers return `None`.
- `result.issues` contains non-fatal validation findings. Inspect it before
  downstream processing.
- Schema and parsing failures raise an exception, commonly `ValueError`,
  `FileNotFoundError`, or `RuntimeError` for an insufficient atom trace.
- Passing adapter validation establishes the in-memory geometry contract, not
  suitability for a specific invariant or application.
- For PDB and mmCIF results, source-selection details are available through
  dedicated fields such as `pdb_id`, `chain_id`, `atom_name`, and `records`, and
  on `result.graph.graph`.
