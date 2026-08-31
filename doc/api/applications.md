# Applications

Application APIs assemble domain-specific models around the generic graph,
projection, and invariant core. They do not add generic file readers. Use the
{doc}`../feature_status` matrix to distinguish base application helpers from
optional `nodal` workflows and to review their scaling boundaries.

The application layer is pre-alpha. Prefer the generic core when an ordinary
embedded `networkx.MultiGraph` already represents the scientific object.

## Mathematical graph families

```{eval-rst}
.. automodule:: knotted_graph.applications.mathematical
   :members: graph_family_names, build_graph_case
```

## Analytic knot-field deformations

`KnotFunction` and `KnotFunctionPath` are constructed from the base
{mod}`knotted_graph.inputs` API. Running a sampled level-set deformation needs
the `knot-fields` extra because each cell extracts a 3-D level set and spatial
graph. The returned result retains per-cell errors and phase signatures rather
than silently discarding failed cells.

```{eval-rst}
.. automodule:: knotted_graph.applications.knot_deformation
   :members: KnotDeformationRecord, KnotDeformationScan, KnotDeformationScanResult
```

## Unified Yamada phase maps

`make_yamada_phase_map` provides one finite-grid interface for analytic knot
fields, nodal Bloch-vector paths, and in-memory material Hamiltonians. Install
`knot-fields` for knot sources or `nodal` for Hamiltonian/material sources.
The source objects remain application-specific; this function is not a generic
Hamiltonian-file reader.

```{eval-rst}
.. automodule:: knotted_graph.applications.phase_maps
   :members: YamadaPhaseRecord, YamadaPhaseMapResult, MaterialBandEnergySurface, align_material_hamiltonians, pad_material_hamiltonian, make_yamada_phase_map
```

## Nodal and material workflows

Executing `NodalSkeleton` or `MaterialFermiSurface` requires the `nodal` extra;
the listed material symbolic-Hamiltonian constructors themselves use the base
SymPy stack. These workflows operate on in-memory SymPy Hamiltonians or Bloch
vectors rather than Hamiltonian files.

```{eval-rst}
.. automodule:: knotted_graph.applications.nodal.models
   :members:

.. automodule:: knotted_graph.applications.nodal.skeleton
   :members: NodalSkeleton

.. automodule:: knotted_graph.applications.materials
   :members: H_Ti3Al_sympy, H_D6_sympy, H_YH3_sympy

.. py:class:: MaterialFermiSurface
   :module: knotted_graph.applications.materials

   Analyze an in-memory Hermitian multiband Hamiltonian and expose its sampled
   surface and embedded skeleton.  This class is loaded lazily and requires the
   ``nodal`` optional dependencies; see the feature-status matrix before using
   it in a workflow.
```
