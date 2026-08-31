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
