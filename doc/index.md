# KnottedGraph

`knotted_graph` is a generic package for spatial graph computation, planar
diagram construction, and graph polynomial invariants. Application workflows,
such as non-Hermitian nodal skeletons, are layered on top of the generic core.

```{image} ../assets/paper/architecture.svg
:alt: KnottedGraph architecture
:width: 90%
:align: center
```

## Package Shape

The public source layout follows the architecture diagram:

- `knotted_graph.core` for spatial graph utilities and graph constructors.
- `knotted_graph.inputs` for adapters from external coordinate, polymer,
  biomolecular, CSV, and surface-mesh data.
- `knotted_graph.projection` for rotations, planar diagrams, PD codes, and
  graph-to-Yamada entry points.
- `knotted_graph.invariants.yamada` for Yamada polynomial evaluation engines.
- `knotted_graph.layout.repulsive` for optional curve-network relaxation.
- `knotted_graph.applications.nodal` for the non-Hermitian nodal-skeleton
  application.

```{toctree}
:maxdepth: 2

installation
quickstart
applications/index
api/index
developer/architecture
```
