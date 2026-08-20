# Architecture

The repository is organized around generic mathematical components first, with
applications layered on top.

## Submodules

- `knotted_graph.core`: embedded graph validation, graph constructors, and graph
  simplification helpers.
- `knotted_graph.inputs`: adapters that normalize external data into core
  graph or mesh objects.
- `knotted_graph.extraction`: optional skeleton-image conversion helpers.
- `knotted_graph.projection`: rotations, projection sampling, and PD-code
  objects.
- `knotted_graph.invariants.yamada`: Yamada polynomial evaluation for resolved
  graphs and diagrams.
- `knotted_graph.layout.repulsive`: optional curve-network relaxation.
- `knotted_graph.visualization`: generic graph plotting helpers.
- `knotted_graph.applications.nodal`: non-Hermitian nodal-skeleton workflow.

## Import Boundary

`import knotted_graph` must not import optional application stacks such as
PyVista, scikit-image, poly2graph, minorminer, or Plotly. Application workflows
are imported explicitly from their application namespace.

## Yamada Projection Policy

`knotted_graph.projection.compute_yamada_polynomial` is the canonical embedded
spatial-graph entry point. If `rotation_angles` is supplied, it computes exactly
that projection. Otherwise it samples `num_rotation_samples=10` rotations,
chooses the valid projection with the fewest crossings, and emits a
`RuntimeWarning` when the selected diagram has at least 10 crossings.

## Inspection-Oriented API Direction

Intermediate objects are part of the public user story, not only internal
implementation details. Users should be able to inspect or export the major
pipeline stages used in the paper figures: imported input, surface or mesh,
skeleton image, raw spatial graph, simplified spatial graph, sampled
projections, selected planar diagram, PD code, and invariant output.

Current graph-to-Yamada workflows expose this pattern through
`sample_projections`, `select_projection`, and
`compute_yamada_polynomial(..., return_result=True)`. Future high-level
pipeline helpers should preserve these lower-level APIs while collecting the
same intermediate objects into a convenient result object for tutorials,
figures, and reproducible analyses.
