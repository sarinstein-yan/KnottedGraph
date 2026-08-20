# Architecture

The repository is organized around generic mathematical components first, with
applications layered on top.

## Submodules

- `knotted_graph.core`: embedded graph validation, graph constructors, and graph
  simplification helpers.
- `knotted_graph.inputs`: adapters that normalize external data into core
  graph or mesh objects.
- `knotted_graph.extraction`: production sparse skeleton-image conversion.
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

## Skeleton Extraction Policy

`knotted_graph.extraction.skeleton_image_to_graph` is the canonical
skeleton-to-graph entry point. For every 3-D image, `backend="auto"` uses the
second-generation sparse extractor: empty image margins are cropped before
foreground indexing, 26-neighbour adjacency is generated in exact historical
lexicographic order, and returned coordinates remain in the original global
voxel frame. `knotted_graph.extraction.skeleton` exports the same function
objects, so the package and submodule import paths cannot diverge.

The historical `poly2graph.skeleton2graph` parser is retained only behind the
explicit `backend="poly2graph"` compatibility route used by regression and
benchmark code. It is not a normal 3-D production path.

By default, optimized extraction preserves the zero-radius historical topology.
A caller that knows a genuine valence bound may pass `max_junction_degree` to
enable the fail-closed persistence-based junction repair. The library does not
assume trivalence globally: if no bound is supplied, generic higher-valence
spatial graphs are not contracted merely for optimization.

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
