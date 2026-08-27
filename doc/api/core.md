# Core Spatial Graphs

The core representation is an undirected `networkx.MultiGraph`. Every node has
a finite three-dimensional `pos`; every edge has an ordered `(N, 3)` polyline
`pts` connecting the incident node positions. Parallel edges and self-loops are
preserved.

Use `validate_embedding(graph)` to inspect issues without mutation and
`ensure_embedding(graph)` to return a normalized copy. When `pts` is absent,
normalization materializes a straight segment between the endpoint positions.

`simplify_edges` collapses degree-two chains in components containing cycles or
junctions while preserving their embedded polylines. For an acyclic graph it
returns normalized connectivity unchanged; it never silently converts a path
or tree into isolated nodes. `remove_leaf_nodes` is the explicit structural
operation that repeatedly removes degree-one branches.

```{eval-rst}
.. automodule:: knotted_graph.core
   :members:
```
