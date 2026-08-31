# Layout

The repulsive-layout interface accepts an embedded graph and returns a
`GraphLayoutResult`. The direct graph call imports with the base package. The
`repulsion` extra adds Biopython/Plotly helpers used by protein examples and
HTML rendering:

```bash
uv sync --extra repulsion
```

That extra does **not** install the external C++ Repulsor solver or its native
libraries. Complete the {doc}`../user_guide/repulsive_layout` setup before
calling the solver. Validate `result.graph` before continuing to projection;
layout is a geometric preprocessing step, not an invariant calculation.

## Public Python interface

```{eval-rst}
.. automodule:: knotted_graph.layout.repulsive
   :members: GraphLayoutResult, DriverConfig, SolverOptions, relax_spatial_graph
```
