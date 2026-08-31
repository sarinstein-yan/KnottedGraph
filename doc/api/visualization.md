# Visualization

Visualization helpers are optional presentation tools. Install the `viz` extra
before using the two Plotly functions:

```bash
uv sync --extra viz
```

They consume points or an already constructed graph; they do not parse files,
validate an embedding, choose a projection for invariant computation, or change
graph topology. For an embedded graph:

```python
from knotted_graph.visualization import plot_3D_graph_plotly

figure = plot_3D_graph_plotly(graph)
figure.show()
```

In headless environments, configure a non-interactive renderer or export the
returned Plotly figure instead of calling `show()`. See the
{doc}`../troubleshooting` guide for display-backend problems.

`standard_petersen_layout` and `draw_petersen_embedding` use the base Matplotlib
stack. The Plotly helpers require `viz`. All four names are exported from
`knotted_graph.visualization`, not from the package root.

## Public helpers

```{eval-rst}
.. autofunction:: knotted_graph.visualization.standard_petersen_layout

.. autofunction:: knotted_graph.visualization.draw_petersen_embedding

.. py:function:: plot_3D_and_projections_plotly(points)
   :module: knotted_graph.visualization

   Plot a three-dimensional point cloud together with its three coordinate-plane
   projections.

   :param points: Array-like points with three coordinates per row.
   :returns: A Plotly figure containing one 3-D and three 2-D panels.
   :rtype: plotly.graph_objects.Figure

.. autofunction:: knotted_graph.visualization.plot_3D_graph_plotly
```
