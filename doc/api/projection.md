# Projection and PD Code

Projection converts an embedded spatial `MultiGraph(pos/pts)` into a regular
two-dimensional diagram. Geometric intersections in that view become crossing
records; they do **not** become graph vertices.

Use {py:func}`knotted_graph.projection.select_projection` when you want to
inspect the chosen view before evaluating an invariant. With explicit
`rotation_angles`, exactly that view is used. Without them, KnottedGraph samples
the requested number of orientations and selects the valid projection with the
fewest crossings.

```python
from knotted_graph.projection import select_projection

projection = select_projection(graph, num_rotation_samples=10)
print(projection.rotation_angles)
print(projection.num_crossings)
print(projection.pd_code)
```

Angles are in degrees. Use an uppercase order such as `ZYX` for extrinsic
rotations or a lowercase order such as `xyz` for intrinsic rotations. A mixed-
case or non-axis order is rejected before geometry is processed.

If some sampled views are degenerate, a warning reports how many failed while
valid projections remain available. Collinear overlaps are not silently
interpreted as crossings. See [Troubleshooting](../troubleshooting.md) for
regular-projection and runtime guidance.

```{eval-rst}
.. automodule:: knotted_graph.projection.pd_code
   :members:
```
