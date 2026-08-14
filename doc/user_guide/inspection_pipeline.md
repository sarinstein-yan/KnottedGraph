# Inspecting Intermediate Objects

Many scientific users need more than a final polynomial. They may want to see
the input geometry, surface, skeleton, raw graph, simplified graph, planar
diagram, PD code, selected projection, and final invariant. KnottedGraph exposes
these objects through modular functions so intermediate stages can be inspected,
visualized, or exported.

## A Complete Inspection Example

Surfaces in KnottedGraph can come from several sources: imported meshes,
biomolecular or geometric constructions, or application-specific computations.
This section shows one application-driven case. The surface below is called an
`exceptional_surface` because it comes from the exceptional set of a sampled
non-Hermitian Hamiltonian, but the same inspection pattern applies after any
workflow has produced a spatial graph.

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import solomon_bloch_vector
from knotted_graph.projection import sample_projections, select_projection

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
bloch_vector = solomon_bloch_vector(0.42, k_symbols=(kx, ky, kz))

ske = NodalSkeleton(
    bloch_vector,
    k_symbols=(kx, ky, kz),
    dimension=48,
)

surface = ske.exceptional_surface_pv
surface_for_plot = surface.connectivity("largest")
skeleton_points = ske.skeleton_coords
raw_graph = ske.skeleton_graph(simplify=False, smooth_epsilon=2)
simplified_graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)

projections = sample_projections(simplified_graph, num_rotation_samples=16)
projection = select_projection(simplified_graph, num_rotation_samples=16)
pd_code = projection.pd_code
```

Typical inspection output:

```text
surface.n_points, surface.n_cells
(4560, 9128)

surface_for_plot.n_points, surface_for_plot.n_cells
(4560, 9128)

skeleton_points.shape
(271, 3)

raw_graph.number_of_nodes(), raw_graph.number_of_edges()
(14, 20)

simplified_graph.number_of_nodes(), simplified_graph.number_of_edges()
(6, 12)

[p.num_crossings for p in projections]
[15, 15, 14, 14, 14, 16, 18, 18, 19, 15, 13, 11, 8, 8, 9, 12]

tuple(round(a, 2) for a in projection.rotation_angles), projection.num_crossings
((-149.91, 38.62, 0.0), 8)

pd_code
'V[3,2,0];V[9,11,7];V[21,11,19,10,2,1,12,14];V[27,8,6,25];V[26,13,18];V[27,20,24];X[23,15,22,16];X[5,12,4,13];X[10,19,9,20];X[25,17,26,18];X[6,16,5,17];X[22,15,21,14];X[7,23,8,24];X[4,1,3,0]'
```

The surface, skeleton points, raw graph, simplified graph, and planar diagram
all come from the same `ske` object. The largest connected component is the
full surface in this example, so the displayed mesh is not a cropped fragment.
The surface and `skeleton_coords` are in k-space coordinates. The
`skeleton_graph` object currently stores node `pos` and edge `pts` in skeleton
image-index coordinates, so the graph plots have different numerical axis
ranges even though they come from the same extraction.

## Plot Each Stage

Surface:

```python
import plotly.graph_objects as go

surface = ske.exceptional_surface_pv
surface_for_plot = surface.connectivity("largest").triangulate()
faces = surface_for_plot.faces.reshape(-1, 4)[:, 1:]
points = surface_for_plot.points

fig = go.Figure(
    go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="#263f39",
        opacity=0.94,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/inspection_surface_plot.png
:alt: Output from plotting the largest connected component of the exceptional-surface mesh
:width: 68%
:align: center
```

Skeleton points:

```python
import plotly.graph_objects as go

skeleton_points = ske.skeleton_coords
fig = go.Figure(
    go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode="markers",
        marker=dict(size=3),
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/inspection_skeleton_points.png
:alt: Output from plotting extracted skeleton points
:width: 68%
:align: center
```

Raw spatial graph:

```python
from knotted_graph.visualization import plot_3D_graph_plotly

raw_graph = ske.skeleton_graph(simplify=False, smooth_epsilon=2)
raw_fig = plot_3D_graph_plotly(raw_graph)
raw_fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
    ),
)
raw_fig.show()
```

```{image} ../assets/plot_outputs/inspection_raw_graph_plotly.png
:alt: Output from plotting the raw nodal-skeleton spatial graph
:width: 70%
:align: center
```

The raw graph keeps intermediate skeleton vertices that are useful for
debugging the extraction before simplification.

Simplified spatial graph:

```python
simplified_graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
simplified_fig = plot_3D_graph_plotly(simplified_graph)
simplified_fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
    ),
)
simplified_fig.show()
```

```{image} ../assets/plot_outputs/inspection_simplified_graph_plotly.png
:alt: Output from plotting the simplified nodal-skeleton spatial graph
:width: 70%
:align: center
```

Planar diagram:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
for arc in projection.arcs:
    ax.plot(*arc.line.xy, color="tab:blue", linewidth=2)
for vertex in projection.vertices:
    ax.scatter(*vertex.point.xy, color="tab:red", zorder=3)
for crossing in projection.crossings:
    ax.scatter(*crossing.point.xy, marker="x", color="black", zorder=4)

ax.set_aspect("equal")
ax.axis("off")
plt.show()
```

```{image} ../assets/plot_outputs/inspection_planar_diagram.png
:alt: Output from plotting a nodal-skeleton planar diagram
:width: 58%
:align: center
```

## Inspect The Selected Projection

Use the selected projection object when you want to debug the exact diagram
used downstream.

```python
print(f"rotation_angles = {tuple(round(a, 2) for a in projection.rotation_angles)}")
print(f"crossings = {projection.num_crossings}")
print(f"pd_terms = {sorted(projection.pd_code.split(';'))}")
print(f"vertices = {[(v.id, v.key) for v in projection.vertices]}")
print(
    "crossing_points =",
    [(x.id, tuple(round(c, 3) for c in tuple(x.point.coords)[0])) for x in projection.crossings],
)
print(
    "arcs =",
    [(arc.id, arc.start_type, arc.start_id, arc.end_type, arc.end_id) for arc in projection.arcs],
)
```

Output:

```text
rotation_angles = (-149.91, 38.62, 0.0)
crossings = 8
pd_terms = ['V[21,11,19,10,2,1,12,14]', 'V[26,13,18]', 'V[27,20,24]', 'V[27,8,6,25]', 'V[3,2,0]', 'V[9,11,7]', 'X[10,19,9,20]', 'X[22,15,21,14]', 'X[23,15,22,16]', 'X[25,17,26,18]', 'X[4,1,3,0]', 'X[5,12,4,13]', 'X[6,16,5,17]', 'X[7,23,8,24]']
vertices = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
crossing_points = [(0, (-14.301, -36.693)), (1, (-17.317, -33.422)), (2, (-3.589, -45.553)), (3, (-22.789, -35.928)), (4, (-18.683, -36.149)), (5, (-13.848, -36.243)), (6, (-18.152, -40.522)), (7, (-14.174, -32.352))]
arcs = [(0, 'v', 0, 'x', 7), (1, 'x', 7, 'v', 2), (2, 'v', 0, 'v', 2), (3, 'v', 0, 'x', 7), (4, 'x', 7, 'x', 1), (5, 'x', 1, 'x', 4), (6, 'x', 4, 'v', 3), (7, 'v', 1, 'x', 6), (8, 'x', 6, 'v', 3), (9, 'v', 1, 'x', 2), (10, 'x', 2, 'v', 2), (11, 'v', 1, 'v', 2), (12, 'v', 2, 'x', 1), (13, 'x', 1, 'v', 4), (14, 'v', 2, 'x', 5), (15, 'x', 5, 'x', 0), (16, 'x', 0, 'x', 4), (17, 'x', 4, 'x', 3), (18, 'x', 3, 'v', 4), (19, 'v', 2, 'x', 2), (20, 'x', 2, 'v', 5), (21, 'v', 2, 'x', 5), (22, 'x', 5, 'x', 0), (23, 'x', 0, 'x', 6), (24, 'x', 6, 'v', 5), (25, 'v', 3, 'x', 3), (26, 'x', 3, 'v', 4), (27, 'v', 3, 'v', 5)]
```

This output tells the user which projection was used, how many crossings were
found, how the graph vertices and crossings were encoded in PD form, and how
the arcs connect vertices to crossing points.

## Paper Notebook Surface-To-Graph Panels

The appendix notebook uses the same inspection idea on a thicker torus example.
The point of the figure is not only the final graph: it shows the surface, the
skeleton points, and the spatial graph as separate inspectable objects.

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import solomon_bloch_vector

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

ske = NodalSkeleton(
    solomon_bloch_vector(0.2, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=120,
    axis_scale=(1.0, 1.0, 1.5),
)

surface = ske.exceptional_surface_pv.connectivity("largest")
skeleton_points = ske.skeleton_coords
raw_graph = ske.skeleton_graph(simplify=False, smooth_epsilon=2)
simplified_graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)

print(surface.n_points, surface.n_cells)
print(skeleton_points.shape)
print(raw_graph.number_of_nodes(), raw_graph.number_of_edges())
print(simplified_graph.number_of_nodes(), simplified_graph.number_of_edges())
```

Output:

```text
surface.n_points, surface.n_cells
(7176, 14352)

skeleton_points.shape
(618, 3)

raw_graph.number_of_nodes(), raw_graph.number_of_edges()
(20, 20)

simplified_graph.number_of_nodes(), simplified_graph.number_of_edges()
(2, 2)
```

Plot the surface with Plotly by converting the PyVista mesh:

```python
import plotly.graph_objects as go

mesh = surface.triangulate()
faces = mesh.faces.reshape(-1, 4)[:, 1:]
pts = mesh.points

fig = go.Figure(
    go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="#263f39",
        opacity=0.9,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_surface_gamma_0p2.png
:alt: Torus surface at gamma equals 0.2
:width: 70%
:align: center
```

Plot the skeleton points:

```python
fig = go.Figure(
    go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode="markers",
        marker=dict(size=3, color="#348ABD"),
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_skeleton_gamma_0p2.png
:alt: Torus skeleton points at gamma equals 0.2
:width: 70%
:align: center
```

Plot the extracted graph:

```python
from knotted_graph.visualization import plot_3D_graph_plotly

fig = plot_3D_graph_plotly(raw_graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_graph_gamma_0p2.png
:alt: Torus spatial graph at gamma equals 0.2
:width: 70%
:align: center
```

The important inspection rule is that every object shown here comes from the
same `ske` object. Instead of displaying stitched publication panels, this page
keeps the individual outputs separate so users can reproduce and debug each
pipeline stage.

```python
ske_006 = NodalSkeleton(
    solomon_bloch_vector(0.06, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=120,
    axis_scale=(1.0, 1.0, 1.5),
)
surface_006 = ske_006.exceptional_surface_pv.connectivity("largest")

mesh = surface_006.triangulate()
faces = mesh.faces.reshape(-1, 4)[:, 1:]
pts = mesh.points
fig = go.Figure(
    go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="#263f39",
        opacity=0.9,
    )
)
fig.update_layout(title=None, scene=dict(aspectmode="data"))
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_surface_gamma_0p06.png
:alt: Torus surface at gamma equals 0.06
:width: 70%
:align: center
```

```python
skeleton_points_006 = ske_006.skeleton_coords

fig = go.Figure(
    go.Scatter3d(
        x=skeleton_points_006[:, 0],
        y=skeleton_points_006[:, 1],
        z=skeleton_points_006[:, 2],
        mode="markers",
        marker=dict(size=3, color="#348ABD"),
    )
)
fig.update_layout(title=None, scene=dict(aspectmode="data"))
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_skeleton_gamma_0p06.png
:alt: Torus skeleton points at gamma equals 0.06
:width: 70%
:align: center
```

```python
raw_graph_006 = ske_006.skeleton_graph(simplify=False, smooth_epsilon=2)

fig = plot_3D_graph_plotly(raw_graph_006)
fig.update_layout(title=None, scene=dict(aspectmode="data"))
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_graph_gamma_0p06.png
:alt: Torus spatial graph at gamma equals 0.06
:width: 70%
:align: center
```

## Recommended Pipeline Mode

For future public-facing releases, the most user-friendly interface would be a
single pipeline result object collecting the same pieces:

```python
pipeline = inspect_spatial_graph(graph, num_rotation_samples=12)

pipeline.input_graph
pipeline.simplified_graph
pipeline.projections
pipeline.selected_projection
pipeline.pd_code
pipeline.yamada_polynomial
```

This would not replace the lower-level modules. Instead, it would provide a
convenient inspection mode for users who want every intermediate object without
manually calling each module.
