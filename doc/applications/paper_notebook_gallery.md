# Notebook Figure Gallery

This page keeps only notebook figures that directly teach the KnottedGraph
pipeline.  The old manuscript assembly figures, crop variants, standalone
NetworkX drawings, planarity-reference drawings, and energy-window schematics
are intentionally not repeated here.  They are paper assets, not library usage
examples.

The rule on this page is strict: every displayed figure is preceded by code
showing the object that produces it.  For 3D spatial graphs, the code always
uses the same explicit `plot_3D_graph_plotly(...)` layout.

## Torus Model: Surface

Start from the Hamiltonian model, construct the nodal skeleton object, and
extract the surface.

```python
import plotly.graph_objects as go
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import pq_torus_knot_bloch_vector

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

ske = NodalSkeleton(
    pq_torus_knot_bloch_vector(1, 2, 0.20, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=120,
    axis_scale=(1.0, 1.0, 1.5),
)

surface = ske.exceptional_surface_pv.connectivity("largest")
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
        color="#1f77b4",
        opacity=0.58,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_surface_gamma_0p2.png
:alt: Torus-model exceptional surface at gamma equals 0.2
:width: 70%
:align: center
```

## Torus Model: Skeleton Points

The skeleton points are extracted from the same `ske` object.

```python
skeleton_points = ske.skeleton_coords

fig = go.Figure(
    go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode="markers",
        marker=dict(size=3, color="#1f77b4"),
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_skeleton_gamma_0p2.png
:alt: Skeleton points extracted from the torus-model surface at gamma equals 0.2
:width: 70%
:align: center
```

## Torus Model: Spatial Graph

The spatial graph is the graph object used by projection and Yamada
calculation.

```python
from knotted_graph.visualization import plot_3D_graph_plotly

raw_graph = ske.skeleton_graph(simplify=False, smooth_epsilon=2)

fig = plot_3D_graph_plotly(raw_graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_graph_gamma_0p2.png
:alt: Spatial graph extracted from the torus-model surface at gamma equals 0.2
:width: 70%
:align: center
```

## Thinner Torus Model

Changing only the model thickness parameter produces a different surface while
keeping the same extraction code.

```python
ske_thin = NodalSkeleton(
    pq_torus_knot_bloch_vector(1, 2, 0.06, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=120,
    axis_scale=(1.0, 1.0, 1.5),
)

surface = ske_thin.exceptional_surface_pv.connectivity("largest")
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
        color="#1f77b4",
        opacity=0.58,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_surface_gamma_0p06.png
:alt: Torus-model exceptional surface at gamma equals 0.06
:width: 70%
:align: center
```

```python
skeleton_points = ske_thin.skeleton_coords

fig = go.Figure(
    go.Scatter3d(
        x=skeleton_points[:, 0],
        y=skeleton_points[:, 1],
        z=skeleton_points[:, 2],
        mode="markers",
        marker=dict(size=3, color="#1f77b4"),
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_skeleton_gamma_0p06.png
:alt: Skeleton points extracted from the torus-model surface at gamma equals 0.06
:width: 70%
:align: center
```

```python
raw_graph = ske_thin.skeleton_graph(simplify=False, smooth_epsilon=2)

fig = plot_3D_graph_plotly(raw_graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_skeletonization_torus_graph_gamma_0p06.png
:alt: Spatial graph extracted from the torus-model surface at gamma equals 0.06
:width: 70%
:align: center
```

## Three-Link Planarity Surfaces

The planarity examples also begin with a Hamiltonian model.  The graph-theory
drawings used in the manuscript are not repeated here; the library-relevant
outputs are the surfaces and the extracted graphs shown in
[Mathematical Workflows](mathematical_workflows.md).

```python
from knotted_graph.applications.nodal.models import threelink_bloch_vector

ske = NodalSkeleton(
    threelink_bloch_vector(0.116, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=100,
    axis_scale=(1.0, 1.0, 1.5),
)
surface = ske.exceptional_surface_pv.connectivity("largest")
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
        color="#1f77b4",
        opacity=0.58,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_planarity_threelink_surf_0p116.png
:alt: Three-link exceptional surface at gamma equals 0.116
:width: 70%
:align: center
```

```python
ske = NodalSkeleton(
    threelink_bloch_vector(0.41, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=100,
    axis_scale=(1.0, 1.0, 1.5),
)
surface = ske.exceptional_surface_pv.connectivity("largest")
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
        color="#1f77b4",
        opacity=0.58,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_planarity_threelink_surf_0p41.png
:alt: Three-link exceptional surface at gamma equals 0.41
:width: 70%
:align: center
```

```python
ske = NodalSkeleton(
    threelink_bloch_vector(0.50, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=100,
    axis_scale=(1.0, 1.0, 1.5),
)
surface = ske.exceptional_surface_pv.connectivity("largest")
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
        color="#1f77b4",
        opacity=0.58,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_planarity_threelink_surf_0p5.png
:alt: Three-link exceptional surface at gamma equals 0.5
:width: 70%
:align: center
```

## Intrinsic-Linkedness Input Surface

The intrinsic-linkedness computation starts from an extracted spatial graph.
The standalone Petersen/certificate drawings are omitted here because they are
graph-theory illustrations rather than KnottedGraph visualization outputs.

```python
from knotted_graph.applications.nodal.models import awesome_bloch_vector

ske = NodalSkeleton(
    awesome_bloch_vector(0.16, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=80,
)
surface = ske.exceptional_surface_pv.connectivity("largest")
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
        color="#1f77b4",
        opacity=0.58,
    )
)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_intrinsic_awesomesurface_auto.png
:alt: Awesome-model surface used as input to the intrinsic-linkedness workflow
:width: 70%
:align: center
```

The extracted graph and linkedness diagnostics are discussed in
[Mathematical Workflows](mathematical_workflows.md), where the surface,
spatial graph, and graph-theory result appear in that order.
