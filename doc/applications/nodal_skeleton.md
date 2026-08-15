# Non-Hermitian Nodal Skeletons

The nodal-skeleton workflow is the original physics application that motivated
this package. It samples a two-band Hamiltonian in 3D momentum space, extracts
the exceptional-surface interior, skeletonizes that region, and returns a
spatial multigraph that can be projected to a planar diagram.

The word "exceptional" is specific to this non-Hermitian application. In the
generic library, surfaces can also come from imported meshes or other external
workflows; once a spatial graph is available, projection and Yamada computation
use the same API.

Input source: a symbolic Bloch vector or Hamiltonian sampled on a 3D
momentum-space grid.

The input is a two-band non-Hermitian Bloch Hamiltonian

```{math}
H(\mathbf{k}) =
d_x(\mathbf{k})\sigma_x+
d_y(\mathbf{k})\sigma_y+
d_z(\mathbf{k})\sigma_z,\qquad
\mathbf{d}(\mathbf{k})=(\operatorname{Re} f,\; i\gamma,\; \operatorname{Im} f).
```

For the torus-knot/link model used below,

```{math}
\begin{aligned}
z &= \cos(2k_z)+c+i(\cos k_x+\cos k_y+\cos k_z-m),\\
w &= \sin k_x+i\sin k_y,\\
f_{p,q}(\mathbf{k}) &= z^p-w^q.
\end{aligned}
```

The exceptional surface is extracted from this Hamiltonian first; only after
that does the library skeletonize the surface and convert the result to an
embedded spatial graph.

## Minimal Workflow

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import hopf_link_bloch_vector

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
bloch_vector = hopf_link_bloch_vector(0.3, k_symbols=(kx, ky, kz))

ske = NodalSkeleton(
    bloch_vector,
    k_symbols=(kx, ky, kz),
    dimension=48,
)

surface = ske.exceptional_surface_pv
skeleton_points = ske.skeleton_coords
graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)

print(graph.number_of_nodes(), graph.number_of_edges())
print(graph.graph["is_trivalent"])
```

Output:

```text
2 3
True
```

The returned graph follows the generic spatial-graph contract: each node has a
finite 3D `pos` attribute and each geometric edge has a `pts` polyline. The
same `ske` object keeps the surface, skeleton points, raw graph, simplified
graph, and planar diagram available for inspection.

Plot the exceptional surface:

```python
import plotly.graph_objects as go

surface_for_plot = surface.connectivity("largest").triangulate()
faces = surface_for_plot.faces.reshape(-1, 4)[:, 1:]
pts = surface_for_plot.points

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

```{image} ../assets/plot_outputs/nodal_hopf_surface_plot.png
:alt: Output from plotting the exceptional surface before skeletonization
:width: 72%
:align: center
```

Plot the skeleton points extracted from that same surface:

```python
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

```{image} ../assets/plot_outputs/nodal_hopf_skeleton_points.png
:alt: Output from plotting skeleton points extracted from the exceptional surface
:width: 72%
:align: center
```

Plot the simplified spatial graph that will be projected and sent to Yamada
calculation:

```python
from knotted_graph.visualization import plot_3D_graph_plotly

fig = plot_3D_graph_plotly(graph)
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

```{image} ../assets/plot_outputs/nodal_hopf_graph_plotly.png
:alt: Output from plotting the simplified spatial graph extracted from the surface
:width: 72%
:align: center
```

For a more verbose debugging view, including raw graph and planar diagram
objects, see [Inspecting Intermediate Objects](../user_guide/inspection_pipeline.md).

## Several Nodal Models

The same workflow can be run on different nodal models. The Solomon example
below uses a thicker value of `gamma` so the plotted graph does not come from a
thin, visually ragged surface.

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import (
    awesome_bloch_vector,
    hopf_link_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)
from knotted_graph.projection import compute_yamada_polynomial

Y = sp.Symbol("Y")
kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

examples = [
    ("Hopf link", hopf_link_bloch_vector, 0.3),
    ("Trefoil", trefoil_bloch_vector, 0.3),
    ("Solomon", solomon_bloch_vector, 0.55),
    ("Awesome", awesome_bloch_vector, 0.16),
]

graphs = {}

for name, model, gamma in examples:
    ske = NodalSkeleton(
        model(gamma, k_symbols=(kx, ky, kz)),
        k_symbols=(kx, ky, kz),
        dimension=48,
    )
    graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
    graphs[name] = graph
    result = compute_yamada_polynomial(
        graph,
        Y,
        num_rotation_samples=12,
        return_result=True,
        n_jobs=1,
    )
    print(
        name,
        (graph.number_of_nodes(), graph.number_of_edges()),
        graph.graph["is_trivalent"],
        result.projection.num_crossings,
        sp.expand(result.polynomial),
    )
```

Output:

```text
Hopf link (2, 3) True 1 -Y**4 - Y**3 - 2*Y**2 - Y - 1
Trefoil (2, 4) False 3 -Y**6 - 2*Y**5 - 5*Y**4 - 5*Y**3 - 5*Y**2 - 2*Y - 1
Solomon (2, 5) False 4 -Y**6 - Y**5 - 3*Y**4 - 2*Y**3 - 3*Y**2 - Y - 1
Awesome (5, 8) False 1 Y**7 - Y**6 + 2*Y**5 - 3*Y**4 + Y**3 - 4*Y**2 - 2
```

Each final expression is displayed in the manuscript convention
$\Upsilon(G;Y)$.

```python
from knotted_graph.visualization import plot_3D_graph_plotly

fig = plot_3D_graph_plotly(graphs["Hopf link"])
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

```{image} ../assets/plot_outputs/nodal_hopf_graph_plotly.png
:alt: Output from plotting the Hopf-link nodal skeleton graph
:width: 70%
:align: center
```

```python
fig = plot_3D_graph_plotly(graphs["Trefoil"])
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

```{image} ../assets/plot_outputs/nodal_trefoil_graph_plotly.png
:alt: Output from plotting the trefoil nodal skeleton graph
:width: 70%
:align: center
```

```python
fig = plot_3D_graph_plotly(graphs["Solomon"])
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

```{image} ../assets/plot_outputs/nodal_solomon_graph_plotly.png
:alt: Output from plotting the thicker Solomon nodal skeleton graph
:width: 70%
:align: center
```

```python
fig = plot_3D_graph_plotly(graphs["Awesome"])
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

```{image} ../assets/plot_outputs/nodal_awesome_graph_plotly.png
:alt: Output from plotting the awesome nodal skeleton graph
:width: 70%
:align: center
```

## Constant-Energy Surface Extraction

The appendix notebooks use the same model constructors to generate
constant-energy surface examples. For the user guide, the important part is the
object pipeline rather than the stitched paper layout: choose a Bloch-vector
model, extract the surface, skeletonize it, and keep both objects available for
inspection.

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import (
    awesome_bloch_vector,
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    threelink_bloch_vector,
    trefoil_bloch_vector,
    unknot_bloch_vector,
)
from knotted_graph.visualization import plot_3D_graph_plotly

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

models = {
    "Unknot": unknot_bloch_vector,
    "Hopf link": hopf_link_bloch_vector,
    "Trefoil": trefoil_bloch_vector,
    "Solomon's knot": solomon_bloch_vector,
    "Three-link": threelink_bloch_vector,
    "Torus (1, 2)": lambda gamma, k_symbols: pq_torus_knot_bloch_vector(
        1, 2, gamma, k_symbols=k_symbols
    ),
    "Awesome knotted graph": awesome_bloch_vector,
}

model_outputs = {}

for name, builder in models.items():
    surface_ske = NodalSkeleton(
        builder(0.2, k_symbols=(kx, ky, kz)),
        k_symbols=(kx, ky, kz),
        dimension=120,
        axis_scale=(1.0, 1.0, 1.5),
    )
    surface = surface_ske.exceptional_surface_pv.connectivity("largest")

    graph_ske = NodalSkeleton(
        builder(0.2, k_symbols=(kx, ky, kz)),
        k_symbols=(kx, ky, kz),
        dimension=120,
        axis_scale=(1.0, 1.0, 1.5),
    )
    graph = graph_ske.skeleton_graph(simplify=True, smooth_epsilon=2)

    print(name, surface.n_points, surface.n_cells)
    print("nodes_edges =", (graph.number_of_nodes(), graph.number_of_edges()))
    model_outputs[name] = (surface, graph)
```

Output from this sweep is a compact checklist of generated objects:

```text
Unknot 7422 14844
nodes_edges = (1, 1)
Hopf link 10228 20464
nodes_edges = (4, 6)
Trefoil 12625 25232
nodes_edges = (8, 12)
Solomon's knot 7176 14352
nodes_edges = (2, 2)
Three-link 15336 30676
nodes_edges = (2, 3)
Torus (1, 2) 12920 25840
nodes_edges = (1, 1)
Awesome knotted graph 16555 33114
nodes_edges = (5, 8)
```

To visualize a selected representative, use the same Plotly graph call shown in
the previous section:

```python
surface, graph = model_outputs["Awesome knotted graph"]
fig = plot_3D_graph_plotly(graph)
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

```{image} ../assets/plot_outputs/nodal_awesome_gamma0p2_graph_plotly.png
:alt: Output from plotting the awesome nodal skeleton graph at gamma equals 0.2
:width: 70%
:align: center
```

## Berry-Curvature Slices

The Berry-curvature flow is another object produced from the same Hamiltonian
and sampled $k$-space grid. For Bloch vectors with two real components and one
purely imaginary component, the package computes

```{math}
\boldsymbol{\Omega}(\mathbf{k})
=
\frac{\gamma}{2(\gamma^2-d_1(\mathbf{k})^2-d_2(\mathbf{k})^2)^{3/2}}
\nabla d_2(\mathbf{k})\times \nabla d_1(\mathbf{k}),
```

up to the component-order sign determined by the Bloch-vector convention. The
useful public workflow is:

```{math}
H(\mathbf{k})
\longrightarrow
\text{exceptional surface}
\longrightarrow
\boldsymbol{\Omega}(\mathbf{k})\text{ flow}
\longrightarrow
\text{oriented skeleton graph}.
```

The Berry-curvature slice objects are generated after the Hopf-link surface has
been extracted. Users can take planar slices of the surface, plot the
intersection curves, and compare those curves with the oriented skeleton.

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import hopf_link_bloch_vector

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

ske = NodalSkeleton(
    hopf_link_bloch_vector(0.8, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=120,
    axis_scale=(1.0, 1.0, 1.5),
)

surface = ske.exceptional_surface_pv.connectivity("largest")
graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)

sections = [
    surface.slice(normal=(1, 0, 0), origin=(0, 0, 0)),
    surface.slice(normal=(0, 1, 0), origin=(0, 0, 0)),
    surface.slice(normal=(0, 0, 1), origin=(0, 0, 0)),
]

print("surface =", (surface.n_points, surface.n_cells))
print("slice_points =", [section.n_points for section in sections])
print("graph =", (graph.number_of_nodes(), graph.number_of_edges()))
```

Output:

```text
surface = (28084, 56172)
slice_points = [787, 704, 0]
graph = (2, 3)
```

Plot the surface together with the slicing planes:

```python
# Notebook helper built from the public surface object and PyVista slices.
fig = plot_surface_with_slice_planes(surface, sections)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_berry_gamma0p8_planes.png
:alt: Output from plotting the Hopf-link exceptional surface with Berry-slice planes
:width: 88%
:align: center
```

Plot the Berry-curvature contours on the selected slices:

```python
berry = ske.berry_curvature
fig = plot_berry_slice_contours(surface, sections, berry)
fig.show()
```

```{image} ../assets/paper_notebook_outputs/appendix_berry_hopf_intersections.png
:alt: Output from plotting Berry-curvature contours on planar surface sections
:width: 92%
:align: center
```

Plot the 3D Berry-curvature flow and the corresponding oriented graph:

```python
plotter = ske.plot_berry_curvature(
    show_surf=True,
    surf_color="#1f77b4",
    surf_opacity=0.12,
    glyph_factor=0.08,
    glyph_tolerance=0.025,
)
oriented_graph = graph.copy()
plot_oriented_spatial_graph(oriented_graph, plotter=plotter)
plotter.show()
```

```{image} ../assets/nodal/field_berry.png
:alt: Output from plotting Berry-curvature flow and the corresponding oriented graph
:width: 88%
:align: center
```

The slice objects are ordinary PyVista datasets, while the final skeleton is the
same embedded `MultiGraph` contract used by projection and Yamada calculation.

## Real-Material Examples

The material examples from the manuscript are now collected in
[Material Fermi-Surface Fingerprints](material_fingerprints.md). That chapter
shows the material name, physical role, rendered Hamiltonian, notebook code,
surface and knotted-graph output, PD-code output, and $\Upsilon(G;Y)$ or
$\Upsilon_{\partial F}$ result for the material cases.
