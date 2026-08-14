# Non-Hermitian Nodal Skeletons

The nodal-skeleton workflow is the original physics application that motivated
this package. It samples a two-band Hamiltonian in 3D momentum space, extracts
the exceptional-surface interior, skeletonizes that region, and returns a
spatial multigraph that can be projected to a planar diagram.

The word "exceptional" is specific to this non-Hermitian application. In the
generic library, surfaces can also come from imported meshes or other external
workflows; once a spatial graph is available, projection and Yamada computation
use the same API.

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
stage-by-stage plots of the exceptional surface, skeleton points, raw graph,
simplified graph, and planar diagram are shown in
[Inspecting Intermediate Objects](../user_guide/inspection_pipeline.md).

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
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
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
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
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
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
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
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
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
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.75, z=1.05)),
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

The Berry-curvature slice objects are generated after the Hopf-link surface has
been extracted. The useful public pattern is to take planar slices of the surface,
plot the intersection curves, and compare those curves with the skeleton.

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

The slice objects are ordinary PyVista datasets, so a user can plot them with
their preferred PyVista or Plotly routine. The reusable package state is the
surface, the slice list, and the extracted graph.

## Real-Material Examples

The material examples from the manuscript are now collected in
[Material Fermi-Surface Fingerprints](material_fingerprints.md). That chapter
shows the material name, physical role, rendered Hamiltonian, notebook code,
surface and knotted-graph output, PD-code output, and $\Upsilon(G;Y)$ or
$\Upsilon_{\partial F}$ result for the material cases.
