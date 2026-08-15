# Mathematical Workflows

KnottedGraph can also be used without geometric input. In this mode, users build
abstract or crossing-free graph families and compute Yamada polynomials as a
tool for pattern discovery.

## Main Graph-Family Examples

The recursive route evaluates crossing-free `networkx.MultiGraph` objects
directly. This is the right interface when the graph is already combinatorial
and no projection or PD-code generation is required.

```python
import sympy as sp
import networkx as nx

from knotted_graph.core import BouquetGraph, ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

Y = sp.Symbol("Y")

examples = {
    "Bouquet_4": BouquetGraph(4),
    "Theta_5": ThetaGraph(5),
    "K_4": nx.MultiGraph(nx.complete_graph(4)),
    "K_{2,3}": nx.MultiGraph(nx.complete_bipartite_graph(2, 3)),
    "Grid_{2,3}": nx.MultiGraph(nx.grid_2d_graph(2, 3)),
}

for name, graph in examples.items():
    upsilon = compute_yamada_polynomial_recursive(graph, Y)
    print(f"Upsilon({name}; Y) = {sp.expand(upsilon)}")
```

For the main mathematical example set used in the notebook, the outputs are:

```text
Upsilon(Bouquet_4; Y) = -Y**4 - 4*Y**3 - 10*Y**2 - 16*Y - 19 - 16/Y - 10/Y**2 - 4/Y**3 - 1/Y**4
Upsilon(Theta_5; Y) = -Y**4 - 3*Y**3 - 8*Y**2 - 11*Y - 14 - 11/Y - 8/Y**2 - 3/Y**3 - 1/Y**4
Upsilon(PeriodicTheta_4; Y) = Y**7 + 2*Y**6 + 13*Y**5 + 18*Y**4 + 60*Y**3 + 64*Y**2 + 125*Y + 97 + 125/Y + 64/Y**2 + 60/Y**3 + 18/Y**4 + 13/Y**5 + 2/Y**6 + Y**(-7)
Upsilon(Sierpinski_{3,2}; Y) = -Y**4 - Y**3 - 4*Y**2 - 3*Y - 6 - 3/Y - 4/Y**2 - 1/Y**3 - 1/Y**4
Upsilon(K_4; Y) = Y**3 + 2*Y + 2/Y + Y**(-3)
Upsilon(K_{2,3}; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
Upsilon(K_{2,1,1}; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
Upsilon(Fan_4; Y) = Y**3 + Y**2 + 3*Y + 2 + 3/Y + Y**(-2) + Y**(-3)
Upsilon(Wheel_4; Y) = -Y**4 - 4*Y**2 - Y - 6 - 1/Y - 4/Y**2 - 1/Y**4
Upsilon(Ladder_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
Upsilon(CircularLadder_3; Y) = -Y**4 + Y**3 - 3*Y**2 + 2*Y - 4 + 2/Y - 3/Y**2 + Y**(-3) - 1/Y**4
Upsilon(Grid_{2,3}; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
Upsilon(Cylinder_{2,3}; Y) = -Y**4 + Y**3 - 3*Y**2 + 2*Y - 4 + 2/Y - 3/Y**2 + Y**(-3) - 1/Y**4
Upsilon(Friendship_2; Y) = -Y**2 - 2*Y - 3 - 2/Y - 1/Y**2
Upsilon(MobiusLadder_3; Y) = -Y**4 + Y**3 - 4*Y**2 + Y - 6 + 1/Y - 4/Y**2 + Y**(-3) - 1/Y**4
```

These are crossing-free graph evaluations. The input is purely combinatorial,
so the result does not depend on a projection choice.

## Full Structured Dataset

The earlier math notebook also produced a systematic CSV sweep. Download it
here:
{download}`structured_graph_yamada_dataset.csv <../assets/data/structured_graph_yamada_dataset.csv>`.

Use the CSV loader when you want every row in manuscript notation:

```python
import csv
from pathlib import Path

dataset_path = Path("structured_graph_yamada_dataset.csv")
if not dataset_path.exists():
    dataset_path = Path("doc/assets/data/structured_graph_yamada_dataset.csv")

with dataset_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        name = row["graph_name"]
        params = row["varying_params"]
        print(f"Upsilon({name}{params}; Y) = {row['yamada']}")
```

The complete stored dataset is:

```{literalinclude} ../assets/data/structured_graph_yamada_dataset.csv
:language: text
:caption: Full structured graph Yamada dataset
```

## Nodal-Model Yamada Rows

The same invariant call can be used after a geometric model has been converted
to a spatial graph. In documentation pages, show the row values directly rather
than hiding them inside a preassembled paper panel: the user should see which
graph was extracted, how many crossings the selected projection has, and the
resulting $\Upsilon(G;Y)$.

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

rows = []
for name, model, gamma in examples:
    ske = NodalSkeleton(
        model(gamma, k_symbols=(kx, ky, kz)),
        k_symbols=(kx, ky, kz),
        dimension=48,
    )
    graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
    result = compute_yamada_polynomial(
        graph,
        Y,
        num_rotation_samples=12,
        return_result=True,
        n_jobs=1,
    )
    rows.append((name, graph, result.projection, sp.expand(result.polynomial)))

for name, graph, projection, upsilon in rows:
    print(name)
    print("nodes_edges =", (graph.number_of_nodes(), graph.number_of_edges()))
    print("crossings =", projection.num_crossings)
    print(f"Upsilon(G; Y) = {upsilon}")
```

Typical output:

```text
Hopf link
nodes_edges = (2, 3)
crossings = 1
Upsilon(G; Y) = -Y**4 - Y**3 - 2*Y**2 - Y - 1
Trefoil
nodes_edges = (2, 4)
crossings = 3
Upsilon(G; Y) = -Y**6 - 2*Y**5 - 5*Y**4 - 5*Y**3 - 5*Y**2 - 2*Y - 1
Solomon
nodes_edges = (2, 5)
crossings = 4
Upsilon(G; Y) = -Y**6 - Y**5 - 3*Y**4 - 2*Y**3 - 3*Y**2 - Y - 1
Awesome
nodes_edges = (5, 8)
crossings = 1
Upsilon(G; Y) = Y**7 - Y**6 + 2*Y**5 - 3*Y**4 + Y**3 - 4*Y**2 - 2
```

The graph plots for these same examples are shown once in
[Non-Hermitian Nodal Skeletons](../applications/nodal_skeleton.md). This page
keeps the focus on the mathematical output.

## Intrinsic Linkedness From Extracted Graphs

Intrinsic linkedness is a downstream graph-theory question after KnottedGraph
has produced a spatial graph. The library-relevant part is the extraction of
that graph from a surface model; abstract minor searches or graph-family
theorems can then be applied with specialized graph tools.

```python
import networkx as nx
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import awesome_bloch_vector

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

ske = NodalSkeleton(
    awesome_bloch_vector(0.16, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=48,
)

surface = ske.exceptional_surface_pv.connectivity("largest")
graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
simple_graph = nx.Graph(graph)

print("surface_points_cells =", (surface.n_points, surface.n_cells))
print("nodes_edges =", (graph.number_of_nodes(), graph.number_of_edges()))
print("is_planar =", nx.check_planarity(simple_graph)[0])
print("degree_sequence =", sorted(dict(graph.degree()).values()))
```

Output:

```text
surface_points_cells = (11630, 23256)
nodes_edges = (5, 8)
is_planar = True
degree_sequence = [3, 3, 3, 3, 4]
```

First inspect the surface that produced the graph:

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
:alt: Awesome-model surface used before intrinsic linkedness inspection
:width: 68%
:align: center
```

Then inspect the spatial graph extracted from that same surface. The
graph-theory certificate is a downstream mathematical check, so the separate
Petersen/certificate drawings are not repeated here.

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

```{image} ../assets/plot_outputs/nodal_awesome_graph_plotly.png
:alt: Plotly spatial graph extracted from the awesome-model surface
:width: 70%
:align: center
```

## Planarity From Extracted Graphs

Planarity is another downstream inspection step. The useful tutorial pattern is
to show the surface, then the spatial graph extracted from that exact surface,
then the graph-theory result.

```python
import networkx as nx
import plotly.graph_objects as go
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import threelink_bloch_vector
from knotted_graph.visualization import plot_3D_graph_plotly

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

planarity_runs = {}
for gamma in (0.116, 0.41, 0.5):
    ske = NodalSkeleton(
        threelink_bloch_vector(gamma, k_symbols=(kx, ky, kz)),
        k_symbols=(kx, ky, kz),
        dimension=100,
        axis_scale=(1.0, 1.0, 1.5),
    )
    surface = ske.exceptional_surface_pv.connectivity("largest")
    graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
    is_planar = nx.check_planarity(nx.Graph(graph))[0]
    planarity_runs[gamma] = (surface, graph, is_planar)
    print(
        f"gamma={gamma}",
        "surface=", (surface.n_points, surface.n_cells),
        "graph=", (graph.number_of_nodes(), graph.number_of_edges()),
        "is_planar=", is_planar,
    )
```

Output:

```text
gamma=0.116 surface= (8416, 16840) graph= (3, 5) is_planar= True
gamma=0.41 surface= (15476, 30964) graph= (6, 9) is_planar= False
gamma=0.5 surface= (16464, 32932) graph= (2, 3) is_planar= True
```

For $\gamma=0.116$, first inspect the surface:

```python
surface, graph, is_planar = planarity_runs[0.116]

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
:alt: Three-link surface at gamma equals 0.116
:width: 70%
:align: center
```

Then plot the spatial graph extracted from that same surface:

```python
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

```{image} ../assets/plot_outputs/planarity_threelink_graph_gamma_0p116_plotly.png
:alt: Spatial graph extracted from the three-link surface at gamma equals 0.116
:width: 70%
:align: center
```

For $\gamma=0.41$, the same library calls produce a different surface:

```python
surface, graph, is_planar = planarity_runs[0.41]

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
:alt: Three-link surface at gamma equals 0.41
:width: 70%
:align: center
```

and the extracted graph is the non-planar case in the printed diagnostics:

```python
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

```{image} ../assets/plot_outputs/planarity_threelink_graph_gamma_0p41_plotly.png
:alt: Spatial graph extracted from the three-link surface at gamma equals 0.41
:width: 70%
:align: center
```

For $\gamma=0.5$, inspect the thicker surface:

```python
surface, graph, is_planar = planarity_runs[0.5]

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
:alt: Three-link surface at gamma equals 0.5
:width: 70%
:align: center
```

```python
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

```{image} ../assets/plot_outputs/planarity_threelink_graph_gamma_0p5_plotly.png
:alt: Spatial graph extracted from the three-link surface at gamma equals 0.5
:width: 70%
:align: center
```

## Custom NetworkX Graphs

Any `networkx.MultiGraph` can be used for crossing-free recursive evaluation.
Parallel edges and loops are allowed.

```python
import networkx as nx
import sympy as sp

from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

G = nx.MultiGraph()
G.add_edges_from([(0, 1), (0, 1), (0, 1)])

Y = sp.Symbol("Y")
upsilon = compute_yamada_polynomial_recursive(G, Y)
print(f"Upsilon(G; Y) = {sp.expand(upsilon)}")
```

Example output:

```text
Upsilon(G; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
```

This custom graph is another theta graph: two vertices connected by three
parallel edges. Matching the built-in result is a quick check that the graph
family was constructed as intended.

## Pattern Discovery

This mode is useful for:

- testing closed forms for graph families;
- searching for recurrence relations;
- studying degree growth and coefficient patterns;
- comparing recursive and Negami evaluation routes;
- generating computational evidence for conjectures.

For embedded graph families, use the projection tools first, then inspect the
selected projection and Yamada polynomial with `return_result=True`.
