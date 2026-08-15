# Quick Start

This page starts with installation and then shows the shortest path through
KnottedGraph: start from a
two-dimensional surface embedded in 3D, pass to a corresponding spatial graph,
obtain a planar projection and PD code, and compute a Yamada invariant. For
detailed options and methodology, follow the linked User Guide chapters.

## Installation

The core package requires Python 3.11 or newer.

```bash
pip install knotted_graph
```

For development from a checkout:

```bash
git clone https://github.com/sarinstein-yan/KnottedGraph.git
cd KnottedGraph
uv sync --all-groups
```

Install optional extras only for the workflows you need:

```bash
pip install "knotted_graph[nodal]"
pip install "knotted_graph[surface]"
pip install "knotted_graph[repulsion]"
pip install "knotted_graph[all]"
```

The nodal-skeleton application uses optional packages such as PyVista,
scikit-image, poly2graph, and minorminer. The generic graph, projection, and
Yamada APIs do not import those optional stacks.

To build this documentation locally from a checkout:

```bash
uv run --group docs python -m sphinx -b html doc site_preview
```

The central pipeline is

```{math}
\text{surface}\longrightarrow G\subset\mathbb{R}^3
\longrightarrow D(G)\longrightarrow \operatorname{PD}(G)
\longrightarrow \Upsilon(G;Y).
```

## 1. Start From A 2D Surface

This quick example uses a synthetic tube surface whose spine is a knotted
trivalent $K_4$-type spatial graph. This is intentionally a little richer than
a toy loop: it has four real branch vertices, six embedded edges, and every
vertex has degree three. A three-vertex graph with only three edges would be a
cycle, so its vertices would have degree two rather than being real branch
vertices. In application workflows, surfaces can instead come from meshes,
volumes, Hamiltonians, biomolecules, or simulations; see
[Workflow Overview](user_guide/workflow_overview.md) and
[Input Adapters](user_guide/input_adapters.md) for the supported entry points.

```python
import networkx as nx
import numpy as np


def trivalent_k4_spine(samples=90, amplitude=0.75):
    vertices = {
        "a": np.array([-1.15, -0.78, -0.38]),
        "b": np.array([1.18, -0.64, 0.30]),
        "c": np.array([0.86, 0.95, -0.26]),
        "d": np.array([-0.88, 0.84, 0.52]),
    }
    edge_specs = [
        ("a", "b", "ab", np.array([0.00, 0.90, 0.70]), 0.0),
        ("a", "c", "ac", np.array([0.35, -0.15, 1.00]), 1.1),
        ("a", "d", "ad", np.array([0.95, 0.15, -0.25]), 2.2),
        ("b", "c", "bc", np.array([-0.90, 0.25, 0.35]), 0.7),
        ("b", "d", "bd", np.array([-0.20, 1.00, -0.60]), 1.7),
        ("c", "d", "cd", np.array([0.10, -0.90, -0.85]), 2.8),
    ]

    s = np.linspace(0.0, 1.0, samples)
    graph = nx.MultiGraph()
    for vertex_id, pos in vertices.items():
        graph.add_node(vertex_id, pos=pos)

    for u, v, key, bend, phase in edge_specs:
        start = vertices[u]
        end = vertices[v]
        chord = end - start
        bend = bend / np.linalg.norm(bend)
        side = np.cross(chord, bend)
        side = side / np.linalg.norm(side)
        envelope = np.sin(np.pi * s)
        pts = (1 - s)[:, None] * start + s[:, None] * end
        pts += amplitude * envelope[:, None] * (
            np.cos(phase + np.pi * s)[:, None] * bend
            + 0.6 * np.sin(2 * np.pi * s + phase)[:, None] * side
        )
        pts[0] = start
        pts[-1] = end
        graph.add_edge(u, v, key=key, pts=pts)

    graph.graph.update(
        graph_id="quickstart_trivalent_k4_surface_spine",
        input_kind="synthetic_surface_spine",
        is_closed=True,
    )
    return graph


def tube_patch(points, radius=0.11, sides=28):
    tangents = np.gradient(points, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    nearly_parallel = np.abs(np.sum(tangents * reference, axis=1)) > 0.92
    reference[nearly_parallel] = np.array([0.0, 1.0, 0.0])
    normals = np.cross(tangents, reference)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    binormals = np.cross(tangents, normals)

    theta = np.linspace(0.0, 2 * np.pi, sides, endpoint=True)
    circle = (
        np.cos(theta)[None, :, None] * normals[:, None, :]
        + np.sin(theta)[None, :, None] * binormals[:, None, :]
    )
    tube = points[:, None, :] + radius * circle
    return {"x": tube[:, :, 0], "y": tube[:, :, 1], "z": tube[:, :, 2]}


def sphere_patch(center, radius=0.18, samples=28):
    phi = np.linspace(0.0, np.pi, samples)
    theta = np.linspace(0.0, 2 * np.pi, samples)
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    return {
        "x": center[0] + radius * np.sin(phi) * np.cos(theta),
        "y": center[1] + radius * np.sin(phi) * np.sin(theta),
        "z": center[2] + radius * np.cos(phi),
    }


def trivalent_k4_surface_graph(tube_radius=0.12):
    graph = trivalent_k4_spine()
    tube_surfaces = [
        tube_patch(data["pts"], radius=tube_radius)
        for _, _, data in graph.edges(data=True)
    ]
    vertex_surfaces = [
        sphere_patch(data["pos"], radius=1.45 * tube_radius)
        for _, data in graph.nodes(data=True)
    ]
    return tube_surfaces, vertex_surfaces, graph


tube_surfaces, vertex_surfaces, graph = trivalent_k4_surface_graph()

print(f"tube_surface_patches = {len(tube_surfaces)}")
print(f"vertex_surface_patches = {len(vertex_surfaces)}")
print(f"nodes_edges = {(graph.number_of_nodes(), graph.number_of_edges())}")
print(f"degrees = {dict(graph.degree())}")
print(f"vertex_ids = {list(graph.nodes)}")
```

Output:

```text
tube_surface_patches = 6
vertex_surface_patches = 4
nodes_edges = (4, 6)
degrees = {'a': 3, 'b': 3, 'c': 3, 'd': 3}
vertex_ids = ['a', 'b', 'c', 'd']
```

Plot the surface:

```python
import plotly.graph_objects as go

fig = go.Figure()
for patch in tube_surfaces:
    fig.add_trace(
        go.Surface(
            x=patch["x"],
            y=patch["y"],
            z=patch["z"],
            surfacecolor=np.zeros_like(patch["x"]),
            colorscale=[[0, "#1f77b4"], [1, "#1f77b4"]],
            showscale=False,
            opacity=0.42,
        )
    )
for patch in vertex_surfaces:
    fig.add_trace(
        go.Surface(
            x=patch["x"],
            y=patch["y"],
            z=patch["z"],
            surfacecolor=np.zeros_like(patch["x"]),
            colorscale=[[0, "#d62728"], [1, "#d62728"]],
            showscale=False,
            opacity=0.92,
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

```{image} assets/plot_outputs/quickstart_surface_plotly.png
:alt: Output from plotting the quick-start 2D tube surface
:width: 72%
:align: center
```

This is the input geometry. The blue tubes form the surface neighborhood of
the embedded spine, and the red balls mark the four degree-three branch
vertices. The next step inspects the corresponding spatial graph object used
by the rest of the package.

## 2. Inspect The Surface Spine As A Spatial Graph

```python
print(f"nodes_edges = {(graph.number_of_nodes(), graph.number_of_edges())}")
print(f"closed = {graph.graph['is_closed']}")
print(f"degrees = {dict(graph.degree())}")
print("vertices:")
for node, data in graph.nodes(data=True):
    coords = tuple(round(c, 3) for c in data["pos"])
    print(f"  {node}: {coords}")

print("edges:")
for source, target, key, data in graph.edges(keys=True, data=True):
    print(f"  {key}: {source} -> {target}, points={data['pts'].shape}")
```

Output:

```text
nodes_edges = (4, 6)
closed = True
degrees = {'a': 3, 'b': 3, 'c': 3, 'd': 3}
vertices:
  a: (-1.15, -0.78, -0.38)
  b: (1.18, -0.64, 0.3)
  c: (0.86, 0.95, -0.26)
  d: (-0.88, 0.84, 0.52)
edges:
  ab: a -> b, points=(90, 3)
  ac: a -> c, points=(90, 3)
  ad: a -> d, points=(90, 3)
  bc: b -> c, points=(90, 3)
  bd: b -> d, points=(90, 3)
  cd: c -> d, points=(90, 3)
```

Plot the corresponding spatial graph:

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

```{image} assets/plot_outputs/quickstart_surface_spine_graph_plotly.png
:alt: Output from plotting the quick-start surface spine graph
:width: 72%
:align: center
```

This plot checks the first transition:

```{math}
\text{surface}\longrightarrow G\subset\mathbb{R}^3.
```

For inspecting automatically extracted surfaces, skeleton points, raw graphs,
simplified graphs, edge polylines, and graph metadata, see
[Inspecting Intermediate Objects](user_guide/inspection_pipeline.md). For a
full surface-to-skeleton application, see
[Non-Hermitian Nodal Skeletons](applications/nodal_skeleton.md).

## 3. Obtain And Plot A Planar Projection

Spatial-graph invariants are evaluated from a planar diagram. For the first
example, use an explicit view with separated crossings. If `rotation_angles` is
omitted, `select_projection` samples viewing directions and returns a valid
projection, favoring one with fewer crossings.

```python
from knotted_graph.projection import select_projection

projection = select_projection(graph, rotation_angles=(0.0, 0.0, 0.0))

print(f"rotation_angles = {tuple(round(a, 2) for a in projection.rotation_angles)}")
print(f"crossings = {projection.num_crossings}")
print(f"pd_code = {projection.pd_code}")
```

Output:

```text
rotation_angles = (0.0, 0.0, 0.0)
crossings = 5
pd_code = V[2,0,7];V[1,9,10];V[6,9,13];V[12,8,15];X[11,8,12,7];X[14,5,15,4];X[1,3,0,2];X[10,4,11,3];X[14,6,13,5]
```

Plot the selected planar projection $D(G)$:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.8, 4.0))
for arc in projection.arcs:
    ax.plot(*arc.line.xy, color="#1f77b4", linewidth=2.4)
for vertex in projection.vertices:
    ax.scatter(*vertex.point.xy, color="#d62728", s=64, zorder=3)
for crossing in projection.crossings:
    ax.scatter(*crossing.point.xy, marker="x", color="black", s=72, zorder=4)

ax.set_aspect("equal")
ax.axis("off")
plt.show()
```

```{image} assets/plot_outputs/quickstart_surface_projection.png
:alt: Output from plotting the quick-start selected planar projection
:width: 52%
:align: center
```

This is the next transition:

```{math}
G\subset\mathbb{R}^3\longrightarrow D(G)\longrightarrow \operatorname{PD}(G).
```

See [Projection, PD Codes, and Yamada Polynomials](user_guide/projection_yamada.md)
for projection sampling, crossing detection, PD-code construction, and advanced
controls.

## 4. Compute The Yamada Polynomial

```python
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial

Y = sp.Symbol("Y")
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=projection.rotation_angles,
    return_result=True,
)

print(f"Upsilon(G; Y) = {sp.expand(result.polynomial)}")
print(f"crossings = {result.projection.num_crossings}")
```

Output:

```text
Upsilon(G; Y) = -Y**6 - 2*Y**4 - 2*Y**2 - 1
crossings = 5
```

`result.polynomial` is the Yamada invariant of the embedded spatial graph, and
`result.projection` retains the planar diagram used in the calculation. In the
paper notation, this output is
$\Upsilon(G;Y)=-Y^6-2Y^4-2Y^2-1$.

See [Projection, PD Codes, and Yamada Polynomials](user_guide/projection_yamada.md)
for normalization, evaluation algorithms, projection dependence, and
computational considerations.

## 5. Complete Pipeline

Once the surface has produced a spatial graph spine, the full computation is:

```python
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial, select_projection

Y = sp.Symbol("Y")

tube_surfaces, vertex_surfaces, graph = trivalent_k4_surface_graph()

projection = select_projection(graph, rotation_angles=(0.0, 0.0, 0.0))
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=projection.rotation_angles,
    return_result=True,
)

print(result.projection.pd_code)
print(sp.expand(result.polynomial))
```

Conceptually:

```{math}
\text{surface}\to G\subset\mathbb{R}^3
\to D(G)\to \operatorname{PD}(G)\to \Upsilon(G;Y).
```

## Where To Go Next

| If you want to... | Go to |
| --- | --- |
| understand the main object contract | [Workflow Overview](user_guide/workflow_overview.md) |
| understand supported surface, volume, coordinate, biomolecular, and data inputs | [Input Adapters](user_guide/input_adapters.md) |
| inspect surfaces, skeletons, graphs, projections, and metadata | [Inspecting Intermediate Objects](user_guide/inspection_pipeline.md) |
| understand projection, PD codes, and Yamada computation | [Projection, PD Codes, and Yamada Polynomials](user_guide/projection_yamada.md) |
| simplify or relax difficult embedded graphs | [Repulsive Layout](user_guide/repulsive_layout.md) |
| apply the pipeline to non-Hermitian systems | [Non-Hermitian Nodal Skeletons](applications/nodal_skeleton.md) |
| analyze material Fermi surfaces | [Material Fermi-Surface Fingerprints](applications/material_fingerprints.md) |
| work with protein-derived graphs | [Protein-Derived Theta Graphs](applications/biomolecular_protein_workflow.md) |
| use Yamada invariants for graph-family mathematics | [Mathematical Workflows](applications/mathematical_workflows.md) |
| find exact arguments and callable interfaces | [API Reference](api/index.md) |
