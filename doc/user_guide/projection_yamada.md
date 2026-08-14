# Projection, PD Codes, and Yamada Polynomials

The projection layer converts an embedded spatial graph into a planar diagram,
records the diagram as a PD code, and passes that symbolic representation to the
Yamada engine. In these docs, Yamada-polynomial outputs are written in the
manuscript notation $\Upsilon(G;Y)$.

The projection API works on any `networkx.MultiGraph` whose nodes have `pos`
coordinates and whose edges have `pts` polylines. To avoid a toy straight-line
example, this chapter uses a compact trefoil-derived spatial graph and
subdivides two edges so the planar diagram has four graph vertices.

```python
import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import trefoil_bloch_vector


def subdivide_n_longest_edges(graph, n=2):
    out = nx.MultiGraph()
    for node, attrs in graph.nodes(data=True):
        out.add_node(node, **attrs)

    edges = list(graph.edges(keys=True, data=True))
    selected = sorted(
        edges,
        key=lambda edge: len(edge[3].get("pts", [])),
        reverse=True,
    )[:n]
    selected_ids = {(u, v, key) for u, v, key, _ in selected}
    selected_ids |= {(v, u, key) for u, v, key, _ in selected}

    new_index = 0
    for a, b, key, attrs in edges:
        if (a, b, key) in selected_ids:
            pts = np.asarray(attrs["pts"])
            midpoint = len(pts) // 2
            new_node = f"w{new_index}"
            new_index += 1
            out.add_node(new_node, pos=pts[midpoint])

            first = dict(attrs)
            second = dict(attrs)
            first["pts"] = pts[: midpoint + 1]
            second["pts"] = pts[midpoint:]
            out.add_edge(a, new_node, key=f"{key}a", **first)
            out.add_edge(new_node, b, key=f"{key}b", **second)
        else:
            out.add_edge(a, b, key=key, **attrs)

    return out


kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
ske = NodalSkeleton(
    trefoil_bloch_vector(0.3, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=48,
)

base_graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
graph = subdivide_n_longest_edges(base_graph, n=2)
print(graph.number_of_nodes(), graph.number_of_edges())
```

Example output:

```text
4 6
```

Plot the embedded graph before choosing a projection:

```python
from knotted_graph.visualization import plot_3D_graph_plotly

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

```{image} ../assets/plot_outputs/subdivided_trefoil_spatial_graph_plotly.png
:alt: Output from plotting the embedded subdivided trefoil-derived graph
:width: 70%
:align: center
```

## Explicit PD-Code Generation

Use `PDCode` directly when you want to inspect vertices, crossings, and arcs
after computation.

```python
from knotted_graph.projection import PDCode

rotation_angles = (-149.91, 38.62, 0.0)

pd = PDCode(graph)
code = pd.compute(rotation_angles=rotation_angles)

print(f"pd_terms = {sorted(code.split(';'))}")
print(f"vertices = {len(pd.vertices)}")
print(f"crossings = {len(pd.crossings)}")
print(f"arcs = {len(pd.arcs)}")
```

Example output:

```text
pd_terms = ['V[10,3]', 'V[15,5]', 'V[7,11,9,10]', 'V[8,0,6,4]', 'X[0,12,1,13]', 'X[1,12,2,11]', 'X[2,8,3,9]', 'X[4,14,5,15]', 'X[6,13,7,14]']
vertices = 4
crossings = 5
arcs = 16
```

The four `V[...]` entries are graph vertices. The five `X[...]` entries are
crossings in the chosen planar projection. The arcs are the planar pieces
obtained by cutting graph edges at crossings.

Plot the same planar diagram from the `PDCode` object:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
for arc in pd.arcs.values():
    ax.plot(*arc.line.xy, color="tab:blue", linewidth=2)
for vertex in pd.vertices.values():
    ax.scatter(*vertex.point.xy, color="tab:red", zorder=3)
for crossing in pd.crossings.values():
    ax.scatter(*crossing.point.xy, marker="x", color="black", zorder=4)

ax.set_aspect("equal")
ax.axis("off")
plt.show()
```

```{image} ../assets/plot_outputs/subdivided_trefoil_selected_projection.png
:alt: Output from plotting the selected five-crossing planar diagram
:width: 55%
:align: center
```

## Projection Sampling

If no explicit projection is supplied, the high-level Yamada function samples
approximately non-isotopic rotations and selects a valid projection. Sampling is
also useful when users want to inspect several diagrams before choosing one for
a figure.

```python
from knotted_graph.projection import sample_projections, select_projection

all_views = sample_projections(graph, num_rotation_samples=16)
best_view = select_projection(graph, num_rotation_samples=16)

print([p.num_crossings for p in all_views])
print(tuple(round(a, 2) for a in best_view.rotation_angles))
print(f"selected_crossings = {best_view.num_crossings}")
```

Example output:

```text
[4, 4, 4, 5, 5, 5, 5, 4, 3, 3, 3, 3, 5, 3, 3, 3]
(20.06, 57.91, 0.0)
selected_crossings = 3
```

The list reports the number of crossings in each sampled view. For computation,
the selected view is often one of the lower-crossing projections. For
exposition, a user may deliberately choose a higher-crossing sampled view, as
we did above, to display a richer planar diagram.

## Yamada From A Spatial Graph

```python
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial

Y = sp.Symbol("Y")
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=(-149.91, 38.62, 0.0),
    method="negami",
    normalize=True,
    return_result=True,
    n_jobs=1,
)

print(f"Upsilon(G; Y) = {sp.expand(result.polynomial)}")
print(f"crossings = {result.projection.num_crossings}")
print(f"pd_terms = {sorted(result.projection.pd_code.split(';'))}")
```

Example output:

```text
Upsilon(G; Y) = -Y**6 - 2*Y**5 - 5*Y**4 - 5*Y**3 - 5*Y**2 - 2*Y - 1
crossings = 5
pd_terms = ['V[10,3]', 'V[15,5]', 'V[7,11,9,10]', 'V[8,0,6,4]', 'X[0,12,1,13]', 'X[1,12,2,11]', 'X[2,8,3,9]', 'X[4,14,5,15]', 'X[6,13,7,14]']
```

The first line is the normalized $\Upsilon(G;Y)$. The projection fields record
the crossing count and PD terms used to compute it, which makes the calculation
reproducible from the selected planar diagram.

`method="negami"` evaluates resolved crossing-free graphs through the Negami
state sum. `method="recursive"` uses recursive deletion-contraction. Both routes
share the same crossing-resolution stage.

## Crossing-Free Graphs

For abstract crossing-free graphs, the Yamada engine can be called directly.

```python
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

Y = sp.Symbol("Y")
theta = ThetaGraph(3)
upsilon = compute_yamada_polynomial_recursive(theta, Y)
print(f"Upsilon(Theta_3; Y) = {sp.expand(upsilon)}")
```

Expanded output:

```text
Upsilon(Theta_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
```

This direct route evaluates a crossing-free abstract theta graph. It is useful
when the user is studying graph families rather than projections of embedded
spatial data.
