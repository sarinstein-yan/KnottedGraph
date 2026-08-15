# Workflow Overview

KnottedGraph is organized around a small number of reusable representations.
Users can enter the workflow from coordinate data, biomolecular files, polymer
simulation outputs, surface meshes, already embedded spatial graphs, planar
diagrams, or abstract graph families.

When explaining geometric input to new users, use the surface-first order:

\[
\text{2D surface in }\mathbb{R}^3
\longrightarrow
G\subset\mathbb{R}^3
\longrightarrow
D(G)
\longrightarrow
\operatorname{PD}(G)
\longrightarrow
\Upsilon(G;Y).
\]

This is the order used in the [Quick Start](../quickstart.md): first a
two-dimensional surface is displayed, then its spatial-graph spine is plotted,
then the selected planar projection, PD code, and Yamada polynomial are shown.

## Main Stages

| Workflow stage | Package location | Typical object |
| --- | --- | --- |
| Input adapters | `knotted_graph.inputs` | parsed input result with a graph or mesh |
| Skeleton extraction | `knotted_graph.extraction` and applications | skeleton image or graph |
| Graph cleaning | `knotted_graph.core` | embedded `networkx.MultiGraph` |
| Repulsive layout | `knotted_graph.layout.repulsive` | relaxed embedded graph |
| Projection and PD encoding | `knotted_graph.projection` | `ProjectionResult`, `PDCode` |
| Yamada invariant | `knotted_graph.invariants.yamada` | SymPy Laurent polynomial |
| Visualization | `knotted_graph.visualization` and applications | Matplotlib, Plotly, or PyVista figures |

The common exchange object is an embedded `networkx.MultiGraph`. Each node has a
finite 3D `pos` coordinate, and each edge may carry a `pts` polyline describing
its embedded geometry.

```python
from knotted_graph.core import ensure_embedding

graph = ensure_embedding(graph, copy=True, normalize=True)
```

## High-Level Use

For many users, the conceptual path is to begin with a surface or surface-like
geometry, extract or choose a graph spine, and then run the generic graph
pipeline. The full plotted example is in the Quick Start; in compact form, the
computation looks like this:

```python
import sympy as sp

from knotted_graph.core import ensure_embedding
from knotted_graph.projection import compute_yamada_polynomial, select_projection

# The Quick Start defines this helper and plots both objects:
# surface: a 2D tube surface embedded in R^3
# graph: the corresponding trivalent spatial-graph spine
tube_surfaces, vertex_surfaces, graph = trivalent_k4_surface_graph()
graph = ensure_embedding(graph, copy=True, normalize=False)

projection = select_projection(graph, rotation_angles=(0.0, 0.0, 0.0))

Y = sp.Symbol("Y")
upsilon = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=projection.rotation_angles,
)
print(f"Upsilon(G; Y) = {sp.expand(upsilon)}")
print(f"nodes_edges = {(graph.number_of_nodes(), graph.number_of_edges())}")
print(f"degrees = {dict(graph.degree())}")
```

Output:

```text
Upsilon(G; Y) = -Y**6 - 2*Y**4 - 2*Y**2 - 1
nodes_edges = (4, 6)
degrees = {'a': 3, 'b': 3, 'c': 3, 'd': 3}
```

For this surface spine, the graph object is easy to inspect:

```python
graph.graph["graph_id"], graph.graph["input_kind"], graph.graph["is_closed"]
graph.number_of_nodes(), graph.number_of_edges()
```

Example output:

```text
('quickstart_trivalent_k4_surface_spine', 'synthetic_surface_spine', True)
(4, 6)
```

This means the surface-spine object is an embedded spatial graph with four real
branch vertices and six edges.

The overview keeps the object contract short. The Quick Start shows the plotted
surface, plotted spatial graph, selected planar projection, printed PD code, and
printed $\Upsilon(G;Y)$ result; the later chapters then explain each reusable
piece in detail.

For application workflows, import the application explicitly:

```python
from knotted_graph.applications.nodal import NodalSkeleton
```

The root import, `import knotted_graph`, intentionally avoids loading optional
application stacks such as PyVista, Plotly, scikit-image, and poly2graph.
