# Workflow Overview

KnottedGraph is organized around a small number of reusable representations.
Users can enter the workflow from coordinate data, biomolecular files, polymer
simulation outputs, surface meshes, already embedded spatial graphs, planar
diagrams, or abstract graph families.

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

For many users, the most direct path is:

```python
import numpy as np
import sympy as sp

from knotted_graph.inputs import from_coordinate_chain
from knotted_graph.projection import compute_yamada_polynomial

t = np.linspace(0.0, 2 * np.pi, 240, endpoint=False)
coords = np.column_stack(
    [
        (1.65 + 0.85 * np.cos(3 * t)) * np.cos(2 * t),
        (1.65 + 0.85 * np.cos(3 * t)) * np.sin(2 * t),
        0.85 * np.sin(3 * t),
    ]
)

parsed = from_coordinate_chain(
    coords,
    closed=True,
    closure="direct",
    input_id="trefoil_curve",
)

Y = sp.Symbol("Y")
upsilon = compute_yamada_polynomial(parsed.graph, Y)
print(f"Upsilon(G; Y) = {sp.expand(upsilon)}")
```

Output:

```text
Upsilon(G; Y) = -Y**11 + Y**9 + Y**8 + Y**7 - Y**4 - Y**3 - Y**2 - Y - 1
```

For this closed trefoil-like coordinate chain, the parsed object is easy to
inspect:

```python
parsed.input_id, parsed.source_format, parsed.closed, parsed.issues
parsed.graph.number_of_nodes(), parsed.graph.number_of_edges()
```

Example output:

```text
('trefoil_curve', 'array', True, [])
(1, 1)
```

This means the input was accepted as an array, no validation issues were found,
and the closed curve became one self-loop edge attached to one anchor node.

The input-adapter chapter shows the corresponding plot for this coordinate
chain. The overview keeps only the end-to-end computation and object contract.

For application workflows, import the application explicitly:

```python
from knotted_graph.applications.nodal import NodalSkeleton
```

The root import, `import knotted_graph`, intentionally avoids loading optional
application stacks such as PyVista, Plotly, scikit-image, and poly2graph.
