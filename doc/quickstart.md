# Quick Start

## Crossing-Free Graph Invariants

```python
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

A = sp.Symbol("A")
theta = ThetaGraph(3)
compute_yamada_polynomial_recursive(theta, A)
```

## Spatial Graph to Yamada Polynomial

Embedded graph entry points expect a `networkx.MultiGraph` with 3D node
positions and edge polylines:

```python
import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial

graph = nx.MultiGraph()
graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
graph.add_node("v", pos=np.array([1.0, 0.0, 0.0]))
graph.add_edge(
    "u",
    "v",
    pts=np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.0], [1.0, 0.0, 0.0]]),
)

A = sp.Symbol("A")
result = compute_yamada_polynomial(graph, A, return_result=True)
result.polynomial, result.projection.num_crossings
```

By default, `compute_yamada_polynomial` samples 10 approximately non-isotopic
rotations and uses the valid projection with the fewest crossings. Supplying
`rotation_angles=(...)` bypasses sampling and computes exactly that view.

If the selected planar diagram has 10 or more crossings, the function emits a
`RuntimeWarning` because state-sum evaluation can become expensive.

```{image} ../assets/paper/pd-code-generation.svg
:alt: PD-code generation
:width: 80%
:align: center
```
