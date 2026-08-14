# Quick Start

## Crossing-Free Graph Invariants

```python
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

Y = sp.Symbol("Y")
theta = ThetaGraph(3)
upsilon = compute_yamada_polynomial_recursive(theta, Y)
print(f"Upsilon(Theta_3; Y) = {sp.expand(upsilon)}")
```

Output:

```text
Upsilon(Theta_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
```

This is the crossing-free value $\Upsilon(\Theta_3;Y)$.

## Spatial Graph to Yamada Polynomial

Embedded graph entry points expect a `networkx.MultiGraph` with 3D node
positions and edge polylines. This example uses a braided theta graph, so the
projection has actual crossings to inspect.

```python
import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial


def braided_theta_graph(samples=160, turns=0.7, amp=0.95):
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([3.0, 0.0, 0.0]))

    s = np.linspace(0.0, 1.0, samples)
    envelope = np.sin(np.pi * s)
    phases = {
        "alpha": 0.0,
        "beta": 2 * np.pi / 3,
        "gamma": 4 * np.pi / 3,
    }

    for key, phase in phases.items():
        angle = 2 * np.pi * turns * s + phase
        pts = np.column_stack(
            [
                3.0 * s,
                amp * envelope * np.cos(angle),
                amp * envelope * np.sin(angle),
            ]
        )
        pts[0] = graph.nodes["u"]["pos"]
        pts[-1] = graph.nodes["v"]["pos"]
        graph.add_edge("u", "v", key=key, pts=pts)

    return graph


graph = braided_theta_graph()
Y = sp.Symbol("Y")
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=(0.0, 0.0, 0.0),
    return_result=True,
)

pd_terms = sorted(result.projection.pd_code.split(";"))
print(f"Upsilon(G; Y) = {sp.expand(result.polynomial)}")
print(f"crossings = {result.projection.num_crossings}")
print(f"pd_terms = {pd_terms}")
```

Output:

```text
Upsilon(G; Y) = -Y**4 - Y**3 - 2*Y**2 - Y - 1
crossings = 4
pd_terms = ['V[3,10,6]', 'V[4,7,0]', 'X[1,8,0,7]', 'X[2,5,1,4]', 'X[9,2,10,3]', 'X[9,6,8,5]']
```

The projection and visualization chapters show the corresponding 3D and planar
plotting calls. The quick start keeps the computational path compact.

By default, `compute_yamada_polynomial` samples 10 approximately non-isotopic
rotations and uses the valid projection with the fewest crossings. Supplying
`rotation_angles=(...)` bypasses sampling and computes exactly that view.

If the selected planar diagram has 10 or more crossings, the function emits a
`RuntimeWarning` because state-sum evaluation can become expensive.
