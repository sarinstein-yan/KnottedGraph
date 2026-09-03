# Mathematical Investigations

<div class="kg-hero">
  <p class="kg-lead">The mathematics applications notebook gathers graph-family calculations, planarity checks, intrinsic-linkedness examples, and Yamada-polynomial comparisons. The two figures below point to the planarity and rich-topology themes developed there.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/applications/02_mathematics_applications.ipynb">Open 02_mathematics_applications.ipynb</a>
    <a href="../data/structured_graph_yamada_dataset.csv">Open structured graph dataset</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/planarity.png" alt="Planarity workflow figure">
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/rich_topology_intrinsic_linkedness_2row.png" alt="Rich topology and intrinsic linkedness figure">
</div>

## Start with an abstract graph family

The base installation provides a catalog of structured undirected graph
families. A minimal exact calculation is:

```python
import sympy as sp

from knotted_graph.applications.mathematical import build_graph_case
from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial

Y = sp.Symbol("Y")
graph, drawing_positions = build_graph_case("theta", 5)
polynomial = compute_graph_yamada_polynomial(graph, Y)
print(polynomial)
```

`drawing_positions` is a two-dimensional layout for inspection. It is not a
three-dimensional embedding and does not create edge `pts` data. Use the direct
graph evaluator for an abstract graph; use the projection route only after you
have a validated spatial embedding.

## What the notebook demonstrates

The notebook progresses from simpler to more specialized questions:

1. compare direct graph evaluation with projection-based evaluation;
2. scan theta and other structured graph families;
3. inspect graph size, loops, multiplicities, and polynomial forms;
4. test planarity, connectivity, and a Petersen-minor certificate;
5. generate and inspect a small structured-graph dataset;
6. reuse PD-code output with user-defined Jones/Alexander calculations; and
7. use exact computations to formulate and test a periodic-theta pattern.

The last stages illustrate computational evidence and held-out checks. A
pattern seen in a finite family is not, by itself, an all-parameter proof.

## Choose the correct route

| Starting object | Route |
| --- | --- |
| Abstract `Graph`/`MultiGraph` | `compute_graph_yamada_polynomial` |
| Named catalog family | `build_graph_case`, then direct evaluation |
| Embedded spatial graph | validate, select a regular projection, then evaluate |
| Existing PD code | use the PD-code/invariant layer directly |

See {doc}`../api/applications` for catalog entry points,
{doc}`../api/yamada` for the direct evaluator, and
{doc}`../user_guide/projection_yamada` for the spatial route and its scaling
limits.
