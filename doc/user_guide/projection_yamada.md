# Projection, PD Codes, And Yamada Polynomials

This page separates three objects that are easy to conflate:

1. an embedded spatial graph in three dimensions;
2. a selected regular planar diagram with over/under information; and
3. the Yamada polynomial computed from the graph/diagram data.

<div class="kg-link-row">
  <a href="../quickstart.html">Run the Quick Start</a>
  <a href="workflow_overview.html">Review the full workflow</a>
  <a href="../api/projection.html">Projection API</a>
  <a href="../api/yamada.html">Yamada API</a>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/pdcode_to_yamada.png" alt="Planar diagram and PD-code data leading to a Yamada-polynomial calculation">
</div>

## Choose the correct entry point

### Abstract graph with no crossing data

Use the direct evaluator when the intended object is an undirected abstract
Graph/MultiGraph:

```python
import sympy as sp
from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial

Y = sp.Symbol("Y")
polynomial = compute_graph_yamada_polynomial(ThetaGraph(3), Y)
```

This route does not invent a spatial embedding or projection.

### Embedded spatial graph

Use the projection helper when node `pos` and edge `pts` geometry matter:

```python
from knotted_graph.projection import compute_yamada_polynomial

result = compute_yamada_polynomial(
    graph,
    Y,
    num_rotation_samples=16,
    n_jobs=1,
    return_result=True,
)

print(result.polynomial)
print(result.projection.rotation_angles)
print(result.projection.num_crossings)
```

`return_result=True` is recommended for scientific work because it retains the
selected projection rather than returning only a polynomial expression.

## What projection selection does

Projection choice is part of the computation, not just plotting. Candidate
rotations are evaluated for regularity and diagram complexity. A candidate may
fail because of overlapping projected segments, a vertex/edge degeneracy, or
an ambiguous crossing event.

For a deterministic example, supply explicit angles:

```python
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=(0.0, 0.0, 0.0),
    n_jobs=1,
    return_result=True,
)
```

Use explicit angles only after confirming that the view is regular. Otherwise,
let `select_projection` sample views and retain its diagnostic result.

## How to read PD-code data

The planar diagram records:

- projected graph arcs;
- true graph vertices;
- transverse crossing locations;
- which arc passes over and under; and
- cyclic/local incidence information needed by the evaluator.

A crossing visible in 2-D is not a graph vertex unless it was already a vertex
of the spatial graph. Changing the viewing direction may change the number and
locations of crossings while leaving the embedded graph and invariant
unchanged.

When debugging, inspect the projection and PD records before blaming the
polynomial evaluator. Many apparent invariant failures are actually invalid or
degenerate projections.

## Variable and normalization conventions

The user-facing examples write the polynomial as
\(\Upsilon(G;Y)\). Some backend, benchmark, or literature-comparison code uses
`A` internally. The symbol name does not change the mathematics, but mixing
normalization conventions can.

Record:

- the symbolic variable;
- whether normalization was enabled;
- the selected evaluation method;
- the exact expression before/after expansion or factorization; and
- the package/backend version.

The Quick Start uses `normalize=False` for the embedded theta graph so its
Laurent expression can be compared directly with the abstract evaluator.

## Interpreting zero

A zero result is not automatically an error. For example, a graph containing a
bridge has zero Yamada polynomial under the implemented convention. First
inspect graph connectivity, bridges, leaf cleanup, and whether the intended
object was open or closed.

Also distinguish:

- a successfully evaluated exact zero;
- a failed projection with no polynomial;
- a skipped or filtered graph; and
- a missing backend or timeout.

These states must not be collapsed into one placeholder value in tables or
figures.

## Cost and worker policy

For \(c\) diagram crossings, a direct state expansion has approximately
\(3^c\) states before reductions. Geometry sampling and projection selection add
their own costs.

- Keep `n_jobs=1` for tutorials and shared/login environments.
- Inspect crossing counts before enabling parallel work.
- Use an explicit cluster allocation for expensive batches.
- Record whether the compiled native backend was available.
- Treat partial projection failures as diagnostics, not silently discarded
  events.

## Formula-discovery and publication notebooks

The formula-discovery notebook is an advanced reproduction artifact, not the
first introduction to Yamada evaluation. Its setup permits ordinary reading
and exploratory execution on review branches. Exact publication regeneration
is opt-in:

```bash
export KNOTTEDGRAPH_STRICT_PUBLICATION_REGENERATION=1
```

Strict mode additionally requires the audited source revision, a clean library
tree, the expected editable checkout, and the optimized factorized native
backend. Without strict mode, the notebook emits explicit warnings instead of
failing solely because the branch was renamed.

Start with {doc}`../quickstart`, then use the
[Advanced and Reproduction notebook](https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/03_advanced_and_reproduction.ipynb).
Open formula discovery only when you need the held-out reconstruction/audit
workflow and understand its native-backend and cache requirements.
