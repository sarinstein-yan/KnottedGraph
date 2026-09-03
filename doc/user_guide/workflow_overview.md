# Workflow Overview

KnottedGraph does not require every user to start from the same scientific
object. The common part begins once an embedded spatial graph is available.

<div class="kg-link-row">
  <a href="../feature_status.html">Choose by starting object</a>
  <a href="input_adapters.html">Load external data</a>
  <a href="projection_yamada.html">Projection and Yamada</a>
  <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/02_core_workflows.ipynb">Open Core Workflows notebook</a>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/skeletonization_steps.png" alt="Surface and volume skeletonization stages leading to an embedded spatial graph">
</div>

## The reusable path

```text
external data / analytic field / in-memory model
    -> adapter or application-specific extraction
    -> embedded MultiGraph(pos, pts)
    -> validate and simplify without changing intended topology
    -> choose a regular planar projection
    -> inspect crossings and PD-code records
    -> evaluate Upsilon(G; Y)
    -> retain settings, diagnostics, and provenance
```

Not every project uses every stage. An abstract graph can go directly to the
crossing-free graph evaluator. A surface loader stops at `PolyData` until the
user chooses an extraction method. A field or Hamiltonian workflow may require
3-D sampling and skeletonization before a graph exists.

## Stage 1: identify what you actually have

Start by distinguishing these cases:

- **Ordered coordinates** describe one sampled curve; they do not encode a
  general graph topology.
- **Node and edge data** can represent a spatial MultiGraph directly.
- **Surface or volume data** require an extraction decision; multiple skeletons
  can be plausible for the same physical geometry.
- **Analytic knot fields and Hamiltonians** are application objects that must be
  sampled at an explicit domain, resolution, and level/energy.
- **Abstract graphs** contain topology but no embedding or crossing data.

Use {doc}`input_adapters` for direct public input support and
{doc}`../feature_status` for application routes.

## Stage 2: inspect the graph contract

```python
from knotted_graph.core import ensure_embedding

graph = ensure_embedding(graph)
print("nodes:", graph.number_of_nodes())
print("edges:", graph.number_of_edges())
print("degrees:", sorted(dict(graph.degree()).values()))
```

Before simplifying anything, inspect:

- connected components;
- vertex degrees and parallel edges;
- leaf/bridge structure;
- edge sampling density;
- coordinate units and scale; and
- whether boundaries or periodic faces were touched during extraction.

Cleanup operations such as smoothing, short-edge contraction, and leaf removal
are not interchangeable. Record which operation was used and compare graph
diagnostics before and after it. A visually smoother graph is not automatically
topologically equivalent.

## Stage 3: select a regular projection

```python
from knotted_graph.projection import select_projection

projection = select_projection(
    graph,
    num_rotation_samples=16,
)
print("angles:", projection.rotation_angles)
print("crossings:", projection.num_crossings)
```

A valid projection must avoid degeneracies such as a projected vertex lying on
an unrelated edge or multiple events collapsing to the same point. The selector
samples candidate views and returns an inspectable result rather than hiding
the chosen rotation.

Projected crossings are diagram events, not new graph vertices. Read the
crossing and arc records before interpreting a PD code.

## Stage 4: compute the invariant with provenance

```python
import sympy as sp
from knotted_graph.projection import compute_yamada_polynomial

Y = sp.Symbol("Y")
result = compute_yamada_polynomial(
    graph,
    Y,
    n_jobs=1,
    return_result=True,
)

print("Upsilon(G; Y) =", result.polynomial)
print("projection crossings =", result.projection.num_crossings)
```

Keeping only the polynomial discards important reproducibility information.
Retain the selected projection, rotation policy, normalization convention,
worker count, backend status, and any failed candidate views.

State evaluation grows approximately as \(3^c\) with projected crossing count
\(c\). Use one worker by default, inspect the chosen view, and request cluster
resources explicitly for expensive cases.

## Extraction and skeletonization are broader than invariant calculation

Skeletonization can be used to obtain compact centerline graphs for geometry
inspection, comparison, routing, or visualization even when no Yamada
polynomial is needed. Conversely, a skeleton extracted from a scalar field is
not automatically the unique topological representation of that field.

For sampled surfaces/volumes, report at least:

- sampling domain and grid resolution;
- threshold, level, energy, or gap parameter;
- boundary contact and periodicity policy;
- skeletonization/extraction method;
- cleanup parameters;
- graph diagnostics before projection; and
- resolution or perturbation checks used to establish stability.

## Tutorial versus reproduction notebooks

The notebooks have different purposes:

| Level | Use | Expectation |
| --- | --- | --- |
| `01_getting_started` | first successful graph/projection/invariant run | modest runtime, copyable cells, expected outputs |
| `02_core_workflows` | inspect each processing stage | reduced-resolution tutorial defaults |
| `03_advanced_and_reproduction` | robustness and diagnostic policy | more computation and domain knowledge |
| application notebooks | domain-specific scientific workflows | optional extras; may include paper-sized modes |
| benchmark notebooks | correctness/performance evidence | native backends, caches, and compute resources may be required |

Do not treat a publication-regeneration notebook as the first tutorial. Start
with the Quick Start and move to reproduction only after the common graph and
projection contracts are clear.

## Reproducibility checklist

For a result intended for comparison or publication, save:

1. package version and Git commit;
2. input identifier, file checksum, or analytic constructor parameters;
3. coordinate units, closure, chain/model, and domain choices;
4. extraction/skeletonization and cleanup settings;
5. graph fingerprint and node/edge/degree diagnostics;
6. projection angles, crossing count, and failed-view diagnostics;
7. Yamada method, normalization, variable, worker count, and backend status;
8. warnings/issues returned by adapters or applications; and
9. runtime environment and relevant optional/native dependency versions.

Continue with {doc}`projection_yamada` for diagram/invariant details or open the
[Core Workflows notebook](https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/02_core_workflows.ipynb)
for a staged executable example.
