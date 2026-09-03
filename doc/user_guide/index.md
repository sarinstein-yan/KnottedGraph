# User Guide

<div class="kg-hero">
  <p class="kg-lead">Start with a small tested route, then move from input handling to graph inspection, projection, invariant evaluation, and only then to application or publication workflows.</p>
  <div class="kg-link-row">
    <a href="../feature_status.html">Choose A Supported Route</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/00_user_guide.ipynb">Open the notebook map</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/01_getting_started.ipynb">Getting Started</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/02_core_workflows.ipynb">Core Workflows</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/03_advanced_and_reproduction.ipynb">Advanced And Reproduction</a>
  </div>
</div>

The {doc}`../feature_status` page distinguishes public adapters, optional
features, application APIs, and external native backends before you choose a
notebook. New users can first run the copyable {doc}`../quickstart` without
placing that page in a second documentation hierarchy.

## Recommended order

| Step | Read or run | What you should understand before continuing |
| --- | --- | --- |
| 1 | {doc}`../installation` and {doc}`../quickstart` | active version/environment and one expected exact result |
| 2 | {doc}`input_adapters` | what your source data becomes and which choices are not automatic |
| 3 | {doc}`workflow_overview` | graph contract, extraction/cleanup boundaries, and provenance |
| 4 | {doc}`projection_yamada` | projection regularity, PD-code meaning, zero versus failure, and cost |
| 5 | {doc}`../applications/index` | one domain-specific application or advanced reproduction route |

The first three notebooks are progressive tutorials. Application notebooks
assume the common concepts above; benchmark notebooks are evidence artifacts
and are not part of the default newcomer path.

<div class="kg-wide-figure">
  <img src="../site_figures/knot_to_spatial_graph.png" alt="Knot and surface workflow leading to a spatial graph">
</div>

```{toctree}
:hidden:
:maxdepth: 2

workflow_overview
input_adapters
projection_yamada
repulsive_layout
../applications/index
```
