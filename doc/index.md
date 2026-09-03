# KnottedGraph

<div class="kg-hero">
  <p class="kg-lead"><strong>KnottedGraph</strong> organizes geometric input, graph extraction, projection, PD encoding, layout, visualization, and Yamada-polynomial computation into a reusable research library. Start with the version-aware installation guide and tested Quick Start, then continue to the workflow pages and notebooks.</p>
  <div class="kg-link-row">
    <a href="installation.html">Install</a>
    <a href="quickstart.html">Quick Start</a>
    <a href="feature_status.html">Choose A Workflow</a>
    <a href="user_guide/index.html">User Guide</a>
    <a href="applications/index.html">Applications</a>
    <a href="troubleshooting.html">Troubleshooting</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="site_figures/architecture.png" alt="KnottedGraph package architecture">
</div>

## New here?

1. Read {doc}`installation` so the 0.2 review API is not confused with the
   legacy PyPI package.
2. Run {doc}`quickstart` and compare the printed polynomial with the expected
   result.
3. Choose your real starting object from {doc}`feature_status`.
4. Read {doc}`user_guide/workflow_overview` before opening a heavy application
   or publication-reproduction notebook.

| Starting point | Go to |
| --- | --- |
| Coordinate, biomolecular, polymer, spatial-CSV, or surface data | {doc}`user_guide/input_adapters` |
| Existing embedded graph | {doc}`user_guide/workflow_overview` |
| Projection, PD code, or invariant question | {doc}`user_guide/projection_yamada` |
| Analytic knot or braid | {doc}`applications/analytic_knot_fields` |
| Nodal/material/phase-map workflow | {doc}`applications/index` |
| Import or native-backend problem | {doc}`troubleshooting` |

The website separates introductory tutorials from application research and
publication reproduction. A linked notebook may require optional dependencies,
cached data, or compute resources; check its top-of-notebook runtime card before
using **Run All**.

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
feature_status
user_guide/index
applications/index
troubleshooting
api/index
developer/index
```
