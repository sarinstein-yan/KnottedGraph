# Material Fermi Surface Fingerprints

<div class="kg-hero">
  <p class="kg-lead">The physics applications notebook explains how material Fermi surfaces are converted into knotted spatial graphs and then summarized by Yamada-polynomial fingerprints. This page uses the feature image as the visual summary of the material-to-graph-to-invariant pipeline.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/applications/01_physics_applications.ipynb">Open 01_physics_applications.ipynb</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/feature_image1.png" alt="Material Fermi-surface fingerprints and Yamada polynomials">
</div>

## What this route accepts

This is an **application workflow**, not a material-file adapter. It starts from
an in-memory symbolic Bloch Hamiltonian $H(k_x,k_y,k_z)$. Install the `nodal`
extra before constructing `MaterialFermiSurface`; install `viz` as well for the
interactive Plotly views used by the notebook.

The supplied examples include symbolic constructors for $Ti_3Al$, a
$D_6$/$TiB_2$ model, and $YH_3$. The notebook also shows a six-band
$Co_2MnGa$ construction and a template for a user-defined SymPy matrix. See the
{doc}`../feature_status` matrix for the exact public/application boundary.

## Data flow

```text
symbolic Hermitian Hamiltonian
    -> sample selected band gap on an explicit k-space grid
    -> threshold and inspect the complete surface
    -> skeletonize the occupied region
    -> construct and validate an embedded MultiGraph
    -> optionally clean the graph
    -> select a projection and compute an invariant
```

The surface and graph are deliberately shown separately. A continuous
isosurface is physical/numerical context; it is not itself the graph used by
the projection and invariant code.

## Decisions you must make explicitly

Before interpreting a fingerprint, record and test:

- the momentum-space `span` and grid `dimension`;
- the selected `band_pair` and `gap_tol`;
- whether the extracted surface touches a domain boundary;
- the skeletonization and graph-cleanup settings;
- graph node/edge/degree diagnostics before and after cleanup; and
- a resolution or threshold perturbation check.

Leaf removal, short-edge contraction, and smoothing can change the scientific
object if used indiscriminately. Inspect the raw graph first and do not treat a
visually cleaner skeleton as proof of topological equivalence.

## How to read the notebook

Sections 2--7 build and diagnose non-Hermitian nodal examples, including the
surface, medial axis, graph, physical fields, and Yamada comparisons. Section 8
then applies the analogous surface-to-graph workflow to the material models.
Many examples use $300^3$ or $400^3$ grids and are publication-scale workloads;
read the Markdown and inspect cached figures before running those cells on a
proper compute resource.

Continue with {doc}`../user_guide/workflow_overview` for the reusable graph
stages and {doc}`../user_guide/projection_yamada` before interpreting a final
polynomial.
