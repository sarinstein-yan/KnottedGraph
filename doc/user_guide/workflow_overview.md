# Workflow Overview

<div class="kg-hero">
  <p class="kg-lead">The core workflow notebook explains how geometric data is inspected stage by stage: surface or volume, skeletonization, graph reconstruction, projection, and invariant calculation. This page uses the skeletonization figure as the visual anchor for that process.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/Latest_Workplace/User_guide/02_core_workflows.ipynb">Open 02_core_workflows.ipynb</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/skeletonization_steps.png" alt="Skeletonization steps from surface to spatial graph">
</div>

## Skeletonization Beyond Invariant Calculation

The normalized graph is useful even when an invariant is not the final goal. It
can retain connectivity, embedded geometry, and flow orientation for
visualization, comparison, and subsequent application-specific analysis. The
gallery below collects twelve result-only examples: biomolecular backbones,
coordinate-derived graphs, integrated flow paths, scalar-phase skeletons, and
a Hamiltonian-derived periodic skeleton.

The accepted Task 2 static gallery collects these twelve examples without
polynomial annotations. This page reproduces that illustration and does not
rerun or independently verify invariant calculations. The annotations remain
omitted because the figure focuses on graph conversion and skeletonization
beyond invariant calculation. The accepted panel identities and composition
are locked by the
[Task 2 figure package](https://github.com/sarinstein-yan/KnottedGraph/tree/407df7ea442603d84a6e390eb78828f39698485b/examples/input_gallery/task2_figures).

<div class="kg-wide-figure">
  <a href="../site_figures/input_skeletonization_beyond_yamada.png">
    <img src="../site_figures/input_skeletonization_beyond_yamada.png" alt="Twelve-panel grid of skeletonization and graph-conversion outputs without invariant labels" loading="lazy" decoding="async">
  </a>
  <p class="kg-caption">Twelve graph-compatible results illustrating biological, coordinate, flow, phase, and Hamiltonian skeletonization workflows.</p>
</div>
