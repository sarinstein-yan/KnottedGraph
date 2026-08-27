# Projection, PD Codes, And Yamada Polynomials

<div class="kg-hero">
  <p class="kg-lead">The projection and Yamada notebooks show how a spatial graph becomes a planar diagram, how the PD-code data is assembled, and how the final invariant \(\Upsilon(G;Y)\) is computed. The figure below summarizes the passage from diagrammatic data to polynomial evaluation.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/Latest_Workplace/User_guide/01_getting_started.ipynb">Open Getting Started</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/Latest_Workplace/User_guide/03_advanced_and_reproduction.ipynb">Open Advanced And Reproduction</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/Latest_Workplace/User_guide/applications/02_mathematics_applications.ipynb">Open Mathematics Applications</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/pdcode_to_yamada.png" alt="PD code to Yamada calculation workflow">
</div>

## Displayed Nonzero Examples

The gallery below contains fifteen source-backed curves, generating spines,
and embedded spatial graphs whose displayed Yamada polynomials are all nonzero
and pairwise distinct. Each footer uses the paper notation
\(\Upsilon(G;Y)\). The displayed two-dimensional crossings belong to the
chosen projection and are not graph vertices.

Some panels were parsed or constructed by application-level code before they
were normalized to the common embedded-graph contract. In particular, their
GraphML or standalone JSON labels do not imply public readers for those formats
in `knotted_graph.inputs`. The accepted panel identities and composition are
locked by the [Task 2 figure package](https://github.com/sarinstein-yan/KnottedGraph/tree/407df7ea442603d84a6e390eb78828f39698485b/examples/input_gallery/task2_figures).

<div class="kg-wide-figure">
  <a href="../site_figures/input_yamada_nonzero.png">
    <img src="../site_figures/input_yamada_nonzero.png" alt="Fifteen-panel grid of distinct spatial graphs and nonzero Yamada polynomials" loading="lazy" decoding="async">
  </a>
  <p class="kg-caption">Fifteen visually distinct input-derived spatial graphs with pairwise-distinct displayed nonzero Yamada polynomials.</p>
</div>
