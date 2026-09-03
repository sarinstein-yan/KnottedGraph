# Application And Reproduction Workflows

<div class="kg-hero">
  <p class="kg-lead">Choose an application only after you understand the embedded-graph contract. Guided application notebooks explain a reusable public route; advanced reproduction notebooks preserve a specific research calculation and may require caches, native backends, or compute resources.</p>
  <div class="kg-link-row">
    <a href="../feature_status.html">Check Feature Status</a>
    <a href="analytic_knot_fields.html">Analytic Knot Fields</a>
    <a href="yamada_formula_discovery.html">Formula Discovery</a>
    <a href="hamiltonian_yamada_phase_maps.html">Hamiltonian Phase Maps</a>
  </div>
</div>

## Choose the right level

| Route | Intended reader | Start here when... | Execution class |
| --- | --- | --- | --- |
| {doc}`mathematical_investigations` | first-time application user | you have a named abstract graph family | guided; base API |
| {doc}`analytic_knot_fields` | application user | you have a knot/link name, torus type, or braid | guided, then compute-intensive extraction |
| {doc}`material_fingerprints` | domain user | you already have an in-memory nodal/material model | optional `nodal`; grid work |
| {doc}`protein_derived_spatial_graphs` | input user | you have PDB/mmCIF data and need to understand the implemented boundary | adapter is public; domain mapping is not yet generic |
| {doc}`yamada_formula_discovery` | researcher reproducing a result | you need the exact dataset/held-out symbolic checks | advanced publication reproduction |
| {doc}`hamiltonian_yamada_phase_maps` | domain researcher | you need a two-parameter Hamiltonian topology scan | advanced, cached, compute-intensive |

The last two routes are deliberately not presented as beginner tutorials. Read
their web pages first; each page describes prerequisites, outputs, interpretation,
and which cells are safe to browse without regenerating publication data.

<div class="kg-wide-figure">
  <img src="../site_figures/feature_image1.png" alt="Material Fermi-surface fingerprints and Yamada polynomials">
</div>

```{toctree}
:hidden:
:maxdepth: 1

material_fingerprints
protein_derived_spatial_graphs
mathematical_investigations
analytic_knot_fields
yamada_formula_discovery
hamiltonian_yamada_phase_maps
```
