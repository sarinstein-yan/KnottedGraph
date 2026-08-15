# Application Tutorials

These tutorials show how the generic KnottedGraph machinery becomes complete
research-facing workflows. If you are learning the reusable library primitives,
start with the [User Guide](../user_guide/index.md); use this section when you
want to see a full scientific or mathematical calculation carried from input
object to plotted graph, planar projection, and $\Upsilon(G;Y)$ output.

In the left navigation this section appears after **Repulsive Layout**, so the
reader first learns the generic input, inspection, projection, invariant, and
layout tools before seeing the domain-specific workflows.

The section is ordered intentionally. It begins with domain applications,
places reproducibility notes after the main physics and biomolecular examples,
and ends with **Mathematical Workflows**, where the same invariant engine is
used for graph-family experiments and conjecture building.

## Tutorial Order

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Non-Hermitian Nodal Skeletons
:link: nodal_skeleton
:link-type: doc

Extract an exceptional-surface skeleton, inspect the spatial graph, project it,
and compute topological invariants.
+++
**1. Start here for surface-to-skeleton physics.**
:::

:::{grid-item-card} Material Fermi-Surface Fingerprints
:link: material_fingerprints
:link-type: doc

Follow the manuscript-facing material workflow from surface data to knotted
spatial-graph fingerprints.
+++
**2. Continue to material Hamiltonian examples.**
:::

:::{grid-item-card} Protein-Derived Theta Graphs
:link: biomolecular_protein_workflow
:link-type: doc

Convert biomolecular coordinate data into embedded theta graphs and continue
through the common layout and invariant pipeline.
+++
**3. Compare with biomolecular inputs.**
:::

:::{grid-item-card} Nodal Skeleton Notebook
:link: nodal_skeleton_notebook
:link-type: doc

Open the notebook version of the nodal-skeleton workflow when you want a
reproducible, cell-by-cell companion to the tutorial page.
+++
**4. Reproduce the nodal example as a notebook.**
:::

:::{grid-item-card} Notebook Figure Gallery
:link: paper_notebook_gallery
:link-type: doc

Inspect the notebook figures that directly teach the framework: Hamiltonian
model, surface, skeleton points, spatial graph, and downstream checks.
+++
**5. Audit the paper figures.**
:::

:::{grid-item-card} Mathematical Workflows
:link: mathematical_workflows
:link-type: doc

Use graph families and Yamada polynomials for combinatorial experiments,
pattern discovery, recurrence searches, and conjecture building.
+++
**6. Finish with mathematical exploration.**
:::

::::

## Section Navigation

The same order is used by the page navigation and the previous/next links:

```{toctree}
:caption: Application Tutorial Order
:maxdepth: 1

nodal_skeleton
material_fingerprints
biomolecular_protein_workflow
nodal_skeleton_notebook.ipynb
paper_notebook_gallery
mathematical_workflows
```

The final page is intentionally **Mathematical Workflows**. It broadens the
section from domain-specific examples to mathematical discovery with the same
Yamada-polynomial engine.
