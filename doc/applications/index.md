# Application Tutorials

These tutorials show how the generic KnottedGraph machinery is used in complete
research-facing workflows. If you are learning the reusable library primitives,
start with the [User Guide](../user_guide/index.md); use this section when you
want an end-to-end application.

## Choose a workflow

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Non-Hermitian Nodal Skeletons
:link: nodal_skeleton
:link-type: doc

Extract an exceptional-surface skeleton, inspect the spatial graph, project it,
and compute topological invariants.
:::

:::{grid-item-card} Material Fermi-Surface Fingerprints
:link: material_fingerprints
:link-type: doc

Follow the manuscript-facing material workflow from surface data to knotted
spatial-graph fingerprints.
:::

:::{grid-item-card} Protein-Derived Theta Graphs
:link: biomolecular_protein_workflow
:link-type: doc

Convert biomolecular coordinate data into embedded theta graphs and continue
through the common layout and invariant pipeline.
:::

:::{grid-item-card} Mathematical Workflows
:link: mathematical_workflows
:link-type: doc

Use graph families and Yamada polynomials for combinatorial experiments,
pattern discovery, recurrence searches, and conjecture building.
:::

::::

## Reproducible notebook material

The executable nodal-skeleton notebook and the notebook-output policy sit after
the main domain tutorials. They are useful for reproducing examples, but they
are supporting material rather than the conceptual entry point to the library.

```{toctree}
:hidden:
:maxdepth: 1

nodal_skeleton
material_fingerprints
biomolecular_protein_workflow
nodal_skeleton_notebook.ipynb
paper_notebook_gallery
mathematical_workflows
```

The section intentionally ends with **Mathematical Workflows**. That page
broadens the perspective from domain-specific examples to mathematical
discovery with the same invariant engine.
