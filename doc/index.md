# KnottedGraph

`knotted_graph` is a computational package for embedded spatial graphs, planar
diagram construction, and graph-polynomial invariants. The documentation is
organized around a simple progression: learn the common graph pipeline first,
then use it in scientific and mathematical applications.

```{image} ../assets/paper/architecture.svg
:alt: KnottedGraph architecture
:width: 88%
:align: center
```

## Where should I start?

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} New to KnottedGraph?
:link: quickstart
:link-type: doc

Run the shortest pipeline from a 2D surface to spatial graph, planar projection,
PD code, and $\Upsilon(G;Y)$.
+++
**Start with the Quick Start →**
:::

:::{grid-item-card} Building a generic spatial-graph pipeline?
:link: user_guide/index
:link-type: doc

Learn inputs, inspection, projection, Yamada evaluation, and layout without
assuming a particular scientific domain.
+++
**Open the User Guide →**
:::

:::{grid-item-card} Applying the library to a research problem?
:link: applications/index
:link-type: doc

Follow complete tutorials for nodal skeletons, materials, proteins, notebooks,
and mathematical graph-family exploration.
+++
**Browse Application Tutorials →**
:::

:::{grid-item-card} Looking for a function or implementation detail?
:link: api/index
:link-type: doc

Use the API reference for callable interfaces, or the developer section for the
package architecture.
+++
**Open the API Reference →**
:::

::::

## Package at a glance

The public source layout mirrors the computational pipeline:

- `knotted_graph.core` — embedded graph validation and graph constructors.
- `knotted_graph.inputs` — adapters for coordinates, polymers, biomolecules,
  CSV data, and surface meshes.
- `knotted_graph.projection` — rotations, planar diagrams, PD codes, and
  graph-to-Yamada entry points.
- `knotted_graph.invariants.yamada` — Yamada-polynomial evaluation engines.
- `knotted_graph.layout.repulsive` — optional curve-network relaxation.
- `knotted_graph.applications` — domain-specific workflows built on the generic
  core.

```{toctree}
:hidden:
:caption: User Guide
:maxdepth: 2

user_guide/index
```

```{toctree}
:hidden:
:caption: Reference & Development
:maxdepth: 2

api/index
developer/index
```
