# User Guide

This guide follows the generic, domain-independent KnottedGraph pipeline:
heterogeneous inputs are normalized into spatial-graph objects, inspected,
projected to planar diagrams and PD codes, and then passed to invariant,
layout, or visualization tools.

Start with [Quick Start](../quickstart.md) for the shortest end-to-end path.
Then use these chapters in order: first the reusable library machinery, then the
application tutorials that show the same machinery in physics, biomolecular,
and mathematical workflows.

```{toctree}
:maxdepth: 2

../quickstart
workflow_overview
input_adapters
inspection_pipeline
projection_yamada
repulsive_layout
../applications/index
```
