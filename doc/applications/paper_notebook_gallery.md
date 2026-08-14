# Paper Notebook Output Policy

The paper notebooks contain many useful publication panels, but the public
documentation should display only outputs that teach a reproducible
KnottedGraph workflow. Composite grids, crop experiments, and figures made only
from generic NetworkX drawing commands should stay as paper assets rather than
API tutorial figures.

## What Is Embedded

| Output type | Documentation page |
| --- | --- |
| Surface, skeleton-point, and spatial-graph inspection outputs generated from one `NodalSkeleton` object | [Inspecting Intermediate Objects](../user_guide/inspection_pipeline.md) |
| Planarity examples shown as surface-to-spatial-graph-to-printed-result workflows | [Mathematical Workflows](../user_guide/mathematical_workflows.md) |
| Public Plotly graph outputs for nodal models | [Non-Hermitian Nodal Skeletons](nodal_skeleton.md) |

## What Is Archived

The extracted notebook images under `doc/assets/paper_notebook_outputs/` are
kept as source material, but not every image is embedded. In particular, the
docs no longer show stitched appendix grids, standalone Petersen/$K_{3,3}$
drawings, multiband gallery panels, or local `build_*` figure wrappers unless
the page also exposes the underlying library objects and plotting code.
