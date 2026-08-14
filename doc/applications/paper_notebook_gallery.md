# Reproducibility and Notebook Outputs

The paper notebooks contain many useful publication panels, but the public
documentation should display only outputs that teach a reproducible
KnottedGraph workflow. Composite grids, crop experiments, and figures made only
from generic NetworkX drawing commands should remain paper assets rather than
API tutorial figures.

This page documents that boundary so readers can distinguish tutorial material
from archived manuscript-generation outputs.

## What is embedded

| Output type | Documentation page |
| --- | --- |
| Surface, skeleton-point, and spatial-graph inspection outputs generated from one `NodalSkeleton` object | [Inspecting Intermediate Objects](../user_guide/inspection_pipeline.md) |
| Planarity examples shown as surface-to-spatial-graph-to-printed-result workflows | [Mathematical Workflows](mathematical_workflows.md) |
| Public Plotly graph outputs for nodal models | [Non-Hermitian Nodal Skeletons](nodal_skeleton.md) |

## What is archived

The extracted notebook images under `doc/assets/paper_notebook_outputs/` are
kept as source material, but not every image is embedded. In particular, the
documentation does not show stitched appendix grids, standalone Petersen or
$K_{3,3}$ drawings, multiband gallery panels, or local `build_*` figure wrappers
unless the page also exposes the underlying library objects and plotting code.

## Why this separation matters

Application tutorials should teach a reader how to reproduce a scientific or
mathematical workflow with the public package. Manuscript-only assembly code is
valuable for provenance, but it should not obscure the public API or compete
with the tutorial sequence in the main navigation.
