# Non-Hermitian Nodal Skeletons

The nodal-skeleton workflow is the original application that motivated this
package. It is now a domain-specific application module:

```python
from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import hopf_link_bloch_vector
```

The workflow samples a two-band non-Hermitian Hamiltonian in 3D momentum space,
extracts the exceptional-surface interior, skeletonizes that region, and returns
a spatial multigraph that can be projected to a planar diagram.

```{image} ../assets/nodal/threelink_gamma=0.1.svg
:alt: Non-Hermitian exceptional skeleton graph
:width: 80%
:align: center
```

## Minimal Workflow

```python
import sympy as sp

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import hopf_link_bloch_vector

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
bloch_vector = hopf_link_bloch_vector(0.1, k_symbols=(kx, ky, kz))

GRID_DIMENSION = 200  # Use 64 for a quick local smoke run.

ske = NodalSkeleton(
    bloch_vector,
    k_symbols=(kx, ky, kz),
    dimension=GRID_DIMENSION,
)

graph = ske.skeleton_graph()
```

The returned graph follows the generic spatial-graph contract: each node has a
finite 3D `pos` attribute and each geometric edge has a `pts` polyline.

## Yamada Polynomial

```python
import sympy as sp

A = sp.Symbol("A")
polynomial = ske.yamada_polynomial(A, num_rotation_samples=10)
```

`NodalSkeleton.yamada_polynomial` delegates to the generic
`knotted_graph.projection.compute_yamada_polynomial` entry point. If no explicit
rotation is supplied, it samples rotations and uses the projection with the
fewest crossings. If the selected diagram has at least 10 crossings, a
`RuntimeWarning` is emitted before polynomial evaluation.

## Physics Figures

The legacy physics media are stored under `doc/assets/nodal/`. They document the
application scenario and are intentionally not used as root-package branding.

```{image} ../assets/nodal/field_berry.png
:alt: Berry curvature visualization
:width: 70%
:align: center
```
