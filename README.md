# `knotted_graph`

![pre-alpha](https://img.shields.io/badge/status-pre--alpha-red?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/knotted_graph)](https://pypi.org/project/knotted_graph/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://sarinstein-yan.github.io/KnottedGraph/)

`knotted_graph` is a computational package for embedded spatial graphs,
planar-diagram construction, and graph polynomial invariants. It provides a
generic core for pure mathematical and computational workflows, with optional
application packages built on top.

<p align="center">
  <img src="assets/paper/architecture.svg" width="780" alt="KnottedGraph architecture">
</p>

## Install

```bash
pip install knotted_graph
```

For local development:

```bash
git clone https://github.com/sarinstein-yan/KnottedGraph.git
cd KnottedGraph
uv sync --all-groups
```

Optional Python extras are split by workflow:

```bash
pip install "knotted_graph[nodal]"
pip install "knotted_graph[surface]"
pip install "knotted_graph[viz]"
pip install "knotted_graph[repulsion]"
pip install "knotted_graph[notebook]"
pip install "knotted_graph[benchmark]"  # includes optional Topoly lasso backend
pip install "knotted_graph[all]"
```

### Repulsive-layout native dependency

The `repulsion` extra installs the **Python-side dependencies**. The optional
Repulsor solver is a separate C++ dependency and is intentionally not vendored
inside `knotted_graph`.

For a reproducible source checkout, use the repository bootstrap:

```bash
python scripts/bootstrap_repulsion.py
export REPULSOR_ROOT="$PWD/external/Repulsor"
```

The bootstrap checks out the exact Repulsor revision pinned for this
KnottedGraph release and initializes its submodules. The C++ driver is compiled
lazily on first use.

On Apple Silicon/macOS the driver uses the system Accelerate framework and
standard C++20 library, so no Homebrew numerical libraries are required. The
reference Linux/WSL build requires OpenBLAS, LAPACK/LAPACKE, `fmt`, and
AMD/SuiteSparse. On Debian/Ubuntu systems these can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y \
  g++ \
  libopenblas-dev \
  liblapack-dev \
  liblapacke-dev \
  libfmt-dev \
  libsuitesparse-dev
```

See `doc/user_guide/repulsive_layout.md` and `THIRD_PARTY_NOTICES.md` for the
full setup and the pinned upstream revision.

## Quick Start

Compute a Yamada polynomial directly from a crossing-free graph:

```python
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial

A = sp.Symbol("A")
theta = ThetaGraph(3)
compute_graph_yamada_polynomial(theta, A)
```

Protein crosslink scans accept local PDB/mmCIF files or RCSB IDs. The workflow
classifies physical crosslinks, extracts the embedded cyclic core, compares
canonical Yamada fingerprints after single/pair/subset perturbations, and can
condition embedding topology on an exact reference with identical abstract
connectivity:

```bash
uv run kg-protein-topology \
  examples/protein_topology/population_conditioned_recovered_v1.csv \
  results/protein_topology/population_conditioned_recovered_v1 \
  --rotation-samples 32 --max-crossings 40 --no-pairs \
  --exact-subsets none --conditioned-robustness \
  --null-replicates 20 --null-seed 2026 \
  --null-embedding-mode coordinate_preserving \
  --null-sampling-mode unique_disulfide_matchings \
  --repulsion-steps 100 --repulsion-max-time 10 \
  --repulsion-free-special-vertices \
  --repulsion-decimation-passes 16 \
  --repulsion-max-points-per-edge 32 --repulsion-fallback-only \
  --null-repulsion-fallback-steps 100 \
  --null-repulsion-fallback-max-time 10 \
  --null-repulsion-fallback-free-special-vertices \
  --null-repulsion-fallback-decimation-passes 16 \
  --null-repulsion-fallback-max-points-per-edge 32 \
  --repulsor-root external/Repulsor \
  --allow-repulsor-certificate-only
```

See `User_guide/applications/03_protein_applications.ipynb` for a runnable cached
example and `User_guide/benchmarks/07_protein_crosslink_topology.ipynb` for the
dataset protocol. The batch writes full state tables, proven or bounded minimum
retained fingerprint-generating sets, arbitrary-order conditioned cooperative
subsets, stable local-lasso groups, exact abstract-isomorphism groups, unique
null provenance, and guarded population statistics. The tracked manifests and
their selection caveats are in `examples/protein_topology/README.md`.
Fingerprint equality is not a proof of isotopy.

For embedded spatial graphs, project the graph to a planar diagram and compute
from the projection with the fewest crossings:

```python
import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial

graph = nx.MultiGraph()
graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
graph.add_node("v", pos=np.array([1.0, 0.0, 0.0]))
graph.add_edge(
    "u",
    "v",
    pts=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.25, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
)

A = sp.Symbol("A")
result = compute_yamada_polynomial(
    graph,
    A,
    return_result=True,
)
print(result.polynomial)
print(result.projection.num_crossings)
```

The non-Hermitian nodal-skeleton workflow is an application package:

```python
from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import hopf_link_bloch_vector
```

## Documentation

The public documentation is available at
<https://sarinstein-yan.github.io/KnottedGraph/>. It contains installation
notes, generic quick starts, application tutorials, API references, and
developer architecture notes.

Build it locally with:

```bash
uv run --group docs python -m sphinx -b html -W --keep-going doc doc/_build/html
```

The root README intentionally stays generic. Physics-specific runnable examples
live in `User_guide/applications/01_physics_applications.ipynb`; website figures
used by the current Sphinx pages live under `doc/assets/site_figures/`.
