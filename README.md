# KnottedGraph

![pre-alpha](https://img.shields.io/badge/status-pre--alpha-red?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/knotted_graph)](https://pypi.org/project/knotted_graph/)
[![Docs](https://img.shields.io/badge/docs-published%20snapshot-blue)](https://sarinstein-yan.github.io/KnottedGraph/)

**KnottedGraph turns geometric, graph, knot-field, and Hamiltonian-derived
objects into inspectable spatial graphs, planar diagrams, and Yamada
polynomials.** It separates the reusable graph/projection/invariant core from
optional scientific applications such as nodal skeletons, analytic knot
fields, material surfaces, and repulsive layouts.

<p align="center">
  <img src="assets/paper/architecture.svg" width="780" alt="KnottedGraph architecture from geometric input to graph, projection, and invariant">
</p>

## Read this version note first

KnottedGraph is pre-alpha and the public distribution channels are not yet
synchronized:

| Channel | Current state |
| --- | --- |
| This review branch | 0.2.0 development API on `codex/arbitrary-knot-user-integration` |
| PyPI | legacy 0.1.2 nodal-only layout; incompatible with the examples below |
| Published website | an earlier documentation snapshot; some pages from this branch are not deployed yet |

Until 0.2 is released and the documentation branch is merged, reviewers and
contributors should use the source checkout below. Do not use an unpinned
`pip install knotted_graph` for these examples.

## Start here

Choose the row that matches the object you already have:

| I have / I want | First entry point | Continue to |
| --- | --- | --- |
| No existing data; I want a five-minute test | [`examples/quickstart.py`](examples/quickstart.py) | [Quick Start](doc/quickstart.md) |
| Ordered coordinates or CSV/DAT/JSON/NPY/TSV/TXT/XYZ | `knotted_graph.inputs.from_coordinate_chain` | [Input handling](doc/user_guide/input_adapters.md) |
| A PDB/mmCIF backbone | `from_pdb_backbone` / `from_mmcif_backbone` | [Input handling](doc/user_guide/input_adapters.md) |
| A GRO snapshot or first LAMMPS frame | `from_gromacs_gro` / `from_lammps_dump` | [Input handling](doc/user_guide/input_adapters.md) |
| Node and edge CSV files for a spatial graph | `from_spatial_graph_csv` | [Input handling](doc/user_guide/input_adapters.md) |
| A named knot/link, torus type, or braid word | `knotted_graph.inputs.KnotFunction` | [Analytic knot fields](doc/applications/analytic_knot_fields.md) |
| An embedded `networkx.MultiGraph` | `knotted_graph.core.ensure_embedding` | [Workflow overview](doc/user_guide/workflow_overview.md) |
| A nodal/material Hamiltonian in memory | application APIs under `knotted_graph.applications` | [Application routes](doc/applications/index.md) |
| A native Repulsor layout workflow | `knotted_graph.layout.repulsive` | [Repulsive layout](doc/user_guide/repulsive_layout.md) |

The complete availability, extra, return-type, and scaling matrix is in
[`doc/feature_status.md`](doc/feature_status.md).

## The core mental model

Most graph-returning routes meet at one data contract:

```text
source data
    -> input adapter / application extractor
    -> networkx.MultiGraph(node pos, edge pts)
    -> embedding validation and cleanup
    -> regular planar projection
    -> PD-code data
    -> Yamada polynomial + projection provenance
```

- Every graph node has a finite three-vector `pos`.
- Every graph edge has sampled 3-D geometry in `pts`; its first and last
  points agree with the endpoint node positions.
- A crossing visible in a 2-D projection is not automatically a graph vertex.
- High-level input loaders return a result object with `.graph` (or `.mesh` for
  surface input), `.issues`, identifiers, and source metadata.
- Closure, chain/model selection, coordinate units, and projection choice are
  scientific decisions and should be recorded explicitly.

## Install the current review build

KnottedGraph requires Python 3.11 or newer. The recommended environment uses
[uv](https://docs.astral.sh/uv/):

```bash
git clone --branch codex/arbitrary-knot-user-integration --single-branch \
  https://github.com/sarinstein-yan/KnottedGraph.git
cd KnottedGraph
uv sync --group dev
uv run python examples/quickstart.py
```

The base installation covers spatial-graph data structures, embedding tools,
projection/PD-code construction, and Yamada evaluation. Install optional
features only when needed:

```bash
uv sync --group dev --extra knot-fields --extra notebook
# or, for the full development/test environment:
uv sync --group dev --all-extras
```

| Extra | Adds |
| --- | --- |
| `knot-fields` | Sampled analytic knot-field level sets and graph extraction |
| `nodal` | Nodal skeleton and Hamiltonian workflows |
| `surface` | PyVista surface-mesh loading |
| `viz` | Plotly and publication-image export tools |
| `repulsion` | Python helpers for the separately installed Repulsor backend |
| `notebook` | JupyterLab |
| `benchmark` | Topoly comparison dependency |
| `all` | All optional Python workflows |

See the [installation guide](doc/installation.md) for pip-based source
installation, native dependencies, and environment verification. See
[troubleshooting](doc/troubleshooting.md) if an import, optional extra, native
backend, projection, or headless-rendering step fails.

## Five-minute Quick Start

The smallest deterministic nonzero example is the crossing-free theta graph:

```python
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial

Y = sp.Symbol("Y")
theta = ThetaGraph(3)
polynomial = sp.expand(compute_graph_yamada_polynomial(theta, Y))
print(f"Upsilon(Theta_3; Y) = {polynomial}")
```

Expected output:

```text
Upsilon(Theta_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
```

Run the maintained example to compare this result with an embedded graph and a
fixed regular projection:

```bash
uv run python examples/quickstart.py
```

The example deliberately uses `n_jobs=1`, reports the selected crossing count,
and verifies that the abstract and embedded calculations agree.

## What the package currently does not claim

The presence of a format in a research figure does not imply a public parser.
There is currently no generic public adapter for GraphML, SWC, arbitrary graph
JSON, NPZ scalar/vector volumes, Hamiltonian files, or general edge lists.
Hamiltonian and field workflows presently start from in-memory objects or
application-specific conversion code. Surface loading returns a
`PyVista.PolyData`; it does not automatically choose a scientifically valid
skeletonization route.

Yamada state evaluation can grow exponentially with projected crossing count.
Inspect the selected projection and use explicit worker/resource settings before
running large calculations.

## Documentation map

- [Installation](doc/installation.md): versions, environments, extras, native boundaries.
- [Quick Start](doc/quickstart.md): copyable graph-to-invariant example with expected output.
- [Input handling](doc/user_guide/input_adapters.md): supported formats, result objects, closure, units, and errors.
- [Workflow overview](doc/user_guide/workflow_overview.md): how the stages fit together.
- [Projection and Yamada](doc/user_guide/projection_yamada.md): projections, PD codes, provenance, and scaling.
- [Feature-status matrix](doc/feature_status.md): public/application/external routes and return types.
- [Application tutorials](doc/applications/index.md): mathematical, physical, knot-field, and reproduction workflows.
- [API reference](doc/api/index.md): public calls grouped by subsystem.
- [Troubleshooting](doc/troubleshooting.md): symptom-based recovery.

The maintained notebooks live under [`User_guide/`](User_guide/). Introductory
notebooks are distinct from publication-reproduction and benchmark notebooks;
the latter may require native backends, cached data, and substantially more
time or memory.

Build the website locally with warnings treated as errors:

```bash
uv run --group docs python -m sphinx -b html -W --keep-going doc doc/_build/html
```

For questions or reproducible bug reports, use the
[GitHub issue tracker](https://github.com/sarinstein-yan/KnottedGraph/issues)
and include the package version, commit, Python version, optional extras, and
native-backend status.
