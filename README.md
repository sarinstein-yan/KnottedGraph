# `knotted_graph`

![pre-alpha](https://img.shields.io/badge/status-pre--alpha-red?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/knotted_graph)](https://pypi.org/project/knotted_graph/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://sarinstein-yan.github.io/KnottedGraph/)

`knotted_graph` is a computational package for embedded spatial graphs,
planar-diagram construction, and graph polynomial invariants. It provides a
generic core for pure mathematical and computational workflows, with optional
application packages built on top.

<p align="center">
  <img src="https://raw.githubusercontent.com/sarinstein-yan/KnottedGraph/Latest_Workplace/assets/paper/architecture.svg" width="780" alt="KnottedGraph architecture">
</p>

> [!IMPORTANT]
> PyPI currently provides the legacy `knotted_graph` 0.1.2 package. Its
> nodal-only package layout is not compatible with the 0.2.0 development API
> documented in this repository. Until 0.2.0 is released, install the current
> API from the `Latest_Workplace` branch as shown below.

## Installation

KnottedGraph requires Python 3.11 or newer. The recommended development setup
uses [uv](https://docs.astral.sh/uv/):

```bash
git clone --branch Latest_Workplace --single-branch \
  https://github.com/sarinstein-yan/KnottedGraph.git
cd KnottedGraph
uv sync --group dev
```

`uv sync --group dev` installs the base library and its test tools, but no
optional feature extras. To install every Python extra as well, use:

```bash
uv sync --group dev --all-extras
```

The extras can also be selected individually:

| Extra | Adds support for |
| --- | --- |
| `nodal` | Non-Hermitian nodal skeleton extraction |
| `surface` | Surface-mesh workflows backed by PyVista |
| `viz` | Plotly and publication-image export tools |
| `repulsion` | Python-side repulsive-layout I/O and visualization |
| `notebook` | JupyterLab |
| `all` | All optional Python workflows, plus `igraph` |

For example, install only the surface and notebook workflows with:

```bash
uv sync --group dev --extra surface --extra notebook
```

The `repulsion` extra does not install the separate C++ Repulsor solver or its
native libraries. See the
[repulsive-layout setup](doc/user_guide/repulsive_layout.md) before using that
backend. The complete installation matrix and a plain-`pip` source-install
alternative are in [the installation guide](doc/installation.md).
If setup or the smoke test fails, start with the
[troubleshooting guide](doc/troubleshooting.md).

## Quick Start

The shortest deterministic example computes a nonzero Yamada polynomial for
the crossing-free theta graph. We use `Y` consistently as the polynomial
variable:

```python
import sympy as sp

from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

Y = sp.Symbol("Y")
theta = ThetaGraph(3)
polynomial = sp.expand(compute_yamada_polynomial_recursive(theta, Y))
print(f"Upsilon(Theta_3; Y) = {polynomial}")
```

Expected output:

```text
Upsilon(Theta_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
```

This value is nonzero because the three parallel edges contain no bridge. A
graph with a bridge has zero Yamada polynomial, so a single open edge is not a
useful installation smoke test.

Run the complete example to evaluate both the abstract graph and a planar 3D
embedding of the same theta graph:

```bash
uv run python examples/quickstart.py
```

The embedded calculation fixes the projection direction for reproducibility,
uses one worker (`n_jobs=1`), verifies zero projected crossings, and checks that
its result equals the abstract calculation. See the
[annotated quick start](doc/quickstart.md) for the full embedded code.

The non-Hermitian nodal-skeleton workflow is an optional application package:

```python
from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import hopf_link_bloch_vector
```

Install the `nodal` extra before using these imports.

## Repulsive-layout native dependency

For a reproducible source checkout, use the repository bootstrap:

```bash
uv sync --extra repulsion
uv run python scripts/bootstrap_repulsion.py --skip-python-install
export REPULSOR_ROOT="$PWD/external/Repulsor"
```

Inside an activated conventional venv that provides pip, the bootstrap can
instead install the Python extra itself with `python scripts/bootstrap_repulsion.py`.

The bootstrap checks out the exact Repulsor revision pinned for this
KnottedGraph release and initializes its submodules. The C++ driver is compiled
lazily on first use.

The reference Linux/WSL build requires a C++20 compiler and the native libraries
linked by the driver: OpenBLAS, LAPACK/LAPACKE, `fmt`, and AMD/SuiteSparse. On
Debian/Ubuntu systems these can be installed with packages such as:

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

## Documentation

The public documentation is available at
<https://sarinstein-yan.github.io/KnottedGraph/>. It contains installation
notes, generic quick starts, application tutorials, API references, and
developer architecture notes. The
[feature-status and workflow matrix](doc/feature_status.md) maps starting data
and goals to the currently documented entry points, extras, return objects, and
runtime boundaries.

Build it locally with:

```bash
uv run --group docs python -m sphinx -b html -W --keep-going doc doc/_build/html
```

The root README intentionally stays generic. Physics-specific guidance lives
under `doc/applications/`; the full executable walkthrough is
[`User_guide/applications/01_physics_applications.ipynb`](User_guide/applications/01_physics_applications.ipynb),
and publication figures live under `assets/paper/`.
