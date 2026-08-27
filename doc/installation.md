# Installation

## Choose the API version first

KnottedGraph requires Python 3.11 or newer. This documentation describes the
0.2.0 development API on the repository's `Latest_Workplace` branch.

PyPI currently provides version 0.1.2. That release contains the legacy,
nodal-only package layout and is not compatible with the imports or examples in
these 0.2.0 development documents. Until 0.2.0 is published, install the current
API from GitHub rather than running an unpinned `pip install knotted_graph`.

## Recommended source setup with uv

Install [uv](https://docs.astral.sh/uv/), then clone the documented branch:

```bash
git clone --branch Latest_Workplace --single-branch \
  https://github.com/sarinstein-yan/KnottedGraph.git
cd KnottedGraph
```

For the generic library and the test tools, run:

```bash
uv sync --group dev
```

This command installs the base dependencies and the `dev` dependency group. It
does **not** install optional feature extras. If you need every optional Python
workflow, use:

```bash
uv sync --group dev --all-extras
```

To add only selected workflows, name each extra explicitly. For example:

```bash
uv sync --group dev --extra surface --extra notebook
```

Use `uv run` so commands execute inside the managed environment:

```bash
uv run python examples/quickstart.py
```

The Quick Start is the base-install smoke test. The complete test suite imports
optional workflows; install all extras before running it:

```bash
uv sync --group dev --all-extras
uv run pytest
```

## Source setup with pip

If uv is unavailable, create and activate a Python 3.11-or-newer virtual
environment, clone `Latest_Workplace` as above, and install the checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Add one or more extras with standard pip syntax:

```bash
python -m pip install -e ".[surface,notebook]"
```

Use `python -m pip install -e ".[all]"` only when all optional Python workflows
are required.

## Optional-feature matrix

The base installation contains the graph data structures, embedded-graph
utilities, projection and PD-code pipeline, Yamada polynomial backends, and
their required numerical dependencies.

| Install target | Additional capability | Important boundary |
| --- | --- | --- |
| `nodal` | Non-Hermitian nodal skeleton extraction | Adds `poly2graph`, scikit-image, PyVista, minorminer, and tabulate |
| `surface` | Surface-mesh workflows | Adds PyVista |
| `viz` | Interactive plots and publication-image export | Adds Plotly, Kaleido, PDF/image conversion packages |
| `repulsion` | Python-side repulsive-layout I/O and visualization | Does not provide the native C++ solver |
| `notebook` | Interactive notebooks | Adds JupyterLab |
| `all` | Every optional Python workflow | Also adds `igraph`; native system dependencies remain separate |

The `dev` and `docs` names are dependency groups, not extras:

```bash
uv sync --group dev       # test and lint tools
uv sync --group docs      # documentation build tools
```

Groups and extras are independent. For example, a documentation environment
with every optional feature is:

```bash
uv sync --group docs --all-extras
```

## Repulsive-layout native dependency

The `repulsion` extra supplies Python packages only. The Repulsor C++ source and
the system libraries used to build it are intentionally not vendored in the
Python wheel. From a source checkout, bootstrap the pinned upstream revision
with:

```bash
uv sync --extra repulsion
uv run python scripts/bootstrap_repulsion.py --skip-python-install
export REPULSOR_ROOT="$PWD/external/Repulsor"
```

The reference Linux/WSL build requires a C++20 compiler, OpenBLAS,
LAPACK/LAPACKE, `fmt`, and AMD/SuiteSparse. Continue with the
{doc}`user_guide/repulsive_layout` guide before running that backend.

## Verify the installation

The repository includes a deterministic smoke test that exercises both the
crossing-free core evaluator and the embedded-graph projection pipeline:

```bash
uv run python examples/quickstart.py
```

The final three lines should be:

```text
Abstract Upsilon(Theta_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
Embedded Upsilon(Theta_3; Y) = -Y**2 - Y - 2 - 1/Y - 1/Y**2
Selected projection crossings = 0
```

Continue with the {doc}`quickstart` after these results appear.
If they do not, use the symptom-based checks in {doc}`troubleshooting`.
