# Repulsive Layout

<div class="kg-hero">
  <p class="kg-lead">The repulsive-layout workflow relaxes an embedded graph geometrically while preserving the intended graph topology. The Python package provides the workflow and topology checks; the optional Repulsor solver is an external C++ dependency.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/01_getting_started.ipynb">Open Getting Started</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/applications/03_protein_applications.ipynb">Open Protein Applications</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/repulsive_curves.png" alt="Repulsive curves workflow">
</div>

## Installation

From a current source checkout, install the Python-side optional dependencies
with either uv or editable pip:

```bash
uv sync --extra repulsion
# or: python -m pip install -e ".[repulsion]"
```

The commands below show the uv route. In an activated pip-based venv, omit the
leading `uv run`.

This does **not** vendor or install the external Repulsor C++ source.

For a repository checkout, prepare the exact Repulsor revision used by the
KnottedGraph release with

```bash
uv run python scripts/bootstrap_repulsion.py --skip-python-install
export REPULSOR_ROOT="$PWD/external/Repulsor"
```

The bootstrap:

1. installs `.[repulsion]` unless `--skip-python-install` is supplied;
2. clones Repulsor when the checkout is absent;
3. checks out the revision pinned in `scripts/bootstrap_repulsion.py`;
4. initializes all Repulsor submodules; and
5. refuses to silently change an existing checkout that is at another revision.

The command above uses `--skip-python-install` because the preceding `uv sync`
already installed the extra. In an activated pip-based venv, omit that flag if
you want the bootstrap to run `pip install -e ".[repulsion]"` for you.

To intentionally use another upstream revision:

```bash
uv run python scripts/bootstrap_repulsion.py \
  --skip-python-install \
  --repulsor-ref <commit-or-tag>
```

Record that override in any reproducibility report.

## Native build requirements

The reference Linux/WSL driver build uses C++20 and links against:

- OpenBLAS;
- LAPACK;
- LAPACKE;
- `fmt`;
- AMD/SuiteSparse; and
- pthreads.

For Debian/Ubuntu:

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

The C++ driver is compiled lazily by
`knotted_graph.layout.repulsive.driver.build_driver()`.

## Run a built-in example

Inspect the available options first:

```bash
uv run kg-repulsive-layout --help
uv run kg-repulsive-layout examples --help
```

After setting `REPULSOR_ROOT`, run a short topology-checked example with one
solver thread:

```bash
uv run kg-repulsive-layout examples \
  --sample 1aoc \
  --steps 10 \
  --threads 1 \
  --out build/repulsive_layout/1aoc_demo
```

The command prints a JSON summary. The output directory contains metadata and,
unless `--no-render` is supplied, HTML views of the initial and final embedded
graphs. Increase the number of steps only after the short run builds and
completes successfully. Use `--verify-topology` when you also want the
independent saved-step verifier; it retains additional intermediate files.

The native solver does not choose scheduler resources for you. On a shared
system, keep `--threads` within the CPU allocation granted to the process.

## Reproducibility check

Before producing paper data, record:

```bash
git -C "$REPULSOR_ROOT" rev-parse HEAD
uv run python -c "import knotted_graph; print(knotted_graph.__file__)"
```

The Repulsor commit should match the pin documented in
`THIRD_PARTY_NOTICES.md` unless an override was intentionally used.
