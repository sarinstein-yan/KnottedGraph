# Repulsive Layout

<div class="kg-hero">
  <p class="kg-lead">The repulsive-layout workflow relaxes an embedded graph geometrically while preserving the intended graph topology. The Python package provides the workflow and topology checks; the optional Repulsor solver is an external C++ dependency.</p>
  <div class="kg-link-row">
    <a href="../../User_guide/01_getting_started.ipynb">Open Getting Started</a>
    <a href="../../User_guide/applications/03_protein_applications.ipynb">Open Protein Applications</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/repulsive_curves.png" alt="Repulsive curves workflow">
</div>

## Installation

Install the Python-side optional dependencies with

```bash
pip install "knotted_graph[repulsion]"
```

This does **not** vendor or install the external Repulsor C++ source.

For a repository checkout, prepare the exact Repulsor revision used by the
KnottedGraph release with

```bash
python scripts/bootstrap_repulsion.py
export REPULSOR_ROOT="$PWD/external/Repulsor"
```

The bootstrap:

1. installs `.[repulsion]` unless `--skip-python-install` is supplied;
2. clones Repulsor when the checkout is absent;
3. checks out the revision pinned in `scripts/bootstrap_repulsion.py`;
4. initializes all Repulsor submodules; and
5. refuses to silently change an existing checkout that is at another revision.

To intentionally use another upstream revision:

```bash
python scripts/bootstrap_repulsion.py \
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

## Reproducibility check

Before producing paper data, record:

```bash
git -C "$REPULSOR_ROOT" rev-parse HEAD
python -c "import knotted_graph; print(knotted_graph.__file__)"
```

The Repulsor commit should match the pin documented in
`THIRD_PARTY_NOTICES.md` unless an override was intentionally used.
