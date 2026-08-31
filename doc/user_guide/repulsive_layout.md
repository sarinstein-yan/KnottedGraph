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

On Apple Silicon/macOS, the driver uses the system Accelerate framework and
standard C++20 `std::format`; no Homebrew numerical libraries are required.
The compiler must support C++20 and the Clang matrix extension.

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

## Protein topology gate

Protein scans use Repulsor through
`knotted_graph.applications.protein.relax_and_analyze_crosslinks`. The
crosslink-supported bridgeless cyclic core is relaxed once. Graph junctions are
pinned by default and move only with the explicit free-special-vertices option.
Safe pre/post-decimation keeps at least three points per edge and accepts a
shortcut only when its swept triangle stays clear of every non-adjacent
segment. Edge-deletion scans proceed only after a matching before/after
fingerprint or an explicitly accepted safe-step certificate.

For an initial diagram that exceeds the exact crossing cap, strict fingerprint
comparison is unavailable. An explicit
`--allow-repulsor-certificate-only` opt-in permits the solver's valid swept
safe-step certificate to gate the relaxed analysis instead. This mode is labeled
`certificate_only`; it must not be reported as a before/after fingerprint match.

Native build failures, exact-fingerprint failures, and fingerprint mismatches are
returned as structured statuses and are retained by the batch output. A mismatch
must not be silently treated as an acceptable geometry cleanup.

```bash
uv run kg-protein-topology proteins.csv results/protein_topology/relaxed \
  --repulsion-steps 100 --repulsion-fallback-only \
  --repulsion-free-special-vertices \
  --repulsion-max-points-per-edge 32 \
  --repulsion-decimation-passes 16 \
  --repulsor-root external/Repulsor \
  --allow-repulsor-certificate-only \
  --rotation-samples 32 --max-crossings 40
```

The analogous `--null-repulsion-fallback-*` options affect only
coordinate-preserving null graphs whose original exact baseline exceeds the
crossing cap. `run_config.json` records every fallback, vertex-motion,
pre-decimation, and post-decimation setting.

## Reproducibility check

Before producing paper data, record:

```bash
git -C "$REPULSOR_ROOT" rev-parse HEAD
python -c "import knotted_graph; print(knotted_graph.__file__)"
```

The Repulsor commit should match the pin documented in
`THIRD_PARTY_NOTICES.md` unless an override was intentionally used.
