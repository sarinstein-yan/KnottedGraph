# Task 2 publication figures

This package is the current, deliberately small figure-generation surface for
the paper. It contains only the final composition code for:

- Main: the framed 2-by-4 input overview;
- Figure S1: the 3-by-5 nonzero Yamada examples, with pairwise-distinct
  polynomials in Computer Modern notation;
- Figure S2: the 3-by-4 skeletonization examples without Yamada footers.

Historical renderers, scheduler files, job logs, generated figures, and large
scientific caches are not part of this package.

## Accepted panel bundle

The three compositors use publication-approved panel PNGs as external inputs.
Their relative paths, titles, order, and SHA-256 checksums are locked in
`specs.py`. The default bundle root is:

```text
examples/input_gallery/figures/
```

That directory is generated data and is intentionally not stored in Git. A
different bundle location can be supplied with `--asset-root`. Every input is
verified before any output is written; a missing or changed panel fails the
build with its expected and actual digest. The bundle therefore remains a
separate paper artifact; this repository does not silently download an
unversioned dataset.

## Usage

Install the optional plotting dependencies:

```bash
python -m pip install -e '.[figures]'
```

Run the commands below from the repository root. The `examples` tree is kept
outside the library wheel, so this is a repository example API rather than a
public `knotted_graph` package API.

Verify the accepted inputs:

```bash
python -m examples.input_gallery.task2_figures verify \
  --asset-root /path/to/accepted/panels
```

Build one figure or all three:

```bash
python -m examples.input_gallery.task2_figures main
python -m examples.input_gallery.task2_figures s1
python -m examples.input_gallery.task2_figures s2
python -m examples.input_gallery.task2_figures all
```

Use `--output-dir` to select the destination. By default, PNG, SVG, PDF, and a
small JSON provenance record are written to the ignored local `_build/`
directory. No caption or figure number is embedded in the images.

For compatibility with the accepted manuscript asset mapping, the S2 source
stem remains `appendix_s3_skeletonization_beyond_yamada_v9`; the command target,
Python function, provenance key, and manuscript label are all `s2`.

The optional dependency versions are pinned to the accepted rendering
environment. Panel-input hashes are strict. PNG pixel identity is checked when
the accepted release images are present; SVG/PDF byte identity is not promised
across a deliberately changed renderer environment.

## Python API inside a repository checkout

```python
from pathlib import Path
from examples.input_gallery.task2_figures import render_main, render_s1, render_s2

render_main(asset_root=Path("accepted-panels"), output_dir=Path("build"))
render_s1(asset_root=Path("accepted-panels"), output_dir=Path("build"))
render_s2(asset_root=Path("accepted-panels"), output_dir=Path("build"))
```

The code is scheduler-independent. Resource selection and job submission stay
outside the repository. On a managed cluster, `verify` is a lightweight check;
run publication-resolution rendering through the site's approved compute-node
workflow rather than on a login node.
