# KnottedGraph Collaborator Handoff

This repository contains the current KnottedGraph library, the reorganized user-guide notebooks, and the Sphinx website source. Development branches should preserve the public interfaces and validate changes against the repository test and notebook workflows before integration.

## Where to Continue

- Main notebook entry point: `User_guide/00_user_guide.ipynb`
- Core workflows: `User_guide/01_getting_started.ipynb`, `User_guide/02_core_workflows.ipynb`, `User_guide/03_advanced_and_reproduction.ipynb`
- Applications:
  - `User_guide/applications/01_physics_applications.ipynb`
  - `User_guide/applications/02_mathematics_applications.ipynb`
  - `User_guide/applications/03_protein_applications.ipynb`
  - `User_guide/applications/04_analytic_knot_fields.ipynb`
  - `User_guide/applications/05_yamada_formula_discovery.ipynb`
  - `User_guide/applications/06_hamiltonian_yamada_phase_maps.ipynb`
- Correctness/performance notebooks:
  - `User_guide/benchmarks/01_yamada_sanity_checks.ipynb`
  - `User_guide/benchmarks/02_application_regression_checks.ipynb`
  - `User_guide/benchmarks/03_knottedgraph_vs_topoly_scaling.ipynb`
  - `User_guide/benchmarks/04_thick_handlebody_validation.ipynb`
- Website source: `doc/`

## Generated Website Output

`doc/_build/` and `site_preview/` are generated Sphinx build directories and are ignored by Git. This avoids committing duplicate HTML, static assets, and Sphinx caches. The reproducible website source is `doc/` together with the tracked figure assets under `doc/assets/`.

## Rebuild the Local Website Preview

From the repository root:

```bash
uv sync --all-extras --group docs
uv run --group docs python -m sphinx -b html -W --keep-going doc site_preview
open site_preview/index.html
```

If an existing virtual environment is already installed:

```bash
.venv/bin/python -m sphinx -b html -W --keep-going doc site_preview
open site_preview/index.html
```

The built preview is local. Publication to GitHub Pages is handled by `.github/workflows/docs.yml` on the configured deployment branch.

## Validation Before Integration

Run the complete test suite and the repository consistency audit before accepting a branch:

```bash
uv sync --all-extras --group dev --group docs
uv run --no-sync pytest -q
uv run --no-sync python dev/check_repository_consistency.py
```

For changes affecting application outputs, also execute `User_guide/benchmarks/02_application_regression_checks.ipynb` through `dev/execute_notebook.py`.
