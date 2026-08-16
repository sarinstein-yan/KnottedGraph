# Latest_Workplace Collaborator Handoff

This branch contains the current working version of the KnottedGraph library, the reorganized user-guide notebooks, and the Sphinx website source.

## Where to Continue

- Main notebook entry point: `User_guide/00_user_guide.ipynb`
- Core workflows: `User_guide/01_getting_started.ipynb`, `User_guide/02_core_workflows.ipynb`, `User_guide/03_advanced_and_reproduction.ipynb`
- Applications:
  - `User_guide/applications/01_physics_applications.ipynb`
  - `User_guide/applications/02_mathematics_applications.ipynb`
  - `User_guide/applications/03_protein_applications.ipynb`
- Website source: `doc/`

## Why `site_preview/` May Not Appear On GitHub

`site_preview/` is a generated Sphinx build directory and is ignored by Git. This avoids committing a large duplicate copy of the documentation, static assets, and Sphinx cache files. The reproducible source is `doc/` plus the figure assets under `doc/assets/`.

## Rebuild The Local Website Preview

From the repository root:

```bash
uv sync --all-extras --group docs
uv run --group docs python -m sphinx -b html doc site_preview
open site_preview/index.html
```

If an existing virtual environment is already installed:

```bash
.venv/bin/python -m sphinx -b html doc site_preview
open site_preview/index.html
```

The built preview is local. To publish the website at the GitHub Pages URL, deploy the generated HTML through the repository's Pages workflow or `gh-pages` branch.
