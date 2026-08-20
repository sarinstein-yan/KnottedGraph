"""Sphinx configuration for KnottedGraph documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "KnottedGraph"
author = "Xianquan (Sarinstein) Yan, Hakan Akgün"
copyright = "2026, KnottedGraph contributors"
release = "0.2.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "KnottedGraph"
html_static_path = ["_static"]
html_extra_path = ["assets"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/sarinstein-yan/KnottedGraph",
    "navigation_depth": 3,
    "show_nav_level": 2,
    "show_toc_level": 2,
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3
nb_execution_mode = "off"

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "Bio",
    "kaleido",
    "minorminer",
    "plotly",
    "poly2graph",
    "pyvista",
    "skimage",
    "skimage.morphology",
    "tabulate",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
