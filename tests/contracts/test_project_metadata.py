"""Keep public installation metadata aligned with the imported package."""

from __future__ import annotations

from pathlib import Path
import tomllib

import knotted_graph


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_imported_version_matches_project_metadata():
    assert knotted_graph.__version__ == PYPROJECT["project"]["version"]


def test_all_extra_contains_every_documented_workflow_extra():
    extras = PYPROJECT["project"]["optional-dependencies"]
    workflow_extras = ("nodal", "surface", "viz", "repulsion", "notebook")
    expected = {
        requirement
        for extra in workflow_extras
        for requirement in extras[extra]
    }

    assert expected <= set(extras["all"])


def test_public_support_urls_are_declared():
    urls = PYPROJECT["project"]["urls"]

    assert urls["Documentation"].startswith("https://")
    assert urls["Issues"].endswith("/issues")
