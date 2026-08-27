"""Lightweight contracts for the three primary tutorial notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROJECT_VERSION = tomllib.loads(
    (ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
NOTEBOOKS = [
    ROOT / "User_guide" / "01_getting_started.ipynb",
    ROOT / "User_guide" / "02_core_workflows.ipynb",
    ROOT / "User_guide" / "03_advanced_and_reproduction.ipynb",
]
PUBLIC_NOTEBOOKS = NOTEBOOKS + [
    ROOT / "User_guide" / "00_user_guide.ipynb",
    ROOT / "User_guide" / "applications" / "03_protein_applications.ipynb",
]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _setup_source(path: Path) -> str:
    notebook = _load_notebook(path)
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "installation_mode" in "".join(cell["source"])
    ]
    assert len(matches) == 1, f"expected one setup cell in {path}"
    return matches[0]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_beginner_notebook_code_is_valid_and_outputs_are_clean(path):
    notebook = _load_notebook(path)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_beginner_notebook_setup_supports_source_checkout(path):
    completed = subprocess.run(
        [sys.executable, "-c", _setup_source(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "source checkout:" in completed.stdout


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_beginner_notebook_setup_supports_installed_package(path, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", _setup_source(path)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "installed package" in completed.stdout
    assert f"version {PROJECT_VERSION}" in completed.stdout


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_beginner_notebook_setup_checks_current_public_api(path):
    source = _setup_source(path)

    assert '"knotted_graph.core"' in source
    assert '"knotted_graph.projection"' in source
    assert "missing the current" in source


def test_public_beginner_notebooks_have_no_internal_ownership_placeholders():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_NOTEBOOKS)
    for phrase in ("To be filled", "To be completed", "Kehan", "Zhaoyun"):
        assert phrase not in text


def test_compute_heavy_notebooks_default_to_quick_mode():
    for name in ("02_core_workflows.ipynb", "03_advanced_and_reproduction.ipynb"):
        text = (ROOT / "User_guide" / name).read_text(encoding="utf-8")
        assert 'RUN_MODE = \\"quick\\"' in text
        assert 'if RUN_MODE not in {\\"quick\\", \\"paper\\"}' in text
