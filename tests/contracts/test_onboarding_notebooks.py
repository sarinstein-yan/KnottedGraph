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
PROFILE_EXPECTATIONS = {
    "01_getting_started.ipynb": {
        "wall_seconds": 22.24,
        "peak_rss_kb": 223668,
        "display_time": "22.24 s",
        "display_memory": "223,668 KB",
        "extras": {"viz", "notebook"},
        "mode": "default",
    },
    "02_core_workflows.ipynb": {
        "wall_seconds": 23.1,
        "peak_rss_kb": 799376,
        "display_time": "23.10 s",
        "display_memory": "799,376 KB",
        "extras": {"nodal", "viz", "notebook"},
        "mode": "quick (96^3)",
    },
    "03_advanced_and_reproduction.ipynb": {
        "wall_seconds": 46.75,
        "peak_rss_kb": 847852,
        "display_time": "46.75 s",
        "display_memory": "847,852 KB",
        "extras": {"nodal", "viz", "notebook"},
        "mode": "quick (96^3)",
    },
}


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


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_beginner_notebook_profiles_are_visible_and_match_measured_evidence(path):
    notebook = _load_notebook(path)
    profile = notebook["metadata"]["knotted_graph"]["notebook_profile"]
    expected = PROFILE_EXPECTATIONS[path.name]
    visible_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert profile["schema_version"] == 1
    assert profile["role"]
    assert profile["audience"]
    assert profile["environment"]
    assert set(profile["extras"]) == expected["extras"]
    assert profile["network"] == "offline after installation"
    assert profile["outputs"]
    assert profile["compute_policy"]

    measurement = profile["measurement"]
    assert measurement == {
        "allocation_cpus": 4,
        "allocation_memory_gb": 16,
        "environment": "Vanda Linux / Python 3.12",
        "job_id": "1335345",
        "mode": expected["mode"],
        "peak_rss_kb": expected["peak_rss_kb"],
        "status": "measured",
        "wall_seconds": expected["wall_seconds"],
        "walltime_limit": "00:30:00",
    }
    assert "**Notebook profile**" in visible_text
    assert expected["display_time"] in visible_text
    assert expected["display_memory"] in visible_text
    assert "PBS 1335345" in visible_text
    assert "allocated 4 CPUs/16 GB" in visible_text
    assert "walltime limit 00:30:00" in visible_text
    assert "not a minimum requirement or a performance guarantee" in visible_text

    project_extras = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["optional-dependencies"]
    assert set(profile["extras"]) <= set(project_extras)

    if path.name != "01_getting_started.ipynb":
        assert profile["modes"]["paper (300^3)"] == "not_profiled"
