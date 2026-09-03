from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE = ROOT / "User_guide"


def _load_maintenance_module(name: str, relative_path: str) -> ModuleType:
    """Load an un-packaged ``dev/`` script without relying on ``sys.path``."""

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load maintenance module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protected_audit = _load_maintenance_module(
    "kg_check_protected_final_audit",
    "dev/check_protected_final_audit.py",
)
repository_consistency = _load_maintenance_module(
    "kg_check_repository_consistency",
    "dev/check_repository_consistency.py",
)
notebook_portability = _load_maintenance_module(
    "kg_validate_notebook_portability",
    "dev/validate_notebook_portability.py",
)

ALLOWED_BOOTSTRAP_CELLS = protected_audit.ALLOWED_BOOTSTRAP_CELLS
EXPECTED_BENCHMARK_FINGERPRINTS = protected_audit.EXPECTED_BENCHMARK_FINGERPRINTS
EXPECTED_GIT_OBJECTS = protected_audit.EXPECTED_GIT_OBJECTS
_benchmark_fingerprint = protected_audit._benchmark_fingerprint
_without_bootstrap_cells = protected_audit._without_bootstrap_cells
_missing_optional_dependency = repository_consistency._missing_optional_dependency
_module_declares_symbol = repository_consistency._module_declares_symbol
validate_notebook = notebook_portability.validate_notebook


def test_notebook_normalizer_is_idempotent() -> None:
    subprocess.run(
        [sys.executable, "dev/normalize_notebook_environments.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_all_user_guide_notebooks_pass_portability_contract() -> None:
    failures = {
        path.relative_to(ROOT).as_posix(): validate_notebook(path)
        for path in sorted(USER_GUIDE.rglob("*.ipynb"))
        if validate_notebook(path)
    }
    assert failures == {}


def test_formula_discovery_is_browsable_outside_a_named_branch() -> None:
    path = USER_GUIDE / "applications" / "05_yamada_formula_discovery.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "EXPECTED_BRANCH" not in source
    assert "STRICT_PUBLICATION_REGENERATION" in source
    assert r"\$$0.35em]" not in source
    assert r"\\[0.35em]" in source


def test_optional_lazy_exports_do_not_trigger_module_getattr() -> None:
    module = ModuleType("example")
    module.__all__ = ["OptionalThing"]

    def fail_if_called(name: str):
        raise AssertionError(f"module __getattr__ was called for {name}")

    module.__getattr__ = fail_if_called
    assert _module_declares_symbol(module, "OptionalThing")
    assert not _module_declares_symbol(module, "MissingThing")


def test_optional_dependency_classifier_covers_python_and_native_extras() -> None:
    assert _missing_optional_dependency(ModuleNotFoundError("pyvista", name="pyvista"))
    assert _missing_optional_dependency(
        ModuleNotFoundError(
            "native",
            name="knotted_graph.invariants.yamada._yamada_native",
        )
    )
    assert not _missing_optional_dependency(ModuleNotFoundError("typo", name="typo"))


def test_protected_notebook_comparison_excludes_only_allowlisted_cell() -> None:
    notebook = {
        "cells": [
            {"id": "setup", "cell_type": "code", "source": ["old setup\n"]},
            {"id": "science", "cell_type": "code", "source": ["result = 1\n"]},
        ],
        "nbformat": 4,
    }
    changed_setup = json.loads(json.dumps(notebook))
    changed_setup["cells"][0]["source"] = ["portable setup\n"]
    assert _without_bootstrap_cells(notebook, {"setup"}) == _without_bootstrap_cells(
        changed_setup, {"setup"}
    )

    changed_science = json.loads(json.dumps(notebook))
    changed_science["cells"][1]["source"] = ["result = 2\n"]
    assert _without_bootstrap_cells(notebook, {"setup"}) != _without_bootstrap_cells(
        changed_science, {"setup"}
    )

    changed_id = json.loads(json.dumps(notebook))
    changed_id["cells"][1]["id"] = "repaired-nbformat-id"
    assert _without_bootstrap_cells(notebook, {"setup"}) == _without_bootstrap_cells(
        changed_id, {"setup"}
    )

    changed_runtime = json.loads(json.dumps(notebook))
    changed_runtime["cells"][1]["execution_count"] = 7
    changed_runtime["cells"][1]["outputs"] = [
        {"name": "stdout", "output_type": "stream", "text": ["local path\n"]}
    ]
    assert _without_bootstrap_cells(notebook, {"setup"}) == _without_bootstrap_cells(
        changed_runtime, {"setup"}
    )


def test_every_benchmark_notebook_has_an_explicit_protection_policy() -> None:
    benchmark_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (USER_GUIDE / "benchmarks").glob("*.ipynb")
    }
    assert set(ALLOWED_BOOTSTRAP_CELLS) == benchmark_paths


def test_notebook_ci_validates_sources_without_committing_generated_edits() -> None:
    workflow = (ROOT / ".github" / "workflows" / "notebooks.yml").read_text(
        encoding="utf-8"
    )
    assert "normalize_notebook_environments.py --check" in workflow
    assert "validate_notebook_portability.py" in workflow
    assert "git commit" not in workflow
    assert "git push" not in workflow
    assert "PYTHONPATH" not in workflow

    notebook_paths = set(re.findall(r"User_guide/[A-Za-z0-9_./-]+\.ipynb", workflow))
    assert notebook_paths
    for relative_path in notebook_paths:
        assert (ROOT / relative_path).is_file(), relative_path


def test_protected_audit_is_content_addressed_and_covers_workflow_triggers() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "arbitrary-knot-fields-final-audit.yml"
    ).read_text(encoding="utf-8")
    assert "python3 dev/check_protected_final_audit.py" in workflow
    assert "protected_baseline=" not in workflow
    for protected_trigger in (
        '"CMakeLists.txt"',
        '"User_guide/benchmarks/**"',
        '"dev/final_performance_audit.py"',
        '"dev/final_performance_audit_medium.py"',
        '"src/knotted_graph/extraction/**"',
        '"src/knotted_graph/invariants/yamada/**"',
        '"src/knotted_graph/projection/**"',
    ):
        assert protected_trigger in workflow

    assert set(EXPECTED_BENCHMARK_FINGERPRINTS) == set(ALLOWED_BOOTSTRAP_CELLS)
    assert set(EXPECTED_GIT_OBJECTS) == {
        "CMakeLists.txt",
        "dev/final_performance_audit.py",
        "dev/final_performance_audit_medium.py",
        "src/knotted_graph/extraction",
        "src/knotted_graph/invariants/yamada",
        "src/knotted_graph/projection",
    }


def test_benchmark_fingerprint_detects_scientific_source_changes() -> None:
    notebook = {
        "cells": [
            {"id": "setup", "cell_type": "code", "source": ["setup = 1\n"]},
            {"id": "science", "cell_type": "code", "source": ["result = 1\n"]},
        ],
        "nbformat": 4,
    }
    baseline = _benchmark_fingerprint(notebook, {"setup"})

    runtime_only = json.loads(json.dumps(notebook))
    runtime_only["cells"][1]["execution_count"] = 1
    runtime_only["cells"][1]["outputs"] = [{"output_type": "stream"}]
    assert _benchmark_fingerprint(runtime_only, {"setup"}) == baseline

    changed = json.loads(json.dumps(notebook))
    changed["cells"][1]["source"] = ["result = 2\n"]
    assert _benchmark_fingerprint(changed, {"setup"}) != baseline


def test_all_notebooks_have_unique_ids_and_clean_outputs() -> None:
    for path in sorted(USER_GUIDE.rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(cell_ids), path
        assert len(cell_ids) == len(set(cell_ids)), path
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None, path
                assert cell.get("outputs") == [], path
