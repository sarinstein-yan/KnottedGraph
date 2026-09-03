"""Protect accepted performance/audit content with history-independent hashes.

Git object IDs identify the exact accepted implementation trees and files.
Benchmark notebooks use a canonical scientific-source fingerprint that ignores
only explicitly allowlisted bootstrap cells, cell IDs, and transient execution
state.  No baseline commit needs to remain in the current history, so this check
continues to work after a rebase or squash merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Content-addressed objects accepted by the final performance audit. Directory
# values are Git tree IDs; file values are Git blob IDs. They do not require the
# commit from which they were recorded to remain in repository history.
EXPECTED_GIT_OBJECTS = {
    "CMakeLists.txt": "42ed69ac293bba15f4209231b5ee309cd149dc74",
    "dev/final_performance_audit.py": "d1d6d2568aa77c0b16ad9afc4872efe2daec6ada",
    "dev/final_performance_audit_medium.py": "227e88fbe16c05b7e7c09d1e09cb6fe9323c36c6",
    "src/knotted_graph/extraction": "47c30ced3fd6499bcf02b32c49936c9ea128d2f5",
    "src/knotted_graph/invariants/yamada": "de87e490047cc4b9150218acfd5a89823d6d093e",
    "src/knotted_graph/projection": "e0354b5f80dda3ecd008da3eab46d3c578bd5804",
}

# These cells contain environment discovery/reporting rather than the accepted
# mathematical constructions, benchmark cases, or result interpretation.
ALLOWED_BOOTSTRAP_CELLS = {
    "User_guide/benchmarks/01_yamada_sanity_checks.ipynb": {"published"},
    # These notebooks need no source-cell exception. They are listed so that
    # cell IDs and transient execution state can still be normalized without
    # weakening the comparison of their scientific source and metadata.
    "User_guide/benchmarks/02_application_regression_checks.ipynb": set(),
    "User_guide/benchmarks/03_knottedgraph_vs_topoly_scaling.ipynb": set(),
    "User_guide/benchmarks/04_thick_handlebody_validation.ipynb": {"af32d418"},
}

EXPECTED_BENCHMARK_FINGERPRINTS = {
    "User_guide/benchmarks/01_yamada_sanity_checks.ipynb": (
        "ae261e4e1ff192f63726e8401ae3c37c6840fc587ee9b9534b255872e0ec67f8"
    ),
    "User_guide/benchmarks/02_application_regression_checks.ipynb": (
        "36d77d39551dd33f823c06ed7e033e4ea9599b59440709f2355111356f458cca"
    ),
    "User_guide/benchmarks/03_knottedgraph_vs_topoly_scaling.ipynb": (
        "24bae6ce245c650788d6fcf1fc7bc2aa8c9d8fed56c121125b47c3dbc161b9e6"
    ),
    "User_guide/benchmarks/04_thick_handlebody_validation.ipynb": (
        "dd58463d6515c9e9faa0d608b5ce81f53a0e886b959fe3b4156252cdc3d96c3e"
    ),
}


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def _notebook_at(revision: str, path: str) -> dict:
    return json.loads(_git("show", f"{revision}:{path}"))


def _without_bootstrap_cells(notebook: dict, allowed_ids: set[str]) -> dict:
    normalized = json.loads(json.dumps(notebook))
    normalized["cells"] = [
        cell
        for cell in normalized.get("cells", [])
        if cell.get("id") not in allowed_ids
    ]
    # Cell IDs and transient Jupyter execution state are editor/runtime
    # bookkeeping, not accepted scientific source. Accepted benchmark results
    # live in separately tracked CSV/cache/ground-truth artifacts.
    for cell in normalized["cells"]:
        cell.pop("id", None)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return normalized


def _benchmark_fingerprint(notebook: dict, allowed_ids: set[str]) -> str:
    canonical = _without_bootstrap_cells(notebook, allowed_ids)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check(revision: str = "HEAD") -> list[str]:
    failures: list[str] = []

    for path, expected in EXPECTED_GIT_OBJECTS.items():
        try:
            actual = _git("rev-parse", f"{revision}:{path}").strip()
        except Exception as exc:
            failures.append(f"could not read protected object {path}: {exc}")
            continue
        if actual != expected:
            failures.append(
                f"protected implementation changed: {path} "
                f"(expected {expected}, found {actual})"
            )

    for path, expected in EXPECTED_BENCHMARK_FINGERPRINTS.items():
        try:
            notebook = _notebook_at(revision, path)
            actual = _benchmark_fingerprint(
                notebook,
                ALLOWED_BOOTSTRAP_CELLS[path],
            )
        except Exception as exc:
            failures.append(f"could not fingerprint protected notebook {path}: {exc}")
            continue
        if actual != expected:
            failures.append(
                f"protected benchmark scientific content changed: {path} "
                f"(expected {expected}, found {actual})"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision",
        default="HEAD",
        help="Committed tree to audit (default: HEAD).",
    )
    args = parser.parse_args()

    failures = check(args.revision)
    if failures:
        print("PROTECTED FINAL-AUDIT FAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "PASS: protected implementation objects and benchmark scientific-source "
        "fingerprints match the accepted content-addressed baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
