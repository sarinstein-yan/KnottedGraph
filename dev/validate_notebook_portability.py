from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
USER_GUIDE = ROOT / "User_guide"

LEGACY_MATH_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
BARE_MATH_BLOCK = re.compile(
    r"(?ms)(?:^|\n)[ \t]*\[[ \t]*\n"
    r".*?\\(?:frac|text|operatorname|mathbb|mathrm|rm|Upsilon|Delta|sum|prod).*?"
    r"\n[ \t]*\][ \t]*(?=\n|$)"
)

YAMADA_MARKERS = (
    "compute_yamada_polynomial(",
    "compute_yamada_polynomial_recursive(",
    "Yamada(",
)

PLOTTING_ONLY = {
    Path("User_guide/benchmarks/06_paper_scaling_publication_figure.ipynb"),
}

# Notebook code must import the installed package. Prepending the raw source tree
# bypasses editable-install import hooks and can hide the compiled _yamada_native
# extension, producing misleading fallback/performance results.
SOURCE_OVERRIDE_PATTERNS = (
    re.compile(r"sys\.path\.insert\([^\n]*src", re.IGNORECASE),
    re.compile(r"PYTHONPATH[^\n]*src", re.IGNORECASE),
)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _cwd_dependent_repo_path(text: str) -> bool:
    bad_literals = (
        'Path("User_guide/',
        "Path('User_guide/",
        'Path("dev/',
        "Path('dev/",
        'Path("src/',
        "Path('src/",
    )
    return any(token in text for token in bad_literals)


def _markdown_math_errors(text: str) -> list[str]:
    errors: list[str] = []
    legacy = [token for token in LEGACY_MATH_DELIMITERS if token in text]
    if legacy:
        errors.append(
            "Markdown math uses legacy delimiters "
            f"{legacy}; use $...$ inline and $$...$$ for display math"
        )
    if BARE_MATH_BLOCK.search(text):
        errors.append(
            "Markdown contains a standalone [ ... ] LaTeX block; "
            "use $$...$$ for display math"
        )
    return errors


def validate_notebook(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    errors: list[str] = []
    code_texts: list[str] = []

    for index, cell in enumerate(notebook.get("cells", [])):
        source = _cell_source(cell)

        if cell.get("cell_type") == "markdown":
            for error in _markdown_math_errors(source):
                errors.append(f"cell {index}: {error}")
            continue

        if cell.get("cell_type") != "code":
            continue

        code_texts.append(source)
        try:
            ast.parse(source or "pass")
        except SyntaxError as exc:
            errors.append(f"cell {index}: Python syntax error: {exc}")

        # Historical application regression intentionally injects each detached
        # worktree's source path in a subprocess so two revisions can be compared.
        # It is correctness-only, never a performance benchmark.
        historical_regression = path.name == "02_application_output_regression.ipynb"
        if not historical_regression:
            for pattern in SOURCE_OVERRIDE_PATTERNS:
                if pattern.search(source):
                    errors.append(
                        f"cell {index}: raw src/PYTHONPATH package override is forbidden; "
                        "use the active installed/editable KnottedGraph environment"
                    )
                    break

        if _cwd_dependent_repo_path(source):
            errors.append(
                f"cell {index}: contains a cwd-dependent repository path; "
                "derive it from the resolved repository root"
            )

    joined = "\n".join(code_texts)
    relative = path.relative_to(ROOT)
    evaluates_yamada = any(marker in joined for marker in YAMADA_MARKERS)
    if evaluates_yamada and relative not in PLOTTING_ONLY:
        if "native_available" not in joined or "native_import_error" not in joined:
            errors.append(
                "Yamada is evaluated without printing native_available/native_import_error; "
                "backend provenance is required"
            )

    return errors


def main() -> int:
    notebooks = sorted(USER_GUIDE.rglob("*.ipynb"))
    failures: list[tuple[Path, list[str]]] = []

    for notebook in notebooks:
        errors = validate_notebook(notebook)
        if errors:
            failures.append((notebook, errors))

    if failures:
        print("Notebook portability/environment audit failed:")
        for notebook, errors in failures:
            print(f"- {notebook.relative_to(ROOT)}")
            for error in errors:
                print(f"    {error}")
        return 1

    print(
        f"PASS: {len(notebooks)} User_guide notebooks use installed-package semantics, "
        "repo-rooted paths, valid Python cells, backend provenance for Yamada, "
        "and $...$/$$...$$ Markdown math delimiters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
