from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
USER_GUIDE = ROOT / "User_guide"


# Jupyter renders these notebooks most consistently when Markdown math uses
# $...$ for inline expressions and $$...$$ for display expressions.  Reject
# legacy MathJax delimiters so they cannot silently reappear in future edits.
LEGACY_MATH_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")

# Also catch the malformed standalone form
# [
# E=\frac{3V}{2}
# ]
# without confusing ordinary Markdown links/lists with mathematics.
BARE_MATH_BLOCK = re.compile(
    r"(?ms)(?:^|\n)[ \t]*\[[ \t]*\n"
    r".*?\\(?:frac|text|operatorname|mathbb|mathrm|rm|Upsilon|Delta|sum|prod).*?"
    r"\n[ \t]*\][ \t]*(?=\n|$)"
)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _has_branch_local_bootstrap(text: str) -> bool:
    return (
        "sys.path.insert" in text
        and "src" in text
        and ("Path.cwd" in text or "ROOT" in text or "PROJECT_ROOT" in text)
    )


def _imports_knotted_graph(text: str) -> bool:
    return (
        "import knotted_graph" in text
        or "from knotted_graph" in text
    )


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
    bootstrap_seen = False

    for index, cell in enumerate(notebook.get("cells", [])):
        source = _cell_source(cell)

        if cell.get("cell_type") == "markdown":
            for error in _markdown_math_errors(source):
                errors.append(f"cell {index}: {error}")
            continue

        if cell.get("cell_type") != "code":
            continue

        if _has_branch_local_bootstrap(source):
            bootstrap_seen = True

        if _imports_knotted_graph(source) and not (
            bootstrap_seen or _has_branch_local_bootstrap(source)
        ):
            errors.append(
                f"cell {index}: imports knotted_graph before adding the repository src/ directory to sys.path"
            )

        if _cwd_dependent_repo_path(source):
            errors.append(
                f"cell {index}: contains a cwd-dependent repository path; derive it from the resolved repo root instead"
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
        print("Notebook portability/Markdown audit failed:")
        for notebook, errors in failures:
            print(f"- {notebook.relative_to(ROOT)}")
            for error in errors:
                print(f"    {error}")
        return 1

    print(
        f"PASS: {len(notebooks)} User_guide notebooks have portable branch-local imports, "
        "repo-rooted paths, and $...$/$$...$$ Markdown math delimiters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
