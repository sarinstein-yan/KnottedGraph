from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_GUIDE = ROOT / "User_guide"


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


def validate_notebook(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    errors: list[str] = []
    bootstrap_seen = False

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = _cell_source(cell)
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
        print("Notebook portability audit failed:")
        for notebook, errors in failures:
            print(f"- {notebook.relative_to(ROOT)}")
            for error in errors:
                print(f"    {error}")
        return 1

    print(f"PASS: {len(notebooks)} User_guide notebooks have portable branch-local imports and repo-rooted paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
