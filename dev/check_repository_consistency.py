from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path("dev/check_repository_consistency.py")

STALE_TEXT = {
    "Sanity_checks.ipynb",
    "User_guide/benchmarks/03_application_output_regression.ipynb",
    "User_guide/benchmarks/04_knottedgraph_vs_topoly_fair.ipynb",
    "doc/user_guide/inspection_pipeline.md",
    "doc/applications/nodal_skeleton.md",
    "doc/applications/biomolecular_protein_workflow.md",
    "doc/applications/paper_notebook_gallery.md",
    "doc/applications/mathematical_workflows.md",
    "src/knotted_graph/yamada/",
}

GENERATED_PATHS = {
    "doc/_build/",
    "site_preview/",
}

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".rst",
    ".cpp",
    ".hpp",
    ".h",
    ".c",
    ".ipynb",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
INLINE_PATH_RE = re.compile(r"`((?:User_guide|doc|src|tests|scripts|dev)/[^`]+)`")
EXTRA_RE = re.compile(r"knotted_graph\[([A-Za-z0-9_-]+)\]")
ABSOLUTE_LOCAL_RE = re.compile(
    r"(?:/Users/[^\s\"'`]+|/home/[^\s\"'`]+|[A-Za-z]:\\\\Users\\\\[^\s\"'`]+)"
)
NOTEBOOK_MATRIX_RE = re.compile(r"^\s*-\s+(User_guide/[^\s]+\.ipynb)\s*$", re.M)


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def text_of(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def notebook_sources(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4:
        raise AssertionError(f"{path.relative_to(ROOT)}: expected nbformat 4")
    markdown: list[str] = []
    code: list[str] = []
    for cell in data.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if cell.get("cell_type") == "markdown":
            markdown.append(source)
        elif cell.get("cell_type") == "code":
            code.append(source)
    return markdown, code


def resolve_repo_link(source: Path, target: str) -> Path | None:
    target = target.strip().split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:", "data:", "#")):
        return None

    candidate = (source.parent / target).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return None
    if candidate.exists():
        return candidate

    # Sphinx copies doc/assets to the HTML root through html_extra_path. Raw
    # HTML images therefore legitimately use site_figures/... or ../site_figures/...
    # even though those paths do not exist next to the Markdown source.
    if source.is_relative_to(ROOT / "doc"):
        stripped = target
        while stripped.startswith("../"):
            stripped = stripped[3:]
        stripped = stripped.removeprefix("./")
        asset_candidate = ROOT / "doc" / "assets" / stripped
        if asset_candidate.exists():
            return asset_candidate

    return candidate


def check_links(source: Path, text: str, failures: list[str]) -> None:
    for target in LINK_RE.findall(text) + HTML_LINK_RE.findall(text):
        resolved = resolve_repo_link(source, target)
        if resolved is not None and not resolved.exists():
            failures.append(
                f"broken relative link: {source.relative_to(ROOT)} -> {target}"
            )


def check_toctrees(source: Path, text: str, failures: list[str]) -> None:
    if not source.is_relative_to(ROOT / "doc") or source.suffix != ".md":
        return
    lines = text.splitlines()
    in_tree = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("```{toctree}"):
            in_tree = True
            continue
        if in_tree and line == "```":
            in_tree = False
            continue
        if not in_tree or not line or line.startswith(":"):
            continue
        target = line.split()[0]
        base = source.parent / target
        candidates = [base, base.with_suffix(".md"), base / "index.md"]
        if not any(candidate.exists() for candidate in candidates):
            failures.append(
                f"missing toctree target: {source.relative_to(ROOT)} -> {target}"
            )


def check_inline_paths(source: Path, text: str, failures: list[str]) -> None:
    for path_text in INLINE_PATH_RE.findall(text):
        if path_text in GENERATED_PATHS:
            continue
        if any(token in path_text for token in ("*", "$", "{", "}", " -> ", "<", ">")):
            continue
        # Code-navigation prose may append a Python symbol anchor using
        # file.py::Class.method. Validate the file portion only.
        path_only = path_text.split("::", 1)[0].rstrip(".,:;")
        candidate = ROOT / path_only
        if not candidate.exists():
            failures.append(
                f"stale repository path in {source.relative_to(ROOT)}: `{path_text}`"
            )


def check_knotted_graph_imports(
    notebook: Path, code_cells: list[str], failures: list[str]
) -> None:
    for cell_index, source in enumerate(code_cells):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            failures.append(
                f"invalid Python in {notebook.relative_to(ROOT)} cell {cell_index}: {exc}"
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "knotted_graph" or alias.name.startswith("knotted_graph."):
                        try:
                            importlib.import_module(alias.name)
                        except Exception as exc:
                            failures.append(
                                f"stale import in {notebook.relative_to(ROOT)} cell {cell_index}: "
                                f"import {alias.name} ({type(exc).__name__}: {exc})"
                            )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not (module == "knotted_graph" or module.startswith("knotted_graph.")):
                    continue
                try:
                    imported_module = importlib.import_module(module)
                except Exception as exc:
                    failures.append(
                        f"stale import module in {notebook.relative_to(ROOT)} cell {cell_index}: "
                        f"{module} ({type(exc).__name__}: {exc})"
                    )
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if not hasattr(imported_module, alias.name):
                        failures.append(
                            f"missing imported symbol in {notebook.relative_to(ROOT)} cell {cell_index}: "
                            f"from {module} import {alias.name}"
                        )


def main() -> None:
    failures: list[str] = []
    paths = tracked_files()
    texts: dict[Path, str] = {}

    for path in paths:
        text = text_of(path)
        if text is None:
            continue
        texts[path] = text
        relative = path.relative_to(ROOT)

        # The audit necessarily contains the stale tokens it is designed to
        # reject, so do not audit its own rule declarations as repository prose.
        if relative != SELF:
            for stale in sorted(STALE_TEXT):
                if stale in text:
                    failures.append(f"superseded path/name in {relative}: {stale}")

            local_match = ABSOLUTE_LOCAL_RE.search(text)
            if local_match:
                failures.append(
                    f"machine-specific absolute path in {relative}: {local_match.group(0)}"
                )

        if path.suffix in {".md", ".ipynb"}:
            check_links(path, text, failures)
        if path.suffix == ".md":
            check_toctrees(path, text, failures)
            check_inline_paths(path, text, failures)

    notebooks = sorted(path for path in paths if path.suffix == ".ipynb")
    for notebook in notebooks:
        try:
            markdown_cells, code_cells = notebook_sources(notebook)
        except Exception as exc:
            failures.append(
                f"invalid notebook {notebook.relative_to(ROOT)}: {type(exc).__name__}: {exc}"
            )
            continue
        check_knotted_graph_imports(notebook, code_cells, failures)
        for markdown in markdown_cells:
            check_links(notebook, markdown, failures)

    # Every tracked User_guide notebook must be executed by the notebook CI matrix,
    # and the matrix must not reference a deleted notebook.
    workflow = texts[ROOT / ".github" / "workflows" / "notebooks.yml"]
    matrix = set(NOTEBOOK_MATRIX_RE.findall(workflow))
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in notebooks
        if path.is_relative_to(ROOT / "User_guide")
    }
    for missing in sorted(actual - matrix):
        failures.append(f"User-guide notebook missing from CI matrix: {missing}")
    for stale in sorted(matrix - actual):
        failures.append(f"CI matrix references missing notebook: {stale}")

    # Documented optional dependency names must exist in pyproject.toml.
    pyproject = tomllib.loads(texts[ROOT / "pyproject.toml"])
    extras = set(pyproject["project"].get("optional-dependencies", {}))
    user_text = "\n".join(
        text
        for path, text in texts.items()
        if path == ROOT / "README.md"
        or path.is_relative_to(ROOT / "doc")
        or path.is_relative_to(ROOT / "User_guide")
    )
    for extra in sorted(set(EXTRA_RE.findall(user_text)) - extras):
        failures.append(f"documentation references undefined optional extra: {extra}")

    # Keep the three public version declarations synchronized.
    version = pyproject["project"]["version"]
    package_init = texts[ROOT / "src" / "knotted_graph" / "__init__.py"]
    package_match = re.search(r'^__version__\s*=\s*[\"\']([^\"\']+)', package_init, re.M)
    conf = texts[ROOT / "doc" / "conf.py"]
    docs_match = re.search(r'^release\s*=\s*[\"\']([^\"\']+)', conf, re.M)
    if not package_match or package_match.group(1) != version:
        failures.append(f"package __version__ does not match pyproject version {version}")
    if not docs_match or docs_match.group(1) != version:
        failures.append(f"doc release does not match pyproject version {version}")

    if failures:
        print("REPOSITORY CONSISTENCY FAILURES:")
        for index, failure in enumerate(sorted(set(failures)), start=1):
            print(f"{index:02d}. {failure}")
        raise SystemExit(1)

    print(
        "PASS: repository-wide consistency audit succeeded for "
        f"{len(paths)} tracked files and {len(notebooks)} notebooks."
    )


if __name__ == "__main__":
    main()
