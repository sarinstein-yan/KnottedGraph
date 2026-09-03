from __future__ import annotations

import ast
import importlib
import json
import re
import runpy
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
NOTEBOOK_CI_PATH_RE = re.compile(r"User_guide/[A-Za-z0-9_./-]+\.ipynb")

OPTIONAL_IMPORT_ROOTS = {
    "Bio",
    "igraph",
    "kaleido",
    "minorminer",
    "pandas",
    "PIL",
    "plotly",
    "poly2graph",
    "pyvista",
    "skimage",
    "tabulate",
    "topoly",
}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    paths = [ROOT / line for line in out.splitlines() if line]
    return [path for path in paths if path.exists()]


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

    # Raw HTML buttons in Sphinx source point at built pages. During a source
    # audit, resolve them to the corresponding Markdown document.
    if source.is_relative_to(ROOT / "doc") and candidate.suffix == ".html":
        markdown_candidate = candidate.with_suffix(".md")
        if markdown_candidate.exists():
            return markdown_candidate

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
        path_only = path_text.split("::", 1)[0].rstrip(".,:;")
        candidate = ROOT / path_only
        if not candidate.exists():
            failures.append(
                f"stale repository path in {source.relative_to(ROOT)}: `{path_text}`"
            )


def check_source_hygiene(source: Path, text: str, failures: list[str]) -> None:
    for stale in sorted(STALE_TEXT):
        if stale in text:
            failures.append(f"superseded path/name in {source.relative_to(ROOT)}: {stale}")
    local_match = ABSOLUTE_LOCAL_RE.search(text)
    if local_match:
        failures.append(
            f"machine-specific absolute path in {source.relative_to(ROOT)}: {local_match.group(0)}"
        )


def _plain_python_source(source: str) -> str:
    """Remove notebook-only magic/shell lines before AST import auditing."""
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def _missing_optional_dependency(exc: BaseException) -> bool:
    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return False
    if exc.name.startswith("knotted_graph.invariants.yamada._yamada_"):
        return True
    return exc.name.split(".", 1)[0] in OPTIONAL_IMPORT_ROOTS


def _module_declares_symbol(module, name: str) -> bool:
    namespace = vars(module)
    if name in namespace:
        return True
    declared = namespace.get("__all__", ())
    return name in declared


def check_knotted_graph_imports(
    notebook: Path,
    code_cells: list[str],
    failures: list[str],
    optional_skips: list[str],
) -> None:
    for cell_index, source in enumerate(code_cells):
        try:
            tree = ast.parse(_plain_python_source(source))
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
                            if _missing_optional_dependency(exc):
                                optional_skips.append(
                                    f"{notebook.relative_to(ROOT)} cell {cell_index}: "
                                    f"import {alias.name} requires optional dependency {exc.name}"
                                )
                                continue
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
                    if _missing_optional_dependency(exc):
                        optional_skips.append(
                            f"{notebook.relative_to(ROOT)} cell {cell_index}: "
                            f"from {module} requires optional dependency {exc.name}"
                        )
                        continue
                    failures.append(
                        f"stale import module in {notebook.relative_to(ROOT)} cell {cell_index}: "
                        f"{module} ({type(exc).__name__}: {exc})"
                    )
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if not _module_declares_symbol(imported_module, alias.name):
                        failures.append(
                            f"missing imported symbol in {notebook.relative_to(ROOT)} cell {cell_index}: "
                            f"from {module} import {alias.name}"
                        )


def main() -> None:
    failures: list[str] = []
    optional_skips: list[str] = []
    paths = tracked_files()
    texts: dict[Path, str] = {}

    for path in paths:
        text = text_of(path)
        if text is None:
            continue
        texts[path] = text
        relative = path.relative_to(ROOT)

        # Notebook JSON contains saved execution outputs, which may legitimately
        # record the machine that last produced a figure. Portability concerns
        # executable/markdown source, so notebook hygiene is checked below after
        # parsing cells rather than against serialized outputs.
        if relative != SELF and path.suffix != ".ipynb":
            check_source_hygiene(path, text, failures)

        if path.suffix == ".md":
            check_links(path, text, failures)
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
        check_knotted_graph_imports(notebook, code_cells, failures, optional_skips)
        for source in [*markdown_cells, *code_cells]:
            check_source_hygiene(notebook, source, failures)
        for markdown in markdown_cells:
            check_links(notebook, markdown, failures)

    workflow = texts[ROOT / ".github" / "workflows" / "notebooks.yml"]
    ci_covered = set(NOTEBOOK_CI_PATH_RE.findall(workflow))
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in notebooks
        if path.is_relative_to(ROOT / "User_guide")
    }
    for missing in sorted(actual - ci_covered):
        failures.append(f"User-guide notebook missing from notebook CI coverage: {missing}")
    for stale in sorted(ci_covered - actual):
        failures.append(f"notebook CI references missing notebook: {stale}")

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

    version = pyproject["project"]["version"]
    package_init = texts[ROOT / "src" / "knotted_graph" / "__init__.py"]
    package_match = re.search(r'^__version__\s*=\s*[\"\']([^\"\']+)', package_init, re.M)
    docs_release = runpy.run_path(str(ROOT / "doc" / "conf.py")).get("release")
    if not package_match or package_match.group(1) != version:
        failures.append(f"package __version__ does not match pyproject version {version}")
    if docs_release != version:
        failures.append(f"doc release does not match pyproject version {version}")

    if failures:
        print("REPOSITORY CONSISTENCY FAILURES:")
        for index, failure in enumerate(sorted(set(failures)), start=1):
            print(f"{index:02d}. {failure}")
        raise SystemExit(1)
    if optional_skips:
        print("Optional-dependency import checks skipped:")
        for note in sorted(set(optional_skips)):
            print(f"- {note}")
    print("Repository consistency audit passed.")


if __name__ == "__main__":
    main()
