"""Offline contracts for public documentation paths and built HTML."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import runpy
import sys
import tomllib
from urllib.parse import unquote, urlsplit

import knotted_graph


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
GITHUB_FILE_URL = re.compile(
    r"https://(?:"
    r"github\.com/sarinstein-yan/KnottedGraph/(?:blob|raw)/"
    r"|raw\.githubusercontent\.com/sarinstein-yan/KnottedGraph/"
    r")([^/\s)\"'>]+)/([^\s)\"'>]+)"
)


def _public_markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "doc").rglob("*.md"))]


def _local_markdown_targets(path: Path) -> list[Path]:
    targets: list[Path] = []
    for raw_target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        targets.append((path.parent / unquote(parsed.path)).resolve())
    return targets


def test_readme_local_links_resolve_inside_the_repository():
    for target in _local_markdown_targets(ROOT / "README.md"):
        assert target.is_relative_to(ROOT)
        assert target.exists(), target


def test_fixed_github_file_links_share_one_ref_and_resolve_locally():
    matches: list[tuple[str, str]] = []
    for path in _public_markdown_files():
        matches.extend(GITHUB_FILE_URL.findall(path.read_text(encoding="utf-8")))

    assert matches
    assert {ref for ref, _ in matches} == {"Latest_Workplace"}
    for _, relative_path in matches:
        local_path = ROOT / unquote(relative_path.split("#", 1)[0])
        assert local_path.exists(), local_path


def test_documentation_release_is_derived_from_project_version():
    project_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    namespace = runpy.run_path(str(ROOT / "doc" / "conf.py"))

    assert knotted_graph.__version__ == project_version
    assert namespace["release"] == project_version


def test_current_version_is_reflected_in_onboarding_prose():
    project_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    major_minor = ".".join(project_version.split(".")[:2])
    expected_references = {
        ROOT / "README.md": f"{project_version} development API",
        ROOT / "doc" / "installation.md": f"{project_version} development API",
        ROOT / "doc" / "quickstart.md": f"{project_version} development API",
        ROOT / "User_guide" / "01_getting_started.ipynb": (
            f"**{major_minor} development API**"
        ),
        ROOT / "User_guide" / "02_core_workflows.ipynb": (
            f"{major_minor} is released"
        ),
        ROOT / "User_guide" / "03_advanced_and_reproduction.ipynb": (
            f"installed {major_minor} package"
        ),
    }

    for path, expected in expected_references.items():
        assert expected in path.read_text(encoding="utf-8")
    assert "legacy `knotted_graph` 0.1.2" in (ROOT / "README.md").read_text(
        encoding="utf-8"
    )


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        for name in ("id", "name"):
            if attributes.get(name):
                self.anchors.add(str(attributes[name]))
        for name in ("href", "src"):
            if attributes.get(name):
                self.references.append(str(attributes[name]))


def _parse_page(path: Path) -> _PageParser:
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def validate_html_tree(html_root: Path) -> list[str]:
    """Return local file/fragment failures in a built Sphinx HTML tree."""

    html_root = html_root.resolve()
    if not html_root.is_dir():
        return [f"HTML root is not a directory: {html_root}"]
    if not (html_root / "index.html").is_file():
        return [f"HTML root has no index.html: {html_root}"]

    pages = [
        path
        for path in html_root.rglob("*.html")
        if "_static" not in path.relative_to(html_root).parts
    ]
    if not pages:
        return [f"HTML root contains no pages: {html_root}"]
    parsed_pages = {path: _parse_page(path) for path in pages}
    failures: list[str] = []

    for source, parsed_source in list(parsed_pages.items()):
        for reference in parsed_source.references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc or reference.startswith(("data:", "mailto:")):
                continue

            target_path = unquote(parsed.path)
            if target_path.startswith("/"):
                target = html_root / target_path.lstrip("/")
            elif target_path:
                target = source.parent / target_path
            else:
                target = source
            target = target.resolve()

            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(html_root) or not target.exists():
                failures.append(f"{source.relative_to(html_root)} -> missing {reference}")
                continue

            fragment = unquote(parsed.fragment)
            if fragment and target.suffix.lower() == ".html":
                parsed_target = parsed_pages.get(target)
                if parsed_target is None:
                    parsed_target = _parse_page(target)
                    parsed_pages[target] = parsed_target
                if fragment not in parsed_target.anchors:
                    failures.append(
                        f"{source.relative_to(html_root)} -> missing fragment {reference}"
                    )

    return failures


def test_html_validator_detects_missing_files_and_fragments(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="present.html#missing">fragment</a>'
        '<img src="missing.png">',
        encoding="utf-8",
    )
    (tmp_path / "present.html").write_text(
        '<h1 id="present">Present</h1>', encoding="utf-8"
    )

    failures = validate_html_tree(tmp_path)

    assert any("missing.png" in failure for failure in failures)
    assert any("present.html#missing" in failure for failure in failures)


def test_html_validator_rejects_missing_or_empty_build_roots(tmp_path):
    assert validate_html_tree(tmp_path / "missing")
    assert validate_html_tree(tmp_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-root", type=Path, required=True)
    args = parser.parse_args(argv)
    failures = validate_html_tree(args.html_root)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"HTML link contract passed: {args.html_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
