#!/usr/bin/env python3
"""Normalize Markdown math delimiters in User_guide notebooks.

The documentation stack used for these notebooks is most reliable with
``$...$`` for inline math and ``$$...$$`` for display math.  This utility
converts legacy ``\\(...\\)`` and ``\\[...\\]`` delimiters while preserving the
notebook's existing JSON formatting byte-for-byte apart from those replacements.

It also verifies that every replaced delimiter occurs in a Markdown cell; this
prevents an accidental text-level replacement inside Python code or outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


TOKENS = {
    r"\(": "$",
    r"\)": "$",
    r"\[": "$$",
    r"\]": "$$",
}

# A conservative detector for the common malformed form
#
# [
# E=\frac{3V}{2}
# ]
#
# which some Markdown renderers show literally instead of as display math.
BARE_BLOCK_RE = re.compile(
    r"(?ms)(?P<prefix>^|\n)[ \t]*\[[ \t]*\n"
    r"(?P<body>.*?\\(?:frac|text|operatorname|mathbb|mathrm|rm|Upsilon|Delta|sum|prod).*?)"
    r"\n[ \t]*\][ \t]*(?=\n|$)"
)


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _replace_in_source(source, old: str, new: str):
    if isinstance(source, list):
        return [part.replace(old, new) for part in source]
    return str(source).replace(old, new)


def _normalize_bare_blocks(source):
    """Convert only strongly LaTeX-looking standalone [ ... ] blocks."""
    text = "".join(source) if isinstance(source, list) else str(source)

    def repl(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        return f"{match.group('prefix')}$$\n{body}\n$$"

    normalized = BARE_BLOCK_RE.sub(repl, text)
    if normalized == text:
        return source, 0

    # Preserve list-vs-string source representation, but a modified list cell is
    # safely represented as one Markdown string.  This affects only malformed
    # bracket blocks and not ordinary delimiter conversions.
    if isinstance(source, list):
        return normalized.splitlines(keepends=True), 1
    return normalized, 1


def inspect_notebook(path: Path) -> tuple[dict, dict[str, int], int]:
    raw = path.read_text(encoding="utf-8")
    notebook = json.loads(raw)
    markdown_cells = [
        cell for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]

    markdown_counts = {
        token: sum(_source_text(cell).count(token) for cell in markdown_cells)
        for token in TOKENS
    }

    # Compare against encoded JSON text. If a token occurs outside Markdown,
    # refuse the formatting-preserving global replacement.
    for token, markdown_count in markdown_counts.items():
        encoded_token = json.dumps(token, ensure_ascii=False)[1:-1]
        raw_count = raw.count(encoded_token)
        if raw_count != markdown_count:
            raise RuntimeError(
                f"{path}: {token!r} occurs {raw_count} times in raw JSON but "
                f"{markdown_count} times in Markdown cells; refusing global replacement."
            )

    bare_blocks = sum(
        len(BARE_BLOCK_RE.findall(_source_text(cell)))
        for cell in markdown_cells
    )
    return notebook, markdown_counts, bare_blocks


def normalize_file(path: Path, *, write: bool) -> tuple[int, int]:
    raw = path.read_text(encoding="utf-8")
    notebook, counts, bare_blocks = inspect_notebook(path)
    delimiter_changes = sum(counts.values())

    updated = raw
    for token, replacement in TOKENS.items():
        encoded_token = json.dumps(token, ensure_ascii=False)[1:-1]
        updated = updated.replace(encoded_token, replacement)

    # Bare bracket blocks require cell-aware rewriting. They are rare; only
    # reserialize the notebook when such a malformed block is actually present.
    if bare_blocks:
        changed = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            source, n = _normalize_bare_blocks(cell.get("source", ""))
            if n:
                cell["source"] = source
                changed = True
        if changed:
            # Apply ordinary delimiter normalization in parsed Markdown too.
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "markdown":
                    continue
                for token, replacement in TOKENS.items():
                    cell["source"] = _replace_in_source(
                        cell.get("source", ""), token, replacement
                    )
            updated = json.dumps(
                notebook,
                ensure_ascii=False,
                indent=1,
            ) + "\n"

    if write and updated != raw:
        path.write_text(updated, encoding="utf-8")

    return delimiter_changes, bare_blocks


def remaining_legacy_math(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        text = _source_text(cell)
        legacy = [token for token in TOKENS if token in text]
        bare = bool(BARE_BLOCK_RE.search(text))
        if legacy or bare:
            problems.append(
                f"{path}: markdown cell {index}: "
                f"legacy={legacy}, bare_bracket_math={bare}"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite notebooks instead of only checking/reporting",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    notebooks = sorted((args.root / "User_guide").rglob("*.ipynb"))
    if not notebooks:
        raise RuntimeError("No User_guide notebooks found.")

    total_delimiters = 0
    total_bare = 0
    changed_files = []

    for path in notebooks:
        before = path.read_bytes()
        n_delimiters, n_bare = normalize_file(path, write=args.write)
        total_delimiters += n_delimiters
        total_bare += n_bare
        if args.write and path.read_bytes() != before:
            changed_files.append(path)

    problems = []
    for path in notebooks:
        problems.extend(remaining_legacy_math(path))

    print(f"Scanned {len(notebooks)} notebooks.")
    print(f"Legacy delimiter occurrences found: {total_delimiters}")
    print(f"Malformed standalone bracket-math blocks found: {total_bare}")
    if changed_files:
        print("Updated:")
        for path in changed_files:
            print(" -", path.relative_to(args.root))

    if problems:
        print("Remaining Markdown math problems:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)

    if not args.write and (total_delimiters or total_bare):
        print("Run with --write to normalize these notebooks.", file=sys.stderr)
        raise SystemExit(1)

    print("PASS: notebook Markdown math uses $...$ / $$...$$ delimiters.")


if __name__ == "__main__":
    main()
