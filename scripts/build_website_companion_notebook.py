"""Build a notebook companion from the public documentation pages.

The generated notebook keeps the website code blocks and embeds the displayed
text/image outputs directly after the relevant cells. It is intentionally a
presentation notebook: heavy scientific examples are shown with pre-rendered
website outputs rather than executed during notebook construction.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "doc"
OUT = ROOT / "User_guide" / "website_examples.ipynb"

DOC_FILES = [
    "index.md",
    "quickstart.md",
    "user_guide/workflow_overview.md",
    "user_guide/input_adapters.md",
    "user_guide/inspection_pipeline.md",
    "user_guide/projection_yamada.md",
    "user_guide/repulsive_layout.md",
    "applications/index.md",
    "applications/nodal_skeleton.md",
    "applications/biomolecular_protein_workflow.md",
    "applications/material_fingerprints.md",
    "applications/paper_notebook_gallery.md",
    "applications/mathematical_workflows.md",
]

FENCE_RE = re.compile(r"^```(.*)$")
CELL_COUNTER = 0


def next_cell_id() -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"kg-{CELL_COUNTER:05d}"


def markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": source,
    }


def code_cell(
    source: str,
    *,
    outputs: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "id": next_cell_id(),
        "execution_count": None,
        "metadata": metadata,
        "outputs": outputs or [],
        "source": source,
    }


def stream_output(text: str) -> dict[str, Any]:
    if text and not text.endswith("\n"):
        text += "\n"
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text,
    }


def image_output(path: Path, *, width: int | None = None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        data: dict[str, Any] = {"image/svg+xml": path.read_text(encoding="utf-8")}
    else:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        data = {mime: encoded}

    metadata: dict[str, Any] = {}
    key = next(iter(data))
    metadata[key] = {"alt": path.stem.replace("_", " ")}
    if width is not None:
        metadata[key]["width"] = width

    return {
        "output_type": "display_data",
        "data": data,
        "metadata": metadata,
    }


def markdown_embedded_image(path: Path, *, width: int | None = None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        mime = "image/svg+xml"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    alt = path.stem.replace("_", " ")
    width_attr = f' width="{width}"' if width is not None else ""
    return markdown_cell(
        f'<img src="data:{mime};base64,{encoded}" alt="{alt}"{width_attr} />\n'
    )


def resolve_doc_path(doc_file: Path, path_text: str) -> Path:
    cleaned = path_text.strip().strip("<>").strip()
    cleaned = cleaned.split()[0]
    return (doc_file.parent / cleaned).resolve()


def parse_image_directive(first_line: str) -> str:
    # first_line is like "{image} ../assets/foo.png"
    return first_line.split("}", 1)[1].strip()


def literalinclude_cell(doc_file: Path, first_line: str) -> dict[str, Any]:
    target_text = first_line.split("}", 1)[1].strip()
    target = resolve_doc_path(doc_file, target_text)
    if target.exists():
        content = target.read_text(encoding="utf-8")
        suffix = target.suffix.lower().lstrip(".") or "text"
        return markdown_cell(f"```{suffix}\n{content}\n```")
    return markdown_cell(f"> Missing literalinclude target: `{target_text}`")


def clean_markdown_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_option_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```{toctree}"):
            skip_option_block = True
            continue
        if skip_option_block:
            if stripped == "```":
                skip_option_block = False
            continue
        if stripped.startswith(("::::{grid}", ":::{grid-item-card}", "::::", ":::")):
            continue
        if stripped.startswith((":link:", ":link-type:", ":gutter:", "+++", ":caption:", ":maxdepth:")):
            continue
        cleaned.append(line)
    return cleaned


def flush_markdown(buffer: list[str], cells: list[dict[str, Any]]) -> None:
    cleaned = clean_markdown_lines(buffer)
    text = "\n".join(cleaned).strip()
    if text:
        cells.append(markdown_cell(text + "\n"))
    buffer.clear()


def is_executable_python(doc_file: Path, block_text: str) -> bool:
    """Return whether a website Python block should run in the companion notebook."""
    relative = doc_file.relative_to(DOC_ROOT).as_posix()

    # Material examples still depend on the paper-only NodalSkeletonMultiBand
    # helper. Show the code, but do not make Run All depend on that helper.
    if relative == "applications/material_fingerprints.md":
        return False

    reference_tokens = [
        "inspect_spatial_graph(",
        "relax_spatial_graph(",
        "layout.workspace",
        "relaxed_graph",
        "NodalSkeletonMultiBand",
        "plot_material_transition",
        "plot_surface_with_slice_planes",
        "plot_berry_slice_contours",
        "plot_oriented_spatial_graph",
    ]
    if any(token in block_text for token in reference_tokens):
        return False

    return True


def reference_code_cell(block_text: str, language: str = "python") -> dict[str, Any]:
    return markdown_cell(
        f"```{language}\n{block_text.rstrip()}\n```\n\n"
        "_Reference code shown from the website. It is not executed by Run All._\n"
    )


def parse_doc_page(doc_file: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    lines = doc_file.read_text(encoding="utf-8").splitlines()
    buffer: list[str] = []
    last_executable_code_index: int | None = None
    i = 0

    while i < len(lines):
        match = FENCE_RE.match(lines[i])
        if not match:
            buffer.append(lines[i])
            i += 1
            continue

        flush_markdown(buffer, cells)
        fence_info = match.group(1).strip()
        i += 1
        block: list[str] = []
        while i < len(lines) and lines[i].strip() != "```":
            block.append(lines[i])
            i += 1
        i += 1
        block_text = "\n".join(block).rstrip()

        if fence_info in {"python", "py"}:
            if is_executable_python(doc_file, block_text):
                cells.append(code_cell(block_text + "\n", tags=["website-code"]))
                last_executable_code_index = len(cells) - 1
            else:
                cells.append(reference_code_cell(block_text, "python"))
                last_executable_code_index = None
        elif fence_info == "text":
            if last_executable_code_index is not None:
                cells[last_executable_code_index]["outputs"].append(stream_output(block_text))
            else:
                cells.append(markdown_cell(f"```text\n{block_text}\n```\n"))
        elif fence_info in {"bash", "shell", "console"}:
            cells.append(
                markdown_cell(
                    "```bash\n"
                    + block_text
                    + "\n```\n\n"
                    "_Shown as a command block; run manually only when appropriate._\n"
                )
            )
            last_executable_code_index = None
        elif fence_info.startswith("{math}"):
            cells.append(markdown_cell("$$\n" + block_text + "\n$$\n"))
            last_executable_code_index = None
        elif fence_info.startswith("{image}"):
            image_text = parse_image_directive(fence_info)
            image_path = resolve_doc_path(doc_file, image_text)
            if image_path.exists():
                width = 980 if image_path.name.startswith("material_") else 900
                if last_executable_code_index is not None:
                    cells[last_executable_code_index]["outputs"].append(
                        image_output(image_path, width=width)
                    )
                else:
                    cells.append(markdown_embedded_image(image_path, width=width))
            else:
                cells.append(markdown_cell(f"> Missing image target: `{image_text}`\n"))
            last_executable_code_index = None
        elif fence_info.startswith("{literalinclude}"):
            cells.append(literalinclude_cell(doc_file, fence_info))
            last_executable_code_index = None
        else:
            cells.append(markdown_cell(f"```{fence_info}\n{block_text}\n```\n"))
            last_executable_code_index = None

    flush_markdown(buffer, cells)
    return cells


def collect_referenced_images(cells: list[dict[str, Any]]) -> int:
    count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for output in cell["outputs"]:
            if output.get("output_type") == "display_data":
                if any(key.startswith("image/") for key in output.get("data", {})):
                    count += 1
    return count


def build_notebook() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cells.append(
        markdown_cell(
            "# KnottedGraph Website Companion Notebook\n\n"
            "This notebook mirrors the public documentation website as a readable "
            "Jupyter companion. It keeps the website code blocks and embeds the "
            "displayed outputs immediately after the relevant code cells. The "
            "notebook is meant for inspection, teaching, and paper-development "
            "work; heavy cells are shown with the already-rendered website outputs.\n\n"
            f"Generated from the local documentation sources on **{now}**.\n"
        )
    )
    cells.append(
        code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'doc').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "DOC_ROOT = PROJECT_ROOT / 'doc'\n"
            "SRC_ROOT = PROJECT_ROOT / 'src'\n"
            "if str(SRC_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC_ROOT))\n\n"
            "print(f'PROJECT_ROOT = {PROJECT_ROOT}')\n"
            "print(f'DOC_ROOT = {DOC_ROOT}')\n",
            outputs=[
                stream_output(
                    "PROJECT_ROOT = <repository root>\n"
                    "DOC_ROOT = <repository root>/doc\n"
                )
            ],
            tags=["setup"],
        )
    )

    for relative in DOC_FILES:
        doc_file = DOC_ROOT / relative
        cells.append(
            markdown_cell(
                f"\n---\n\n# Website Page: `{relative}`\n\n"
                f"Source file: `{doc_file.relative_to(ROOT)}`\n"
            )
        )
        cells.extend(parse_doc_page(doc_file))

    image_count = collect_referenced_images(cells)
    cells.append(
        markdown_cell(
            "\n---\n\n# Notebook Audit\n\n"
            "The embedded outputs above are the same rendered figures/text outputs "
            "used by the website pages. This final section gives a quick inventory "
            "so it is easier to verify coverage after future documentation edits.\n"
        )
    )
    cells.append(
        code_cell(
            "from pathlib import Path\n\n"
            "page_sources = [\n"
            + "".join(f"    {relative!r},\n" for relative in DOC_FILES)
            + "]\n"
            "print(f'website_pages_included = {len(page_sources)}')\n"
            "print(f'embedded_figure_outputs = {"
            + str(image_count)
            + "}')\n"
            "print('pages:')\n"
            "for page in page_sources:\n"
            "    print(' -', page)\n",
            outputs=[
                stream_output(
                    f"website_pages_included = {len(DOC_FILES)}\n"
                    f"embedded_figure_outputs = {image_count}\n"
                    "pages:\n"
                    + "".join(f" - {relative}\n" for relative in DOC_FILES)
                )
            ],
            tags=["audit"],
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
            "knotted_graph": {
                "source": "documentation website",
                "generated_by": "scripts/build_website_companion_notebook.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    code_cells = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "code")
    markdown_cells = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    figure_outputs = collect_referenced_images(notebook["cells"])
    print(f"wrote {OUT}")
    print(f"code_cells = {code_cells}")
    print(f"markdown_cells = {markdown_cells}")
    print(f"embedded_figure_outputs = {figure_outputs}")


if __name__ == "__main__":
    main()
