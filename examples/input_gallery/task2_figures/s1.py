"""Render the current 3-by-5 nonzero-Yamada appendix figure."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt

from .common import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_OUTPUT_DIR,
    RenderResult,
    load_rgb,
    new_figure,
    panel_bbox,
    panel_label,
    save_figure,
    validate_assets,
)
from .specs import PanelSpec, S1
from .style import publication_style


def format_yamada_lines(polynomial: str, *, single_line_limit: int = 53) -> list[str]:
    """Rename the audit indeterminate A to Y and use at most two CM lines."""

    expression = polynomial.replace("**", "^").replace("*", "").replace("A", "Y")
    terms = [
        re.sub(r"\s+", " ", term.strip())
        for term in re.findall(r"[+-]?\s*[^+-]+", expression)
        if term.strip()
    ]
    if not terms:
        raise ValueError("A nonempty Yamada polynomial is required")

    def joined(items: list[str]) -> str:
        return " ".join(items)

    prefix_measure = "Υ(G;Y) = "
    if len(terms) == 1 or len(prefix_measure) + len(joined(terms)) <= single_line_limit:
        split_lines = [terms]
    else:
        candidates = []
        for split in range(1, len(terms)):
            first = joined(terms[:split])
            second = joined(terms[split:])
            lengths = (len(prefix_measure) + len(first), len(second))
            candidates.append(((max(lengths), abs(lengths[0] - lengths[1])), split))
        _, split = min(candidates)
        split_lines = [terms[:split], terms[split:]]

    rendered: list[str] = []
    for index, line_terms in enumerate(split_lines):
        line = re.sub(r"Y\^(\d+)", r"Y^{\1}", joined(line_terms))
        prefix = r"\Upsilon(G;Y) = " if index == 0 else ""
        rendered.append(f"${prefix}{line}$")
    return rendered


def _draw_panel(fig, bbox, *, index: int, panel: PanelSpec, image) -> None:
    polynomial = panel.polynomial
    if not polynomial or polynomial.strip() == "0":
        raise ValueError(f"S1 requires a nonzero polynomial for {panel.key}")
    formula_lines = format_yamada_lines(polynomial)

    x, y, width, height = bbox
    pad_x = 0.025 * width
    pad_y = 0.012 * height
    title_h = 0.125 * height
    footer_h = (0.056 + 0.055 * len(formula_lines)) * height
    body_y = y + pad_y + footer_h
    body_h = height - 2.0 * pad_y - title_h - footer_h

    title_ax = fig.add_axes(
        [x + pad_x, y + height - pad_y - title_h, width - 2.0 * pad_x, title_h]
    )
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.78,
        panel_label(index),
        fontsize=8.4,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.54,
        0.72 if "\n" in panel.title else 0.78,
        panel.title,
        fontsize=7.15 if "\n" in panel.title else (7.35 if len(panel.title) > 31 else 7.8),
        fontweight="semibold",
        ha="center",
        va="center",
        linespacing=1.0,
    )

    image_ax = fig.add_axes([x + pad_x, body_y, width - 2.0 * pad_x, body_h])
    image_ax.axis("off")
    image_ax.imshow(image, interpolation="lanczos")

    footer_ax = fig.add_axes([x + pad_x, y + pad_y, width - 2.0 * pad_x, footer_h])
    footer_ax.axis("off")
    footer_ax.text(
        0.5,
        0.51,
        "\n".join(formula_lines),
        fontsize=6.35,
        fontweight="normal",
        linespacing=1.08,
        ha="center",
        va="center",
        color="#202428",
        math_fontfamily="cm",
    )


def render_s1(
    *,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> RenderResult:
    """Validate accepted inputs and render S1 with Computer Modern formulae."""

    paths = validate_assets(S1, Path(asset_root))
    with publication_style():
        fig = new_figure(S1)
        try:
            for index, (panel, path) in enumerate(zip(S1.panels, paths, strict=True)):
                _draw_panel(
                    fig,
                    panel_bbox(index, rows=S1.rows, cols=S1.cols),
                    index=index,
                    panel=panel,
                    image=load_rgb(path),
                )
            return save_figure(
                fig,
                S1,
                Path(output_dir),
                s1_svg_math_as_paths=True,
            )
        finally:
            plt.close(fig)
