"""Render the current 2-by-4 Task 2 main figure."""

from __future__ import annotations

from pathlib import Path

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
from .specs import MAIN, PanelSpec
from .style import publication_style


def _draw_panel(fig, bbox, *, index: int, panel: PanelSpec, image) -> None:
    x, y, width, height = bbox
    pad_x = 0.028 * width
    pad_y = 0.012 * height
    header_h = 0.230 * height
    body_y = y + pad_y
    body_h = height - header_h - 2.0 * pad_y

    header_ax = fig.add_axes(
        [x + pad_x, y + height - pad_y - header_h, width - 2.0 * pad_x, header_h]
    )
    header_ax.axis("off")
    header_ax.text(
        0.0,
        0.82,
        panel_label(index),
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_size = 6.9 if "\n" in panel.title else (7.05 if len(panel.title) > 25 else 7.35)
    header_ax.text(
        0.56,
        0.82,
        panel.title,
        fontsize=title_size,
        fontweight="semibold",
        ha="center",
        va="center",
        linespacing=0.94,
    )
    formats = panel.formats or ""
    multiline = "\n" in formats
    header_ax.text(
        0.56,
        0.39 if multiline else 0.42,
        formats,
        fontsize=5.6 if multiline else 5.75,
        color="#50565A",
        ha="center",
        va="center",
        linespacing=1.12 if multiline else 0.95,
    )

    image_ax = fig.add_axes([x + pad_x, body_y, width - 2.0 * pad_x, body_h])
    image_ax.axis("off")
    image_ax.imshow(image, interpolation="lanczos")


def render_main(
    *,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> RenderResult:
    """Validate the accepted panels and render the current Main figure."""

    paths = validate_assets(MAIN, Path(asset_root))
    with publication_style():
        fig = new_figure(MAIN)
        try:
            for index, (panel, path) in enumerate(zip(MAIN.panels, paths, strict=True)):
                _draw_panel(
                    fig,
                    panel_bbox(index, rows=MAIN.rows, cols=MAIN.cols),
                    index=index,
                    panel=panel,
                    image=load_rgb(path),
                )
            return save_figure(fig, MAIN, Path(output_dir))
        finally:
            plt.close(fig)
