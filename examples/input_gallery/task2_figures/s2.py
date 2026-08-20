"""Render the current 3-by-4 skeletonization appendix figure."""

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
from .specs import PanelSpec, S2
from .style import publication_style


def _draw_panel(fig, bbox, *, index: int, panel: PanelSpec, image) -> None:
    x, y, width, height = bbox
    pad_x = 0.025 * width
    pad_y = 0.012 * height
    title_h = 0.125 * height
    body_y = y + pad_y
    body_h = height - 2.0 * pad_y - title_h

    header_left = x + pad_x
    header_bottom = y + height - pad_y - title_h
    header_width = width - 2.0 * pad_x
    label_fraction = 0.17
    title_start_fraction = 0.19

    label_ax = fig.add_axes(
        [header_left, header_bottom, header_width * label_fraction, title_h]
    )
    label_ax.axis("off")
    label_ax.text(
        0.02,
        0.76,
        panel_label(index),
        fontsize=7.35,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=True,
    )

    title_ax = fig.add_axes(
        [
            header_left + header_width * title_start_fraction,
            header_bottom,
            header_width * (1.0 - title_start_fraction),
            title_h,
        ]
    )
    title_ax.axis("off")
    multiline = "\n" in panel.title
    title_ax.text(
        0.50,
        0.69 if multiline else 0.76,
        panel.title,
        fontsize=6.35 if multiline else 6.65,
        fontweight="semibold",
        ha="center",
        va="center",
        linespacing=0.94,
        clip_on=True,
    )

    image_ax = fig.add_axes([x + pad_x, body_y, width - 2.0 * pad_x, body_h])
    image_ax.axis("off")
    image_ax.imshow(image, interpolation="lanczos")


def render_s2(
    *,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> RenderResult:
    """Validate accepted inputs and render the current S2 figure."""

    paths = validate_assets(S2, Path(asset_root))
    with publication_style():
        fig = new_figure(S2)
        try:
            for index, (panel, path) in enumerate(zip(S2.panels, paths, strict=True)):
                _draw_panel(
                    fig,
                    panel_bbox(index, rows=S2.rows, cols=S2.cols),
                    index=index,
                    panel=panel,
                    image=load_rgb(path),
                )
            return save_figure(fig, S2, Path(output_dir))
        finally:
            plt.close(fig)
