"""Compact Matplotlib assembly helpers for Task 2 appendix figures."""

from __future__ import annotations

from matplotlib.patches import Rectangle


def compact_panel_bboxes(
    n_panels: int,
    *,
    rows: int,
    cols: int,
    left: float = 0.012,
    right: float = 0.988,
    bottom: float = 0.018,
    top: float = 0.982,
    gap_x: float = 0.004,
    gap_y: float = 0.006,
    center_last_row: bool = True,
) -> list[tuple[float, float, float, float]]:
    """Return figure-coordinate boxes with optional centered final row."""
    total_w = right - left
    total_h = top - bottom
    cell_w = (total_w - (cols - 1) * gap_x) / cols
    cell_h = (total_h - (rows - 1) * gap_y) / rows
    bboxes = []
    for row in range(rows):
        remaining = max(n_panels - row * cols, 0)
        count = min(cols, remaining)
        if count <= 0:
            break
        row_width = count * cell_w + (count - 1) * gap_x
        row_left = left + 0.5 * (total_w - row_width) if center_last_row and count < cols else left
        y = top - (row + 1) * cell_h - row * gap_y
        for col in range(count):
            x = row_left + col * (cell_w + gap_x)
            bboxes.append((x, y, cell_w, cell_h))
    return bboxes


def draw_compact_panel(
    fig,
    bbox: tuple[float, float, float, float],
    *,
    label: str,
    title: str,
    source_image,
    result_image,
    result_label: str,
    frame_color: str = "#c4c9ce",
    frame_width: float = 1.1,
) -> None:
    """Draw one boxed source/result panel in figure coordinates."""
    x, y, w, h = bbox
    fig.add_artist(
        Rectangle(
            (x, y),
            w,
            h,
            transform=fig.transFigure,
            fill=False,
            edgecolor=frame_color,
            linewidth=frame_width,
            zorder=30,
            clip_on=False,
        )
    )

    pad_x = 0.018 * w
    pad_y = 0.018 * h
    title_h = 0.105 * h
    title_gap = 0.006 * h
    img_gap = 0.010 * h
    image_h = (h - 2.0 * pad_y - title_h - title_gap - img_gap) / 2.0
    image_w = w - 2.0 * pad_x

    title_ax = fig.add_axes([x + pad_x, y + h - pad_y - title_h, image_w, title_h])
    source_ax = fig.add_axes([x + pad_x, y + pad_y + image_h + img_gap, image_w, image_h])
    result_ax = fig.add_axes([x + pad_x, y + pad_y, image_w, image_h])
    for ax in (title_ax, source_ax, result_ax):
        ax.axis("off")

    title_ax.text(
        0.00,
        0.52,
        label,
        transform=title_ax.transAxes,
        fontsize=13.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.52,
        0.52,
        title,
        transform=title_ax.transAxes,
        fontsize=11.6,
        fontweight="semibold",
        ha="center",
        va="center",
    )

    source_ax.imshow(source_image)
    result_ax.imshow(result_image)
    for text, ax in (("source", source_ax), (result_label, result_ax)):
        ax.text(
            0.025,
            0.925,
            text,
            transform=ax.transAxes,
            fontsize=6.9,
            fontweight="semibold",
            ha="left",
            va="top",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        )
