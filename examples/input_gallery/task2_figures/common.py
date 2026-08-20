"""Scheduler-independent validation and composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from .specs import FigureSpec, PanelSpec
from .style import (
    DIVIDER_WIDTH,
    FRAME_BOUNDS,
    FRAME_COLOR,
    FRAME_WIDTH,
    SAVE_DPI,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_ROOT = PACKAGE_DIR.parent / "figures"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "_build"


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Files produced by one figure render."""

    figure: str
    outputs: tuple[Path, ...]
    summary: Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_asset(panel: PanelSpec, asset_root: Path) -> Path:
    """Resolve and verify one accepted panel without allowing path escape."""

    root = asset_root.expanduser().resolve()
    path = (root / panel.asset).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Panel asset escapes the asset root: {panel.asset}") from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing accepted panel '{panel.key}': {path}\n"
            "Provide --asset-root pointing to the accepted Task 2 panel bundle."
        )
    actual = sha256_file(path)
    if actual != panel.sha256:
        raise ValueError(
            f"Accepted panel hash mismatch for '{panel.key}':\n"
            f"  path: {path}\n  expected: {panel.sha256}\n  actual:   {actual}"
        )
    return path


def validate_assets(spec: FigureSpec, asset_root: Path) -> tuple[Path, ...]:
    """Fail closed unless every panel required by *spec* is present and exact."""

    return tuple(resolve_asset(panel, asset_root) for panel in spec.panels)


def load_rgb(path: Path) -> np.ndarray:
    """Load an accepted panel as an RGB array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def draw_shared_grid(fig, *, rows: int, cols: int) -> None:
    """Draw one continuous outer frame and shared internal dividers."""

    left, right, bottom, top = FRAME_BOUNDS
    fig.add_artist(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            transform=fig.transFigure,
            fill=False,
            edgecolor=FRAME_COLOR,
            linewidth=FRAME_WIDTH,
            zorder=50,
            clip_on=False,
        )
    )
    for col in range(1, cols):
        x = left + (right - left) * col / cols
        fig.add_artist(
            Line2D(
                [x, x],
                [bottom, top],
                transform=fig.transFigure,
                color=FRAME_COLOR,
                linewidth=DIVIDER_WIDTH,
                zorder=50,
            )
        )
    for row in range(1, rows):
        y = bottom + (top - bottom) * row / rows
        fig.add_artist(
            Line2D(
                [left, right],
                [y, y],
                transform=fig.transFigure,
                color=FRAME_COLOR,
                linewidth=DIVIDER_WIDTH,
                zorder=50,
            )
        )


def panel_bbox(index: int, *, rows: int, cols: int) -> tuple[float, float, float, float]:
    """Return the figure-relative bounding box of a row-major panel."""

    left, right, bottom, top = FRAME_BOUNDS
    width = (right - left) / cols
    height = (top - bottom) / rows
    row, col = divmod(index, cols)
    return left + col * width, top - (row + 1) * height, width, height


def panel_label(index: int) -> str:
    """Return the publication label ``(a)``, ``(b)``, ... for *index*."""

    if index < 0 or index >= 26:
        raise ValueError("Panel labels support indices 0 through 25")
    return f"({chr(ord('a') + index)})"


def _save_metadata(suffix: str) -> dict[str, object]:
    creator = "KnottedGraph Task 2 figure builder"
    if suffix == "pdf":
        fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        return {"Creator": creator, "CreationDate": fixed_date, "ModDate": fixed_date}
    if suffix == "svg":
        return {"Creator": creator, "Date": "2000-01-01T00:00:00Z"}
    return {"Software": creator}


def save_figure(
    fig,
    spec: FigureSpec,
    output_dir: Path,
    *,
    s1_svg_math_as_paths: bool = False,
) -> RenderResult:
    """Write PNG, SVG, PDF, and a small provenance summary."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_dir / f"{spec.output_stem}.{suffix}" for suffix in ("png", "svg", "pdf"))
    summary = output_dir / f"{spec.output_stem}.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "figure": spec.key,
        "layout": f"{spec.rows}x{spec.cols}",
        "panel_count": len(spec.panels),
        "caption_embedded": False,
        "figure_number_embedded": False,
        "accepted_inputs_verified": True,
        "audit_variable": "A" if spec.show_yamada else None,
        "display_variable": "Y" if spec.show_yamada else None,
        "display_notation": r"\Upsilon(G;Y)" if spec.show_yamada else None,
        "panels": [
            {
                "panel": chr(ord("a") + index),
                "key": panel.key,
                "title": panel.title,
                "asset": panel.asset.as_posix(),
                "sha256": panel.sha256,
                "formats": panel.formats,
                "audit_polynomial_A": panel.polynomial,
            }
            for index, panel in enumerate(spec.panels)
        ],
    }

    # Stage every format before replacing any published output. A renderer
    # failure therefore leaves the previous complete set untouched instead of
    # exposing a zero-byte or half-written figure.
    with tempfile.TemporaryDirectory(prefix=f".{spec.output_stem}-", dir=output_dir) as directory:
        stage_dir = Path(directory)
        staged_outputs: list[Path] = []
        records: list[dict[str, object]] = []
        bbox = "tight" if spec.tight_bbox else None
        for suffix, destination in zip(("png", "svg", "pdf"), outputs, strict=True):
            staged = stage_dir / destination.name
            svg_fonttype = (
                "path"
                if suffix == "svg" and s1_svg_math_as_paths
                else mpl.rcParams["svg.fonttype"]
            )
            with mpl.rc_context({"svg.fonttype": svg_fonttype}):
                fig.savefig(
                    staged,
                    dpi=SAVE_DPI,
                    bbox_inches=bbox,
                    pad_inches=0,
                    facecolor="white",
                    edgecolor="none",
                    metadata=_save_metadata(suffix),
                )
            if not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError(f"Renderer produced an empty {suffix.upper()} for {spec.key}")
            staged_outputs.append(staged)
            records.append(
                {
                    "format": suffix,
                    "path": destination.name,
                    "bytes": staged.stat().st_size,
                    "sha256": sha256_file(staged),
                }
            )

        payload["outputs"] = records
        staged_summary = stage_dir / summary.name
        staged_summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if staged_summary.stat().st_size == 0:
            raise RuntimeError(f"Renderer produced an empty summary for {spec.key}")

        for staged, destination in zip(staged_outputs, outputs, strict=True):
            os.replace(staged, destination)
        os.replace(staged_summary, summary)

    return RenderResult(spec.key, outputs, summary)


def new_figure(spec: FigureSpec):
    """Create a styled Matplotlib figure with the shared grid already drawn."""

    fig = plt.figure(figsize=spec.figsize, facecolor="white")
    draw_shared_grid(fig, rows=spec.rows, cols=spec.cols)
    return fig


def close_figures(figures: Iterable[object]) -> None:
    """Close figures created by an interrupted multi-figure build."""

    for figure in figures:
        plt.close(figure)
