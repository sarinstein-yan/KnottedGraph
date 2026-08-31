"""Contracts for the curated public API landing pages."""

from pathlib import Path
import re

import knotted_graph.visualization as visualization


ROOT = Path(__file__).resolve().parents[2]
API_ROOT = ROOT / "doc" / "api"


def test_api_toctree_includes_visualization_page():
    text = (API_ROOT / "index.md").read_text(encoding="utf-8")
    block = text.split("```{toctree}", 1)[1].split("```", 1)[0]
    entries = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith(":")
    ]

    assert entries.count("visualization") == 1


def test_visualization_page_matches_existing_public_exports():
    text = (API_ROOT / "visualization.md").read_text(encoding="utf-8")
    documented = re.findall(
        r"^\.\. (?:auto|py:)function:: "
        r"(?:knotted_graph\.visualization\.)?([A-Za-z_][A-Za-z0-9_]*)",
        text,
        flags=re.MULTILINE,
    )

    assert documented == visualization.__all__
    for name in documented:
        assert callable(getattr(visualization, name))
    assert "plot_surface_modes" not in text


def test_curated_api_pages_do_not_publish_undocumented_members():
    for name in ("layout.md", "applications.md", "visualization.md"):
        assert ":undoc-members:" not in (API_ROOT / name).read_text(encoding="utf-8")
