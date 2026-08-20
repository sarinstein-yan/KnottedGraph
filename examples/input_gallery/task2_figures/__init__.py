"""Publication figure builders for the three current Task 2 figures.

The package intentionally contains only the final Main, S1, and S2
composition code.  Scientific panel renders are accepted external inputs and
are verified against the SHA-256 values in :mod:`.specs` before use.
"""

from __future__ import annotations

from typing import Any


def render_main(**kwargs: Any):
    """Lazily import and render the current Main figure."""

    from .main import render_main as _render_main

    return _render_main(**kwargs)


def render_s1(**kwargs: Any):
    """Lazily import and render the current S1 figure."""

    from .s1 import render_s1 as _render_s1

    return _render_s1(**kwargs)


def render_s2(**kwargs: Any):
    """Lazily import and render the current S2 figure."""

    from .s2 import render_s2 as _render_s2

    return _render_s2(**kwargs)

__all__ = ["render_main", "render_s1", "render_s2"]
