"""Shared publication style for the current Task 2 figures."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl


RC_PARAMS = {
    "font.family": "serif",
    # The accepted Vanda PDFs embed DejaVu Serif. A fallback chain beginning
    # with Times changes text metrics on machines where Times is installed.
    "font.serif": ["DejaVu Serif"],
    "figure.dpi": 220,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "knottedgraph-task2-figures",
    "savefig.facecolor": "white",
    # Explicitly neutralize any global ``tight`` setting imported by unrelated
    # plotting code. Main uses the full canvas; S1/S2 request tight bounds at
    # their own save call.
    "savefig.bbox": None,
}

FRAME_BOUNDS = (0.006, 0.994, 0.006, 0.994)
FRAME_COLOR = "#31363A"
FRAME_WIDTH = 0.92
DIVIDER_WIDTH = 0.64
SAVE_DPI = 450


@contextmanager
def publication_style():
    """Use a clean, fully restored rc context without changing the backend."""

    defaults = dict(mpl.rcParamsDefault)
    defaults.pop("backend", None)
    with mpl.rc_context():
        mpl.rcParams.update(defaults)
        mpl.rcParams.update(RC_PARAMS)
        yield
