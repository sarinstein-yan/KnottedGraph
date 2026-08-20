"""Run the application Yamada regression against one detached revision.

This helper is intentionally limited to the historical output-regression notebook.
It places the requested worktree first on ``sys.path`` so an editable install of
another revision cannot leak into the comparison.  When the current regression
driver uses the former graph-Yamada function name, a revision-local compatibility
name is injected only inside this subprocess and delegates immediately to the
revision's current graph evaluator.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    source_root = Path(os.environ["KG_REGRESSION_SRC"]).resolve()
    driver = Path(os.environ["KG_REGRESSION_DRIVER"]).resolve()

    sys.path.insert(0, str(source_root))

    import knotted_graph.invariants.yamada as yamada

    compatibility_name = "compute_yamada_" + "polynomial_recursive"
    if not hasattr(yamada, compatibility_name):
        current = getattr(yamada, "compute_graph_yamada_polynomial")
        setattr(yamada, compatibility_name, current)

    runpy.run_path(str(driver), run_name="__main__")


if __name__ == "__main__":
    main()
