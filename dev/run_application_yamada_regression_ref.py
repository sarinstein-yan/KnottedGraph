"""Run the application Yamada regression against one detached revision.

This helper is intentionally limited to the historical application-output regression.
It places the requested worktree's ``src`` directory first on ``sys.path`` so an
editable install of another revision cannot leak into the comparison. A tiny API
compatibility shim is installed only inside this subprocess so the current regression
driver can execute against ``Latest_Workplace`` as well as the current optimized head.
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

    # Current optimized revisions expose compute_graph_yamada_polynomial; the
    # historical Latest_Workplace revision exposes compute_yamada_polynomial_recursive.
    # Make either spelling available without changing either revision's implementation.
    if not hasattr(yamada, "compute_graph_yamada_polynomial"):
        yamada.compute_graph_yamada_polynomial = yamada.compute_yamada_polynomial_recursive
    if not hasattr(yamada, "compute_yamada_polynomial_recursive"):
        yamada.compute_yamada_polynomial_recursive = yamada.compute_graph_yamada_polynomial

    runpy.run_path(str(driver), run_name="__main__")


if __name__ == "__main__":
    main()
