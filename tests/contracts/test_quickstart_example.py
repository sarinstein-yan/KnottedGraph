"""Keep the maintained quick-start example deterministic and nontrivial."""

import runpy
from pathlib import Path

import sympy as sp


EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "quickstart.py"


def test_quickstart_abstract_and_embedded_results_agree():
    namespace = runpy.run_path(str(EXAMPLE_PATH))
    abstract, embedded = namespace["compute_quickstart"]()

    Y = sp.Symbol("Y")
    expected = -Y**2 - Y - 2 - Y**-1 - Y**-2

    assert abstract != 0
    assert sp.simplify(abstract - expected) == 0
    assert sp.simplify(embedded.polynomial - expected) == 0
    assert embedded.projection.num_crossings == 0
