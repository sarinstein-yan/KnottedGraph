"""Generic public API for spatial graph computation and invariants."""

from knotted_graph._benchmark_cache import install_benchmark_cache as _install_benchmark_cache

_install_benchmark_cache()

from knotted_graph import core as _core
from knotted_graph import projection as _projection
from knotted_graph.invariants import yamada as _yamada

from knotted_graph.core import *
from knotted_graph.projection import *
from knotted_graph.invariants.yamada import *

__version__ = "0.2.0"
__description__ = (
    "A computational package for spatial graphs and graph polynomial invariants."
)

__all__ = (
    _core.__all__
    + _projection.__all__
    + _yamada.__all__
)
