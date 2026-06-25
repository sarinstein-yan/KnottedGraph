"""Public API for the generic KnottedGraph package.

The package-level import is intentionally lightweight. Domain-specific
application workflows such as ``NodalSkeleton`` and optional visualization or
surface-mode helpers are loaded only when their exported names are requested.
"""

from importlib import import_module

from knotted_graph import examples as _examples
from knotted_graph import util as _util
from knotted_graph.yamada import geom as _yamada_geom
from knotted_graph.yamada import pd_code as _yamada_pd_code
from knotted_graph.yamada import planar_diagram as _yamada_planar_diagram
from knotted_graph.yamada import polynomial as _yamada_polynomial
from knotted_graph.yamada import util as _yamada_util

from knotted_graph.examples import *
from knotted_graph.util import *
from knotted_graph.yamada import *

__version__ = "0.1.3"
__description__ = (
    "A general computational package for spatial graphs and their invariants."
)

_POLY2GRAPH_EXPORTS = [
    "kron_batch",
    "eig_batch",
    "skeleton2graph",
    "skeleton2graph_batch",
    "shift_matrix",
    "hk2hz",
    "hz2hk",
    "expand_hz_as_hop_dict",
]

_VIS_EXPORTS = [
    "standard_petersen_layout",
    "draw_petersen_embedding",
    "plot_3D_and_projections_plotly",
    "plot_3D_graph_plotly",
    "plot_surface_modes",
]

_SURFACE_MODE_EXPORTS = [
    "hop_dict_by_direction",
    "H_batch_from_hop_dict",
    "H_batch",
]

_LAZY_EXPORT_MODULES = {
    "NodalSkeleton": "knotted_graph.NodalSkeleton",
    **{name: "poly2graph" for name in _POLY2GRAPH_EXPORTS},
    **{name: "knotted_graph.vis" for name in _VIS_EXPORTS},
    **{name: "knotted_graph.surface_modes" for name in _SURFACE_MODE_EXPORTS},
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = (
    ["NodalSkeleton"]
    + _POLY2GRAPH_EXPORTS
    + _yamada_geom.__all__
    + _yamada_util.__all__
    + _yamada_polynomial.__all__
    + _yamada_pd_code.__all__
    + _yamada_planar_diagram.__all__
    + _util.__all__
    + _VIS_EXPORTS
    + _SURFACE_MODE_EXPORTS
    + _examples.__all__
)
