"""Projection, planar diagram, and PD-code utilities."""

from importlib import import_module

from .geom import Arc, Crossing, Vertex
from .rotations import cut_line_string, generate_isotopy_angles, get_rotation_matrix

_PD_CODE_EXPORTS = {
    "PDCode",
    "ProjectionResult",
    "YamadaComputationResult",
    "compute_pd_code",
    "compute_yamada_polynomial",
    "explode_to_segments",
    "find_all_crossings",
    "project_crossings_on_edge",
    "sample_projections",
    "select_projection",
}

_PLANAR_DIAGRAM_EXPORTS = {
    "PlanarDiagram",
}

_LAZY_EXPORTS = {
    **{name: "knotted_graph.projection.pd_code" for name in _PD_CODE_EXPORTS},
    **{name: "knotted_graph.projection.planar_diagram" for name in _PLANAR_DIAGRAM_EXPORTS},
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "Arc",
    "Crossing",
    "Vertex",
    "cut_line_string",
    "generate_isotopy_angles",
    "get_rotation_matrix",
    *_PD_CODE_EXPORTS,
    *_PLANAR_DIAGRAM_EXPORTS,
]
