"""User-facing input adapters for KnottedGraph.

These helpers convert common external data formats into the geometric objects
used by the rest of the package: embedded ``networkx.MultiGraph`` objects and
surface meshes. Surface-mesh support is loaded lazily because it depends on
PyVista.
"""

from importlib import import_module

from .coordinate_chain import (
    CoordinateInputResult,
    coordinates_to_multigraph,
    from_coordinate_chain,
    validate_coords,
)
from .mmcif import (
    MMCIFBackboneInputResult,
    from_mmcif_backbone,
)
from .pdb import (
    PDBBackboneInputResult,
    from_nucleic_acid_backbone,
    from_pdb_backbone,
    from_protein_ca_backbone,
)
from .polymer import (
    PolymerInputResult,
    from_gromacs_gro,
    from_lammps_dump,
    write_gro_coords,
    write_lammps_dump,
)
from .spatial_graph_csv import (
    SpatialGraphInputResult,
    from_spatial_graph_csv,
    validate_spatial_graph,
)

_SURFACE_MESH_EXPORTS = {
    "SurfaceInputResult",
    "from_surface_mesh",
    "validate_surface_mesh",
}


def __getattr__(name: str):
    if name not in _SURFACE_MESH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("knotted_graph.inputs.surface_mesh")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "CoordinateInputResult",
    "MMCIFBackboneInputResult",
    "PDBBackboneInputResult",
    "PolymerInputResult",
    "SpatialGraphInputResult",
    "SurfaceInputResult",
    "coordinates_to_multigraph",
    "from_coordinate_chain",
    "from_gromacs_gro",
    "from_lammps_dump",
    "from_mmcif_backbone",
    "from_nucleic_acid_backbone",
    "from_pdb_backbone",
    "from_protein_ca_backbone",
    "from_spatial_graph_csv",
    "from_surface_mesh",
    "validate_coords",
    "validate_spatial_graph",
    "validate_surface_mesh",
    "write_gro_coords",
    "write_lammps_dump",
]
