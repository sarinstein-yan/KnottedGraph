"""Optimized Hermitian multiband material-surface workflow.

The scientific implementation is retained in the private base module.  This
public class applies the same occupied-box Lee skeletonization used by the
non-Hermitian nodal workflow, so material masks no longer skeletonize empty
volume margins before entering the production sparse graph extractor.
"""

from __future__ import annotations


from ._material_surface_base import MaterialFermiSurface as _MaterialFermiSurfaceBase

__all__ = ["MaterialFermiSurface"]


class MaterialFermiSurface(_MaterialFermiSurfaceBase):
    """Hermitian multiband Fermi-surface workflow with optimized skeletonization.

    All Hamiltonian, gap, graph post-processing, visualization, and public API
    behavior is inherited unchanged.  Only the Lee skeletonization stage is
    specialized to crop empty margins first and restore the result into the
    original global voxel frame.
    """
