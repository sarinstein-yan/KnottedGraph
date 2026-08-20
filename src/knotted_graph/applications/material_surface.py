"""Optimized Hermitian multiband material-surface workflow.

The scientific implementation is retained in the private base module.  This
public class applies the same occupied-box Lee skeletonization used by the
non-Hermitian nodal workflow, so material masks no longer skeletonize empty
volume margins before entering the production sparse graph extractor.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import skimage.morphology as morph
from numpy.typing import NDArray

from ._material_surface_base import MaterialFermiSurface as _MaterialFermiSurfaceBase

__all__ = ["MaterialFermiSurface"]


class MaterialFermiSurface(_MaterialFermiSurfaceBase):
    """Hermitian multiband Fermi-surface workflow with optimized skeletonization.

    All Hamiltonian, gap, graph post-processing, visualization, and public API
    behavior is inherited unchanged.  Only the Lee skeletonization stage is
    specialized to crop empty margins first and restore the result into the
    original global voxel frame.
    """

    @cached_property
    def _skeleton_image(self) -> NDArray:
        """Morphological skeleton computed only on the occupied mask box."""
        mask = np.asarray(self._interior_mask, dtype=bool)
        occupied = [
            np.flatnonzero(mask.any(axis=axes))
            for axes in ((1, 2), (0, 2), (0, 1))
        ]

        if any(len(indices) == 0 for indices in occupied):
            image = np.zeros_like(mask, dtype=bool)
        else:
            slices = tuple(
                slice(
                    max(0, int(indices[0]) - 1),
                    min(mask.shape[axis], int(indices[-1]) + 2),
                )
                for axis, indices in enumerate(occupied)
            )
            skeleton_crop = morph.skeletonize(mask[slices], method="lee")
            image = np.zeros_like(mask, dtype=bool)
            image[slices] = skeleton_crop

        if np.sum(image) == 0:
            raise ValueError(
                "The skeleton image is empty. "
                "Try increasing gap_tol, checking band_pair, or enlarging the k-span."
            )
        return image
