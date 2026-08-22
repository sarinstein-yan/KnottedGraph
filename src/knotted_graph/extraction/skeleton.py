"""Canonical optimized skeletonization and skeleton-to-graph APIs."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from ._optimized import extract as _optimized_extract

__all__ = [
    "skeletonize_volume",
    "skeleton_image_to_graph",
    "topology_aware_skeleton_image_to_graph",
]


def skeletonize_volume(mask: ArrayLike, *, padding: int = 1) -> np.ndarray:
    """Skeletonize only the occupied 3-D bounding box and restore global indices.

    Cropping empty margins changes only the amount of work performed by Lee
    thinning. A small zero-valued padding is retained around the occupied box so
    boundary conditions match full-volume thinning for interior objects.

    ``scikit-image`` is imported only when volume skeletonization is requested;
    skeleton-to-graph extraction itself remains available with base dependencies.
    """
    image = np.asarray(mask, dtype=bool)
    if image.ndim != 3:
        raise ValueError("mask must be a three-dimensional array")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    occupied = np.argwhere(image)
    if occupied.size == 0:
        raise ValueError("The interior mask does not contain any True voxels.")

    try:
        from skimage import morphology
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency boundary
        raise ModuleNotFoundError(
            "skeletonize_volume requires scikit-image; install the nodal extra."
        ) from exc

    starts = np.maximum(occupied.min(axis=0) - padding, 0)
    stops = np.minimum(occupied.max(axis=0) + padding + 1, image.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(starts, stops))
    cropped = image[slices]
    local = morphology.skeletonize(cropped, method="lee")

    skeleton = np.zeros_like(image, dtype=bool)
    skeleton[slices] = local
    if not np.any(skeleton):
        raise ValueError("Skeletonization produced no points.")
    return skeleton


def skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Convert a 3-D skeleton image with the canonical sparse extractor."""
    return _optimized_extract(
        np.asarray(skeleton_image),
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )


def topology_aware_skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Alias for :func:`skeleton_image_to_graph` with explicit topology controls."""
    return skeleton_image_to_graph(
        skeleton_image,
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
