"""Skeleton extraction helpers for graph-building workflows."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from . import skeleton as _skeleton
from ._sparse_compat import trace_zero_radius_compatible

__all__ = [
    "skeleton_image_to_graph",
    "topology_aware_skeleton_image_to_graph",
]


def topology_aware_skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Convert a 3-D skeleton with sparse exact-compatible default semantics."""
    image = np.asarray(skeleton_image, dtype=bool)
    if image.ndim != 3:
        raise ValueError("skeleton_image must be a three-dimensional array")

    if max_junction_degree is None:
        _, coords, adjacency = _skeleton._sparse_voxel_adjacency(image)
        return trace_zero_radius_compatible(coords, adjacency)

    return _skeleton.topology_aware_skeleton_image_to_graph(
        image,
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )


def skeleton_image_to_graph(
    skeleton_image: ArrayLike,
    *,
    backend: str = "auto",
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    """Convert a skeleton image, optimizing every normal 3-D call by default."""
    image = np.asarray(skeleton_image)
    if backend == "auto":
        backend = "topology_aware" if image.ndim == 3 else "poly2graph"

    if backend == "topology_aware":
        return topology_aware_skeleton_image_to_graph(
            image,
            max_junction_degree=max_junction_degree,
            adaptive_max_hops=adaptive_max_hops,
            anomaly_ratio=anomaly_ratio,
        )

    return _skeleton.skeleton_image_to_graph(
        image,
        backend=backend,
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
