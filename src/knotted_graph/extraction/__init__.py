"""Skeleton extraction helpers for graph-building workflows."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from . import skeleton as _legacy
from ._optimized import extract as _optimized_extract

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
    """Convert a 3-D skeleton using the production sparse optimizer."""
    return _optimized_extract(
        np.asarray(skeleton_image, dtype=bool),
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
    """Convert a skeleton image, optimizing every normal 3-D call by default.

    ``backend='auto'`` selects the optimized sparse parser for 3-D inputs while
    retaining the historical ``poly2graph`` path for non-3-D compatibility.
    ``backend='poly2graph'`` remains available explicitly for regression tests.
    """
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

    return _legacy.skeleton_image_to_graph(
        image,
        backend=backend,
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
