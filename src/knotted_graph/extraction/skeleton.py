"""Public skeleton-to-graph extraction API.

All normal 3-D calls use the second-generation sparse optimizer.  The historical
``poly2graph`` conversion remains available only through an explicit
``backend='poly2graph'`` request for regression and compatibility work.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from . import _legacy_skeleton as _legacy
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
    """Convert a 3-D skeleton with the production sparse optimizer.

    With ``max_junction_degree=None`` the conversion is topology-preserving and
    exactly compatible with the historical zero-radius parser while avoiding a
    full-volume scan.  Supplying a degree bound enables the fail-closed,
    persistence-based junction repair validated by the synthetic benchmark.
    """
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
    """Convert a skeleton image into an embedded ``networkx.MultiGraph``.

    ``backend='auto'`` selects the optimized sparse parser for every 3-D input.
    Non-3-D inputs retain the historical compatibility behavior.  The legacy
    parser can be selected explicitly with ``backend='poly2graph'``.

    ``max_junction_degree`` is intentionally optional: a known physical or
    mathematical valence bound may be supplied to enable persistent junction
    repair, while generic spatial graphs remain topology-preserving by default.
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

    if backend != "poly2graph":
        raise ValueError(
            "backend must be 'auto', 'poly2graph', or 'topology_aware'"
        )

    return _legacy.skeleton_image_to_graph(
        image,
        backend="poly2graph",
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
