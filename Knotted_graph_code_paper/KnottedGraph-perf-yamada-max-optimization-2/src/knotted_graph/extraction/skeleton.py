"""Convert skeletonized images into spatial graph objects."""

from __future__ import annotations

import networkx as nx
from numpy.typing import ArrayLike

__all__ = ["skeleton_image_to_graph"]


def skeleton_image_to_graph(skeleton_image: ArrayLike) -> nx.MultiGraph:
    """Convert a skeletonized image into a ``networkx.MultiGraph``.

    This helper keeps the optional ``poly2graph`` dependency out of the generic
    package import path.
    """
    from poly2graph import skeleton2graph

    return skeleton2graph(skeleton_image)
