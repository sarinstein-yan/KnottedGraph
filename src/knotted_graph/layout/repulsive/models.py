from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class CurveNetwork:
    """A small spatial curve network prepared for Repulsor optimization."""

    name: str
    node_order: tuple[str, ...]
    node_positions: dict[str, np.ndarray]
    arc_order: tuple[str, ...]
    arc_polylines: dict[str, np.ndarray]
    arc_specs: dict[str, str] = field(default_factory=dict)
    node_colors: dict[str, str] = field(default_factory=dict)
    arc_colors: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepulsiveLayoutResult:
    """Paths and metadata produced by one repulsive-layout run."""

    workspace: Path
    metadata: dict[str, Any]

    @property
    def initial_html(self) -> Path:
        return Path(self.metadata["initial_html"])

    @property
    def final_html(self) -> Path:
        return Path(self.metadata["final_html"])

    @property
    def final_obj(self) -> Path:
        return Path(self.metadata["final_obj"])


@dataclass
class GraphLayoutResult:
    """A relaxed NetworkX spatial graph and files produced by one layout run."""

    graph: nx.MultiGraph
    workspace: Path
    metadata: dict[str, Any]

    @property
    def final_obj(self) -> Path:
        return Path(self.metadata["final_obj"])
