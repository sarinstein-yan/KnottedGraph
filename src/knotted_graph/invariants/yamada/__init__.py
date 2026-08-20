"""Exact optimized Yamada polynomial APIs."""

from __future__ import annotations

import networkx as nx
import sympy as sp

from .algebra import laurent_y_to_sigma_polynomial
from .compact import CompactYamadaEvaluator
from .polynomial import (
    Yamada,
    compute_yamada_from_states,
)


def compute_graph_yamada_polynomial(
    graph: nx.MultiGraph,
    variable: sp.Symbol,
) -> sp.Expr:
    """Compute the crossing-free Yamada polynomial with the fastest exact backend."""
    return CompactYamadaEvaluator().compute(graph, variable)


__all__ = [
    "Yamada",
    "compute_graph_yamada_polynomial",
    "compute_yamada_from_states",
    "laurent_y_to_sigma_polynomial",
]
