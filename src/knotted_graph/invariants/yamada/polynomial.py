"""Exact Yamada evaluation for projected spatial graphs.

All production state construction uses the compact immutable state tables and all
state values use the fastest available exact compact evaluator.  The optional
native extension is selected automatically, with arbitrary-precision Python as
the exact fallback.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import sympy as sp
from joblib import Parallel, delayed

from knotted_graph.projection.geom import Arc, Crossing, Vertex

from .fast import (
    ONE,
    FastNegamiSpecializedEvaluator,
    FastYamadaEvaluator,
    add as laurent_add,
    multiply as laurent_multiply,
    normalize_yamada as normalize_laurent_yamada,
    shift as laurent_shift,
    to_sympy as laurent_to_sympy,
)
from .state_compact import PreparedCompactStateBuilder

__all__ = ["compute_yamada_from_states", "Yamada"]

Port = tuple[int, str]


def _make_fast_evaluator(method: str):
    if method == "negami":
        return FastNegamiSpecializedEvaluator()
    if method == "recursive":
        return FastYamadaEvaluator()
    raise ValueError("method must be either 'negami' or 'recursive'.")


def _evaluate_fast_state(evaluator, graph, exponent: int):
    return exponent, evaluator.compute_laurent(graph)


def _sum_laurent_states_raw(evaluated_states: Iterable[tuple[int, tuple]]) -> tuple:
    total = ()
    for exponent, state_value in evaluated_states:
        total = laurent_add(total, laurent_shift(state_value, exponent))
    return total


def _sum_laurent_states(
    evaluated_states: Iterable[tuple[int, tuple]],
    variable: sp.Symbol,
    normalize: bool,
) -> sp.Expr:
    total = _sum_laurent_states_raw(evaluated_states)
    if normalize:
        total = normalize_laurent_yamada(total)
    return laurent_to_sympy(total, variable)


def compute_yamada_from_states(
    state_graphs: Sequence[Any],
    exponents: Sequence[int],
    variable: sp.Symbol,
    normalize: bool = True,
    n_jobs: int = -1,
    method: str = "negami",
) -> sp.Expr:
    """Compute Yamada from already-resolved exact compact/graph states.

    The evaluator is always the current compact exact implementation.  When the
    native extension is available, a whole state batch is accumulated in C++;
    otherwise the same compact recurrence is evaluated with arbitrary-precision
    Python integers.
    """
    if len(state_graphs) != len(exponents):
        raise ValueError("state_graphs and exponents must have the same length.")
    evaluator = _make_fast_evaluator(method)

    if hasattr(evaluator, "compute_many_laurent"):
        from .compact import CompactGraph

        compact_states = (
            (
                graph
                if isinstance(graph, CompactGraph)
                else CompactGraph.from_networkx(graph),
                exponent,
            )
            for graph, exponent in zip(state_graphs, exponents)
        )
        total = evaluator.compute_many_laurent(compact_states)
        if normalize:
            total = normalize_laurent_yamada(total)
        return laurent_to_sympy(total, variable)

    if n_jobs == 1:
        evaluated_states = (
            _evaluate_fast_state(evaluator, graph, exponent)
            for graph, exponent in zip(state_graphs, exponents)
        )
    else:
        evaluated_states = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_evaluate_fast_state)(evaluator, graph, exponent)
            for graph, exponent in zip(state_graphs, exponents)
        )
    return _sum_laurent_states(evaluated_states, variable, normalize)


def _angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _candidate_ports(arc: Arc, crossing: Crossing) -> list[tuple[Port, float]]:
    candidates: list[tuple[Port, float]] = []
    base = crossing.point
    if arc.start_type == "x" and arc.start_id == crossing.id:
        coords = arc.line.coords[1]
        angle = math.atan2(coords[1] - base.y, coords[0] - base.x)
        candidates.append(((arc.id, "s"), angle))
    if arc.end_type == "x" and arc.end_id == crossing.id:
        coords = arc.line.coords[-2]
        angle = math.atan2(coords[1] - base.y, coords[0] - base.x)
        candidates.append(((arc.id, "e"), angle))
    return candidates


def _port_z(arcs_by_id: dict[int, Arc], port: Port) -> float:
    arc = arcs_by_id[port[0]]
    coords = arc.line.coords[0] if port[1] == "s" else arc.line.coords[-1]
    return float(coords[2])


def _ordered_crossing_ports(
    crossing: Crossing,
    arcs_by_id: dict[int, Arc],
) -> list[Port]:
    """Return crossing half-edge ports in cyclic order, positions 0/2 over."""
    if len(crossing.incident_arcs) != 4:
        raise ValueError(
            f"Crossing {crossing.id} has {len(crossing.incident_arcs)} incidences."
        )

    used: set[Port] = set()
    matched: list[tuple[float, Port]] = []
    for arc_id, angle in crossing.incident_arcs:
        candidates = [
            (port, candidate_angle)
            for port, candidate_angle in _candidate_ports(arcs_by_id[arc_id], crossing)
            if port not in used
        ]
        if not candidates:
            raise ValueError(
                f"Could not map crossing {crossing.id} incidence for arc {arc_id} "
                "to a unique half-edge port."
            )
        port, _ = min(candidates, key=lambda item: _angle_delta(angle, item[1]))
        used.add(port)
        matched.append((angle, port))

    raw_ports = [port for _, port in sorted(matched, key=lambda item: item[0])]
    if _port_z(arcs_by_id, raw_ports[0]) <= _port_z(arcs_by_id, raw_ports[1]):
        raw_ports = raw_ports[1:] + raw_ports[:1]
    return raw_ports


@dataclass
class Yamada:
    vertices: list[Vertex]
    crossings: list[Crossing]
    arcs: list[Arc]

    @classmethod
    def from_PDCode(cls, PDCode: Any) -> "Yamada":
        """Create a calculator from a populated PD-code processor."""
        return cls(
            list(PDCode.vertices.values()),
            list(PDCode.crossings.values()),
            list(PDCode.arcs.values()),
        )

    def _prepare_compact_state_builder(self):
        """Prepare and exactly RII-reduce compact state tables once."""
        prepared = PreparedCompactStateBuilder.prepare(
            self.vertices,
            self.crossings,
            self.arcs,
            _ordered_crossing_ports,
        )
        return prepared.reduce_reidemeister_ii()[0]

    def _iter_compact_states(self):
        """Trace all crossing resolutions directly into compact multigraphs."""
        prepared = self._prepare_compact_state_builder()
        crossing_count = len(prepared.crossing_ids)
        for config in itertools.product((0, 1, 2), repeat=crossing_count):
            yield prepared.build(config), config.count(0) - config.count(1)

    def iter_compact_states(self):
        """Iterate exact compact states for advanced diagnostics.

        This is the inspectable state interface. It deliberately exposes the
        same compact states used by production evaluation rather than rebuilding
        a slower NetworkX representation.
        """
        yield from self._iter_compact_states()

    def _diagram_blocks(self) -> list["Yamada"]:
        """Partition a diagram into exact crossing-interaction blocks."""
        parent: dict[tuple[str, int], tuple[str, int]] = {}

        def add(node: tuple[str, int]) -> None:
            parent.setdefault(node, node)

        def find(node: tuple[str, int]) -> tuple[str, int]:
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != node:
                nxt = parent[node]
                parent[node] = root
                node = nxt
            return root

        def union(left: tuple[str, int], right: tuple[str, int]) -> None:
            add(left)
            add(right)
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for vertex in self.vertices:
            add(("v", int(vertex.id)))
        for crossing in self.crossings:
            add(("x", int(crossing.id)))
        for arc in self.arcs:
            union(
                (str(arc.start_type), int(arc.start_id)),
                (str(arc.end_type), int(arc.end_id)),
            )

        if not parent:
            return [self]
        roots = {node: find(node) for node in parent}
        unique_roots = sorted(set(roots.values()))
        if len(unique_roots) <= 1:
            return [self]

        blocks: list[Yamada] = []
        for root in unique_roots:
            vertex_ids = {
                node_id
                for (kind, node_id), node_root in roots.items()
                if node_root == root and kind == "v"
            }
            crossing_ids = {
                node_id
                for (kind, node_id), node_root in roots.items()
                if node_root == root and kind == "x"
            }
            block_vertices = [v for v in self.vertices if int(v.id) in vertex_ids]
            block_crossings = [
                crossing
                for crossing in self.crossings
                if int(crossing.id) in crossing_ids
            ]
            block_arcs = [
                arc
                for arc in self.arcs
                if roots[(str(arc.start_type), int(arc.start_id))] == root
            ]
            blocks.append(Yamada(block_vertices, block_crossings, block_arcs))
        return blocks

    def _compute_laurent_block(self, evaluator):
        prepared = self._prepare_compact_state_builder()
        if hasattr(evaluator, "compute_prepared_laurent"):
            return evaluator.compute_prepared_laurent(prepared)

        crossing_count = len(prepared.crossing_ids)
        states = (
            (prepared.build(config), config.count(0) - config.count(1))
            for config in itertools.product((0, 1, 2), repeat=crossing_count)
        )
        if hasattr(evaluator, "compute_many_laurent"):
            return evaluator.compute_many_laurent(states)
        evaluated_states = (
            _evaluate_fast_state(evaluator, graph, exponent)
            for graph, exponent in states
        )
        return _sum_laurent_states_raw(evaluated_states)

    def compute(
        self,
        variable: sp.Symbol,
        normalize: bool = True,
        n_jobs: int = -1,
        method: str = "negami",
    ) -> sp.Expr:
        """Compute Yamada through the current compact/native exact pipeline."""
        evaluator = _make_fast_evaluator(method)
        total = ONE
        for block in self._diagram_blocks():
            total = laurent_multiply(total, block._compute_laurent_block(evaluator))
        if normalize:
            total = normalize_laurent_yamada(total)
        return laurent_to_sympy(total, variable)
