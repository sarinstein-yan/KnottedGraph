import itertools
import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
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


__all__ = [
    "compute_negami",
    "compute_yamada_from_states",
    "Yamada",
]


Port = tuple[int, str]


def compute_negami(G: nx.MultiGraph, x: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    """Compute the bivariate Negami polynomial by its defining edge-subset sum.

    This explicit implementation is intentionally retained as an independent
    reference implementation. The public ``method="negami"`` Yamada backend
    uses an exact specialization of the equivalent recursive Negami evaluator.
    """

    edges = list(G.edges(keys=True))
    h_poly = sp.Integer(0)

    for r in range(len(edges) + 1):
        for F in itertools.combinations(edges, r):
            H = G.copy()
            for (u, v, key) in F:
                H.remove_edge(u, v, key=key)

            mu = nx.number_connected_components(H)
            num_vertices = H.number_of_nodes()
            num_edges = H.number_of_edges()
            beta = num_edges - num_vertices + mu

            h_poly += ((-x) ** (-r)) * (x**mu) * (y**beta)

    return sp.expand(h_poly)


def _make_fast_evaluator(method: str):
    if method == "negami":
        return FastNegamiSpecializedEvaluator()
    if method == "recursive":
        return FastYamadaEvaluator()
    raise ValueError("method must be either 'negami' or 'recursive'.")


def _evaluate_fast_state(evaluator, graph, exponent: int):
    return exponent, evaluator.compute_laurent(graph)


def _sum_laurent_states_raw(evaluated_states):
    total = ()
    for exponent, state_value in evaluated_states:
        total = laurent_add(total, laurent_shift(state_value, exponent))
    return total


def _sum_laurent_states(evaluated_states, A: sp.Symbol, normalize: bool) -> sp.Expr:
    total = _sum_laurent_states_raw(evaluated_states)
    if normalize:
        total = normalize_laurent_yamada(total)
    return laurent_to_sympy(total, A)


def compute_yamada_from_states(
    state_graphs: list[nx.MultiGraph],
    exponents: list[int],
    A: sp.Symbol,
    normalize: bool = True,
    n_jobs: int = -1,
    method: str = "negami",
) -> sp.Expr:
    """Compute the Yamada polynomial from already-resolved diagram states.

    The public result and recurrence are unchanged. State values are represented
    internally as exact integer Laurent polynomials and converted to SymPy only
    once, after summation. When the native backend is installed, the entire
    state batch is evaluated and accumulated in one C++ call.
    """

    if len(state_graphs) != len(exponents):
        raise ValueError("state_graphs and exponents must have the same length.")
    if method not in {"negami", "recursive"}:
        raise ValueError("method must be either 'negami' or 'recursive'.")

    evaluator = _make_fast_evaluator(method)
    compact_states = None
    if hasattr(evaluator, "compute_many_laurent"):
        from .compact import CompactGraph

        compact_states = (
            (
                graph if isinstance(graph, CompactGraph) else CompactGraph.from_networkx(graph),
                exponent,
            )
            for graph, exponent in zip(state_graphs, exponents)
        )
        total = evaluator.compute_many_laurent(compact_states)
        if normalize:
            total = normalize_laurent_yamada(total)
        return laurent_to_sympy(total, A)

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

    return _sum_laurent_states(evaluated_states, A, normalize)


def _finalize_yamada_total(
    total_poly: sp.Expr,
    A: sp.Symbol,
    *,
    normalize: bool,
) -> sp.Expr:
    """Reference SymPy finalizer retained for compatibility/internal tests."""
    Y = sp.expand(sp.cancel(total_poly))
    if normalize:
        terms = Y.as_ordered_terms()
        lowest_exp = min(term.as_coeff_exponent(A)[1] for term in terms)
        Y = Y * (-A) ** (-lowest_exp)
        Y = sp.expand(sp.cancel(Y))
    return Y


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
    """Return crossing half-edge ports in cyclic order, with positions 0/2 over."""

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


def _add_adjacency(adjacency: dict[Port, list[Port]], a: Port, b: Port) -> None:
    adjacency.setdefault(a, []).append(b)
    adjacency.setdefault(b, []).append(a)


def _build_state_graph_from_ports(
    vertices: list[Vertex],
    crossings: list[Crossing],
    arcs: list[Arc],
    state: tuple[int, ...],
    *,
    plus_pairs: tuple[tuple[int, int], tuple[int, int]] = ((0, 3), (1, 2)),
    minus_pairs: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (2, 3)),
) -> nx.MultiGraph:
    """Reference NetworkX state builder retained for diagnostics/regression tests."""

    if len(state) != len(crossings):
        raise ValueError("State length must match the number of crossings.")

    graph = nx.MultiGraph()
    vertex_nodes = {vertex.id: ("v", vertex.id) for vertex in vertices}
    crossing_nodes = {crossing.id: ("x", crossing.id) for crossing in crossings}
    state_by_crossing = {
        crossing.id: spin for crossing, spin in zip(crossings, state)
    }
    arcs_by_id = {arc.id: arc for arc in arcs}

    graph.add_nodes_from(vertex_nodes.values(), type="v")
    for crossing in crossings:
        if state_by_crossing[crossing.id] == 2:
            graph.add_node(crossing_nodes[crossing.id], type="x")

    adjacency: dict[Port, list[Port]] = {}
    terminal_for_port: dict[Port, object] = {}

    for arc in arcs:
        start_port = (arc.id, "s")
        end_port = (arc.id, "e")
        _add_adjacency(adjacency, start_port, end_port)

        if arc.start_type == "v":
            terminal_for_port[start_port] = vertex_nodes[arc.start_id]
        elif state_by_crossing[arc.start_id] == 2:
            terminal_for_port[start_port] = crossing_nodes[arc.start_id]

        if arc.end_type == "v":
            terminal_for_port[end_port] = vertex_nodes[arc.end_id]
        elif state_by_crossing[arc.end_id] == 2:
            terminal_for_port[end_port] = crossing_nodes[arc.end_id]

    for crossing in crossings:
        spin = state_by_crossing[crossing.id]
        if spin == 2:
            continue
        if spin not in (0, 1):
            raise ValueError(f"Invalid spin configuration: {spin}")

        ports = _ordered_crossing_ports(crossing, arcs_by_id)
        pairs = plus_pairs if spin == 0 else minus_pairs
        for a, b in pairs:
            _add_adjacency(adjacency, ports[a], ports[b])

    def edge_key(a: Port, b: Port) -> tuple[Port, Port]:
        return tuple(sorted((a, b), key=repr))  # type: ignore[return-value]

    visited: set[tuple[Port, Port]] = set()

    for start_port, start_node in list(terminal_for_port.items()):
        for first_neighbor in adjacency.get(start_port, []):
            first_edge = edge_key(start_port, first_neighbor)
            if first_edge in visited:
                continue
            visited.add(first_edge)

            previous = start_port
            current = first_neighbor
            end_node = None
            while True:
                if current in terminal_for_port:
                    end_node = terminal_for_port[current]
                    break
                choices = [node for node in adjacency.get(current, []) if node != previous]
                if not choices:
                    break
                nxt = choices[0]
                visited.add(edge_key(current, nxt))
                previous, current = current, nxt

            if end_node is not None:
                graph.add_edge(start_node, end_node)

    for port in list(adjacency):
        if port in terminal_for_port:
            continue
        if all(edge_key(port, neighbor) in visited for neighbor in adjacency.get(port, [])):
            continue

        dummy = ("loop", len(graph.nodes))
        graph.add_node(dummy)
        graph.add_edge(dummy, dummy)

        previous = None
        current = port
        while True:
            choices = [node for node in adjacency.get(current, []) if node != previous]
            if not choices:
                break
            nxt = choices[0]
            visited.add(edge_key(current, nxt))
            previous, current = current, nxt
            if current == port:
                break

    return graph


@dataclass
class Yamada:
    vertices: list[Vertex]
    crossings: list[Crossing]
    arcs: list[Arc]

    @classmethod
    def from_PDCode(cls, PDCode: Any) -> "Yamada":
        """Create a Yamada polynomial calculator from a PD-code processor."""
        return cls(
            list(PDCode.vertices.values()),
            list(PDCode.crossings.values()),
            list(PDCode.arcs.values()),
        )

    def _iter_state_graphs(self):
        """Reference NetworkX state iterator retained for compatibility."""
        for config in itertools.product([0, 1, 2], repeat=len(self.crossings)):
            yield (
                _build_state_graph_from_ports(
                    self.vertices,
                    self.crossings,
                    self.arcs,
                    config,
                ),
                config.count(0) - config.count(1),
            )

    def _build_state_graphs(self):
        """Materialize all states for diagnostic/backward-compatible callers."""
        states = list(self._iter_state_graphs())
        return (
            [graph for graph, _ in states],
            [exponent for _, exponent in states],
        )

    def _prepare_compact_state_builder(self):
        """Prepare and exactly RII-reduce the compact state tables once."""
        prepared = PreparedCompactStateBuilder.prepare(
            self.vertices,
            self.crossings,
            self.arcs,
            _ordered_crossing_ports,
        )
        return prepared.reduce_reidemeister_ii()[0]

    def _iter_compact_states(self):
        """Trace all resolutions directly into compact multigraphs."""
        prepared = self._prepare_compact_state_builder()
        crossing_count = len(prepared.crossing_ids)
        for config in itertools.product([0, 1, 2], repeat=crossing_count):
            yield prepared.build(config), config.count(0) - config.count(1)

    def _diagram_blocks(self) -> list["Yamada"]:
        """Partition the PD diagram into exact crossing-interaction blocks.

        Crossings are treated conservatively as interaction terminals joining
        all incident arcs. Therefore two graph components that cross in the
        projection remain in the same block; only pieces with no shared vertex
        and no shared projection crossing are factored. The Yamada state sum is
        exactly multiplicative across the resulting blocks.
        """
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
            start = (str(arc.start_type), int(arc.start_id))
            end = (str(arc.end_type), int(arc.end_id))
            union(start, end)

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
            block_crossings = [x for x in self.crossings if int(x.id) in crossing_ids]
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
            for config in itertools.product([0, 1, 2], repeat=crossing_count)
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
        """Compute the Yamada polynomial using the fastest exact state path.

        Independent crossing-interaction blocks are factored before state
        expansion. Within each block, static diagram connectivity and crossing
        port order are prepared once; every resolution is traced directly into
        a compact immutable multiplicity matrix. When the compiled native
        extension is available, the complete block recurrence/state sum runs in
        C++ and returns one exact Laurent tuple. If its checked int64 coefficient
        path overflows, the block is transparently recomputed with arbitrary-
        precision Python integers. No public interface or result convention is
        changed.

        ``n_jobs`` is retained for API compatibility. The compact shared-memo
        path is used for all values; native code removes the former Python-bound
        recurrence bottleneck without changing scheduling semantics.
        """
        if method not in {"negami", "recursive"}:
            raise ValueError("method must be either 'negami' or 'recursive'.")

        evaluator = _make_fast_evaluator(method)
        blocks = self._diagram_blocks()
        total = ONE
        for block in blocks:
            total = laurent_multiply(total, block._compute_laurent_block(evaluator))

        if normalize:
            total = normalize_laurent_yamada(total)
        return laurent_to_sympy(total, variable)