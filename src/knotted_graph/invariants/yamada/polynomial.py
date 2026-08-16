import itertools
import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import sympy as sp
from joblib import Parallel, delayed

from knotted_graph.projection.geom import Arc, Crossing, Vertex

from .recursive import (
    NegamiRecursiveEvaluator,
    YamadaRecursiveEvaluator,
)


__all__ = [
    "compute_negami",
    "compute_yamada_from_states",
    "Yamada",
]


Port = tuple[int, str]


def compute_negami(G: nx.MultiGraph, x: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    """Compute the bivariate Negami polynomial by its defining edge-subset sum.

    This explicit implementation is intentionally retained as an independent
    reference implementation.  The public ``method="negami"`` Yamada backend
    uses the equivalent recursive Negami evaluator for speed.
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


def compute_yamada_from_states(
    state_graphs: list[nx.MultiGraph],
    exponents: list[int],
    A: sp.Symbol,
    normalize: bool = True,
    n_jobs: int = -1,
    method: str = "negami",
) -> sp.Expr:
    """Compute the Yamada polynomial from resolved diagram states.

    Parameters
    ----------
    state_graphs
        Resolved state graphs.
    exponents
        State exponents corresponding to ``p(s) - m(s)``.
    A
        Polynomial variable.
    normalize
        If true, shift the lowest exponent to zero.
    n_jobs
        Number of thread-parallel jobs used for state-graph evaluation.
    method
        Backend for crossing-free state graphs:

        - ``"negami"`` uses the recursive evaluation of Yamada's auxiliary
          Negami specialization ``h(G;x,y)`` and then substitutes
          ``x=-1`` and ``y=-A-2-A**(-1)``;
        - ``"recursive"`` evaluates ``H(G)`` directly with the Yamada
          deletion-contraction identities.

        Both backends share memoized subproblems across resolved crossing states.

    Returns
    -------
    sympy.Expr
        The Yamada polynomial.
    """

    if len(state_graphs) != len(exponents):
        raise ValueError("state_graphs and exponents must have the same length.")
    if method not in {"negami", "recursive"}:
        raise ValueError("method must be either 'negami' or 'recursive'.")

    if method == "negami":
        x, y = sp.symbols("x y")
        evaluator = NegamiRecursiveEvaluator(x, y)
        state_values = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
        )(
            delayed(evaluator.compute)(G)
            for G in state_graphs
        )
        state_values = [
            sp.expand(
                h_val.xreplace(
                    {x: -1, y: -A - 2 - A ** (-1)}
                )
            )
            for h_val in state_values
        ]
    else:
        evaluator = YamadaRecursiveEvaluator(A)
        state_values = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
        )(
            delayed(evaluator.compute)(G)
            for G in state_graphs
        )

    total_poly = sp.Integer(0)
    for state_value, exp in zip(state_values, exponents):
        total_poly += (A**exp) * state_value

    return _finalize_yamada_total(
        total_poly,
        A,
        normalize=normalize,
    )


def _finalize_yamada_total(
    total_poly: sp.Expr,
    A: sp.Symbol,
    *,
    normalize: bool,
) -> sp.Expr:
    Y = sp.expand(sp.cancel(total_poly))
    if normalize:
        terms = Y.as_ordered_terms()
        lowest_exp = min(term.as_coeff_exponent(A)[1] for term in terms)
        Y = Y * (-A) ** (-lowest_exp)
        Y = sp.expand(sp.cancel(Y))
    return Y


def _evaluate_state_with_exponent(evaluator, graph: nx.MultiGraph, exponent: int):
    return exponent, evaluator.compute(graph)


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
    """Build a resolved state graph by tracing half-edge ports globally.

    The older implementation resolved crossings by mutating a graph in place.
    When an arc connected two crossings, resolving the later crossing could
    re-add a crossing node that had already been removed. Tracing ports first
    avoids that order dependence and also handles self-crossing ports with
    duplicate arc IDs.
    """

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

    # Components with no original or unresolved crossing terminal are loops.
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

    def compute(
        self,
        variable: sp.Symbol,
        normalize: bool = True,
        n_jobs: int = -1,
        method: str = "negami",
    ) -> sp.Expr:
        """Compute the Yamada polynomial without materializing all state graphs."""
        if method not in {"negami", "recursive"}:
            raise ValueError("method must be either 'negami' or 'recursive'.")

        if method == "negami":
            x, y = sp.symbols("x y")
            evaluator = NegamiRecursiveEvaluator(x, y)
        else:
            evaluator = YamadaRecursiveEvaluator(variable)

        evaluated_states = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
        )(
            delayed(_evaluate_state_with_exponent)(
                evaluator,
                graph,
                exponent,
            )
            for graph, exponent in self._iter_state_graphs()
        )

        total_poly = sp.Integer(0)
        for exponent, state_value in evaluated_states:
            if method == "negami":
                state_value = sp.expand(
                    state_value.xreplace(
                        {
                            x: -1,
                            y: -variable - 2 - variable ** (-1),
                        }
                    )
                )
            total_poly += (variable**exponent) * state_value

        return _finalize_yamada_total(
            total_poly,
            variable,
            normalize=normalize,
        )
