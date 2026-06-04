from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.yamada.pd_code import PDCode
from knotted_graph.yamada.polynomial import compute_yamada_from_states


Port = tuple[int, str]


@dataclass(frozen=True)
class FixedYamadaResult:
    raw: sp.Expr
    normalized: sp.Expr
    pd_code: str
    crossings: int


def _angle_delta(a: float, b: float) -> float:
    return abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _candidate_ports(pd: PDCode, arc_id: int, crossing_id: int) -> list[tuple[Port, float]]:
    arc = pd.arcs[arc_id]
    candidates: list[tuple[Port, float]] = []
    crossing = pd.crossings[crossing_id]
    base = crossing.point

    if arc.start_type == "x" and arc.start_id == crossing_id:
        coords = arc.line.coords[1]
        angle = np.arctan2(coords[1] - base.y, coords[0] - base.x)
        candidates.append(((arc_id, "s"), angle))

    if arc.end_type == "x" and arc.end_id == crossing_id:
        coords = arc.line.coords[-2]
        angle = np.arctan2(coords[1] - base.y, coords[0] - base.x)
        candidates.append(((arc_id, "e"), angle))

    return candidates


def _port_z(pd: PDCode, port: Port) -> float:
    arc = pd.arcs[port[0]]
    coords = arc.line.coords[0] if port[1] == "s" else arc.line.coords[-1]
    return float(coords[2])


def ordered_crossing_ports(pd: PDCode, crossing_id: int) -> list[Port]:
    """Return crossing ports in cyclic order, rotated so positions 0/2 are over."""

    crossing = pd.crossings[crossing_id]
    if len(crossing.incident_arcs) != 4:
        raise ValueError(f"Crossing {crossing_id} has {len(crossing.incident_arcs)} incidences.")

    used: set[Port] = set()
    matched: list[tuple[float, Port]] = []
    for arc_id, angle in crossing.incident_arcs:
        candidates = [
            (port, candidate_angle)
            for port, candidate_angle in _candidate_ports(pd, arc_id, crossing_id)
            if port not in used
        ]
        if not candidates:
            raise ValueError(
                f"Could not map crossing {crossing_id} incidence for arc {arc_id} "
                "to a unique half-edge port."
            )
        port, _ = min(candidates, key=lambda item: _angle_delta(angle, item[1]))
        used.add(port)
        matched.append((angle, port))

    raw_ports = [port for _, port in sorted(matched, key=lambda item: item[0])]
    if len(raw_ports) != 4:
        raise ValueError(f"Crossing {crossing_id} did not produce four ports.")

    if _port_z(pd, raw_ports[0]) <= _port_z(pd, raw_ports[1]):
        raw_ports = raw_ports[1:] + raw_ports[:1]
    return raw_ports


def _add_adjacency(adjacency: dict[Port, list[Port]], a: Port, b: Port) -> None:
    adjacency.setdefault(a, []).append(b)
    adjacency.setdefault(b, []).append(a)


def build_state_graph_ports(
    pd: PDCode,
    state: tuple[int, ...],
    *,
    plus_pairs: tuple[tuple[int, int], tuple[int, int]] = ((0, 3), (1, 2)),
    minus_pairs: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (2, 3)),
) -> nx.MultiGraph:
    """Build a resolved state graph by tracing half-edge ports globally.

    This avoids the original sequential node-removal issue where resolving one
    crossing can remove a crossing node that a later crossing resolution then
    re-adds through NetworkX.add_edge().
    """

    num_vertices = len(pd.vertices)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(num_vertices), type="v")
    for crossing_id, spin in enumerate(state):
        if spin == 2:
            graph.add_node(num_vertices + crossing_id, type="x")

    adjacency: dict[Port, list[Port]] = {}
    terminal_for_port: dict[Port, int] = {}

    for arc in pd.arcs.values():
        start_port = (arc.id, "s")
        end_port = (arc.id, "e")
        _add_adjacency(adjacency, start_port, end_port)

        if arc.start_type == "v":
            terminal_for_port[start_port] = arc.start_id
        elif state[arc.start_id] == 2:
            terminal_for_port[start_port] = num_vertices + arc.start_id

        if arc.end_type == "v":
            terminal_for_port[end_port] = arc.end_id
        elif state[arc.end_id] == 2:
            terminal_for_port[end_port] = num_vertices + arc.end_id

    for crossing_id, spin in enumerate(state):
        if spin == 2:
            continue
        ports = ordered_crossing_ports(pd, crossing_id)
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

    # Closed components with no original/s0 vertex become simple loop components.
    for port in list(adjacency):
        if port in terminal_for_port:
            continue
        if all(edge_key(port, neighbor) in visited for neighbor in adjacency.get(port, [])):
            continue

        dummy = max(graph.nodes, default=-1) + 1
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


def compute_yamada_fixed_copy(
    pd: PDCode,
    variable: sp.Symbol,
    *,
    normalize: bool = True,
    n_jobs: int = 1,
) -> sp.Expr:
    state_graphs = []
    exponents = []
    for state in itertools.product([0, 1, 2], repeat=len(pd.crossings)):
        state_graphs.append(build_state_graph_ports(pd, state))
        exponents.append(state.count(0) - state.count(1))
    return compute_yamada_from_states(
        state_graphs,
        exponents,
        variable,
        normalize=normalize,
        n_jobs=n_jobs,
    )


def compute_graph_fixed_copy(
    graph: nx.MultiGraph,
    variable: sp.Symbol,
    *,
    rotation_angles=(0.0, 0.0, 0.0),
    rotation_order: str = "ZYX",
    tolerance: float = 1e-7,
) -> FixedYamadaResult:
    pd = PDCode(graph, tolerance=tolerance)
    code = pd.compute(rotation_angles=rotation_angles, rotation_order=rotation_order)
    raw = sp.expand(compute_yamada_fixed_copy(pd, variable, normalize=False, n_jobs=1))
    normalized = sp.expand(compute_yamada_fixed_copy(pd, variable, normalize=True, n_jobs=1))
    return FixedYamadaResult(
        raw=raw,
        normalized=normalized,
        pd_code=code,
        crossings=len(pd.crossings),
    )
