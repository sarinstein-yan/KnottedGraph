"""Exact low-arity factorization for the Yamada connectivity state sum.

A fixed graph vertex is an equality constraint on all incident ports with one
weight -1. High arity is therefore representational, not mathematical. We
factor each such equality into a deterministic chain of arity <=3 equality
factors joined by identity wires. Exactly one factor carries the original -1
weight; auxiliary equality factors carry +1.

Identity wires are logical vertex identifications, not physical graph edges, so
they carry no edge sign. If an identity quotient closes an already selected
physical path, however, the graph cycle rank increases and the state receives
the usual +q = A^-1 + 2 + A cycle factor. An included physical edge carries
-1 and therefore contributes -q when it closes a cycle.

This is the sole production diagram evaluator. It contains no runtime-, family-,
crossing-count-, or benchmark-dependent dispatch. A missing compiled factorized
extension is treated as an installation error rather than silently selecting an
older diagram algorithm.
"""

from __future__ import annotations

from collections import defaultdict

from .diagram_frontier import _greedy_factor_order, compute_diagram_frontier_laurent

_FACTORIZED_IMPORT_ERROR: Exception | None = None
try:
    from . import _yamada_factorized_frontier
except Exception as exc:  # pragma: no cover - compiler/platform installation guard
    _yamada_factorized_frontier = None
    _FACTORIZED_IMPORT_ERROR = exc

FACTOR_EQUALITY_NEG = 0
FACTOR_EQUALITY_POS = 1
FACTOR_CROSSING = 2
WIRE_PHYSICAL = 0
WIRE_IDENTITY = 1


def native_factorized_available() -> bool:
    return _yamada_factorized_frontier is not None


def factorized_import_error() -> Exception | None:
    return _FACTORIZED_IMPORT_ERROR


def require_native_factorized() -> None:
    """Fail clearly instead of silently selecting a superseded diagram backend."""
    if native_factorized_available():
        return
    detail = f" Original import error: {_FACTORIZED_IMPORT_ERROR!r}." if _FACTORIZED_IMPORT_ERROR else ""
    raise RuntimeError(
        "The optimized factorized Yamada extension is not available. Rebuild the "
        "current checkout in the active environment, for example with "
        "`python -m pip install -e '.[benchmark]' --force-reinstall`." + detail
    )


def build_factorized_frontier(prepared):
    vertex_count = len(prepared.vertex_ids)
    crossing_count = len(prepared.crossing_ids)
    original_port_count = len(prepared.arc_partner)

    vertex_ports = [[] for _ in range(vertex_count)]
    crossing_ports = [[] for _ in range(crossing_count)]
    for port in range(original_port_count):
        fixed = int(prepared.fixed_terminal_index[port])
        crossing = int(prepared.crossing_for_port[port])
        if fixed >= 0:
            vertex_ports[fixed].append(port)
        elif crossing >= 0:
            crossing_ports[crossing].append(port)
        else:
            raise RuntimeError("prepared Yamada port belongs to no factor")

    def neighbor_key(port):
        partner = int(prepared.arc_partner[port])
        fixed = int(prepared.fixed_terminal_index[partner])
        if fixed >= 0:
            return (0, fixed, partner)
        return (1, int(prepared.crossing_for_port[partner]), partner)

    wire_partner = [int(value) for value in prepared.arc_partner]
    wire_type = [WIRE_PHYSICAL] * original_port_count
    port_factor = [-1] * original_port_count
    factor_types = []
    factor_ports = []

    for ports in vertex_ports:
        ordered = sorted(ports, key=neighbor_key)
        if not ordered:
            factor_types.append(FACTOR_EQUALITY_NEG)
            factor_ports.append([])
            continue

        previous_right = None
        for segment_index, original_port in enumerate(ordered):
            factor = len(factor_types)
            local_ports = [original_port]
            port_factor[original_port] = factor

            if previous_right is not None:
                left = len(wire_partner)
                wire_partner.append(previous_right)
                wire_type.append(WIRE_IDENTITY)
                wire_partner[previous_right] = left
                wire_type[previous_right] = WIRE_IDENTITY
                port_factor.append(factor)
                local_ports.append(left)

            if segment_index + 1 < len(ordered):
                right = len(wire_partner)
                wire_partner.append(-1)
                wire_type.append(WIRE_IDENTITY)
                port_factor.append(factor)
                local_ports.append(right)
                previous_right = right
            else:
                previous_right = None

            factor_types.append(
                FACTOR_EQUALITY_NEG if segment_index == 0 else FACTOR_EQUALITY_POS
            )
            factor_ports.append(local_ports)

    crossing_factor_by_index = []
    for ports in crossing_ports:
        if len(ports) != 4:
            raise RuntimeError("prepared crossing must have exactly four ports")
        factor = len(factor_types)
        factor_types.append(FACTOR_CROSSING)
        local_ports = sorted(ports)
        factor_ports.append(local_ports)
        crossing_factor_by_index.append(factor)
        for port in local_ports:
            port_factor[port] = factor

    if any(value < 0 for value in port_factor):
        raise RuntimeError("factorized Yamada port has no owner")
    if any(value < 0 for value in wire_partner):
        raise RuntimeError("unpaired factorization identity wire")

    adjacency = [defaultdict(int) for _ in factor_types]
    for port, partner in enumerate(wire_partner):
        if port >= partner:
            continue
        left = port_factor[port]
        right = port_factor[partner]
        if left != right:
            adjacency[left][right] += 1
            adjacency[right][left] += 1
    factor_order = _greedy_factor_order(adjacency, factor_ports)

    plus_partner = [-1] * len(wire_partner)
    minus_partner = [-1] * len(wire_partner)
    for ports in prepared.ordered_ports:
        for port in ports:
            plus_partner[port] = int(prepared.plus_partner[port])
            minus_partner[port] = int(prepared.minus_partner[port])

    return {
        "factor_types": tuple(factor_types),
        "port_factor": tuple(port_factor),
        "wire_partner": tuple(wire_partner),
        "wire_type": tuple(wire_type),
        "plus_partner": tuple(plus_partner),
        "minus_partner": tuple(minus_partner),
        "factor_order": tuple(factor_order),
        "original_port_count": original_port_count,
        "crossing_factor_by_index": tuple(crossing_factor_by_index),
    }


def compute_factorized_frontier_laurent(prepared):
    """Evaluate one prepared Yamada diagram with the sole production DP."""
    require_native_factorized()
    data = build_factorized_frontier(prepared)
    try:
        # pybind11's STL caster accepts Python tuples directly for std::vector,
        # so avoid seven redundant O(n) list materializations on every call.
        value = _yamada_factorized_frontier.compute_factorized_frontier(
            data["factor_types"],
            data["port_factor"],
            data["wire_partner"],
            data["wire_type"],
            data["plus_partner"],
            data["minus_partner"],
            data["factor_order"],
        )
    except OverflowError:
        # This is an exact arbitrary-precision arithmetic safety net, not solver
        # dispatch: it is reached only if the optimized int64 kernel overflows.
        return compute_diagram_frontier_laurent(prepared)
    return tuple((int(power), int(coefficient)) for power, coefficient in value)
