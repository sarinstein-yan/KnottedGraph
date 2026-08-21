"""Performance-only ordering refinement for exact Yamada frontier evaluation.

The ordinary frontier planner is intentionally cheap.  If its proposed order is
already within the configured production width target, this module returns it
unchanged.  Only a diagram that would otherwise be rejected pays for a
multi-start greedy search.  The chosen order affects runtime only, never the
polynomial.
"""

from __future__ import annotations

from .diagram_frontier import _factor_graph, plan_diagram_frontier


def _greedy_from_first(adjacency, factor_ports, first: int):
    count = len(adjacency)
    if count <= 1:
        return list(range(count))
    unprocessed = set(range(count))
    processed = {first}
    unprocessed.remove(first)
    order = [first]

    while unprocessed:
        def score(node):
            back = sum(
                multiplicity
                for other, multiplicity in adjacency[node].items()
                if other in processed
            )
            future = sum(
                multiplicity
                for other, multiplicity in adjacency[node].items()
                if other in unprocessed
            )
            disconnected = 1 if back == 0 and processed else 0
            return (
                disconnected,
                future - back,
                future,
                -back,
                len(factor_ports[node]),
                node,
            )

        node = min(unprocessed, key=score)
        order.append(node)
        processed.add(node)
        unprocessed.remove(node)
    return order


def plan_frontier_to_target(prepared, target_peak_ports: int):
    """Improve a rejected greedy order using deterministic multi-start search.

    Fast path: return the ordinary plan immediately when it already meets the
    target.  Slow path: reuse the same greedy score from every possible starting
    factor and retain the lexicographically best structural plan.
    """
    target_peak_ports = int(target_peak_ports)
    initial = plan_diagram_frontier(prepared)
    if initial["peak_ports"] <= target_peak_ports:
        return initial

    factor_ports, _port_factor, adjacency, _arcs = _factor_graph(prepared)
    if len(factor_ports) <= 1:
        return initial

    best = initial
    best_key = (
        int(initial["peak_ports"]),
        int(initial["max_boundary_ports"]),
        tuple(initial["factor_order"]),
    )
    candidates = 1
    initial_first = initial["factor_order"][0]
    for first in range(len(factor_ports)):
        if first == initial_first:
            continue
        order = _greedy_from_first(adjacency, factor_ports, first)
        plan = plan_diagram_frontier(prepared, factor_order=order)
        candidates += 1
        key = (
            int(plan["peak_ports"]),
            int(plan["max_boundary_ports"]),
            tuple(plan["factor_order"]),
        )
        if key < best_key:
            best = plan
            best_key = key

    if best is initial:
        return {
            **initial,
            "ordering_multistart": True,
            "ordering_candidates": candidates,
            "initial_peak_ports": int(initial["peak_ports"]),
        }
    return {
        **best,
        "ordering_multistart": True,
        "ordering_candidates": candidates,
        "initial_peak_ports": int(initial["peak_ports"]),
    }
