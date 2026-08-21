from __future__ import annotations

import random

from knotted_graph.invariants.yamada.diagram_frontier import (
    _factor_graph,
    _greedy_factor_order,
    plan_diagram_frontier,
)
from knotted_graph.invariants.yamada.skein_hybrid import _resolution_tables
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def random_crossing_diagram(seed: int, crossings: int):
    rng = random.Random(seed)
    port_count = 4 * crossings
    ordered = tuple(tuple(range(4*i, 4*i+4)) for i in range(crossings))
    ports = list(range(port_count))
    rng.shuffle(ports)
    partner = [-1] * port_count
    for left, right in zip(ports[::2], ports[1::2], strict=True):
        partner[left] = right
        partner[right] = left
    crossing_for = [-1] * port_count
    for crossing, group in enumerate(ordered):
        for port in group:
            crossing_for[port] = crossing
    plus, minus = _resolution_tables(ordered, port_count)
    return PreparedCompactStateBuilder(
        vertex_ids=(),
        crossing_ids=tuple(range(crossings)),
        ordered_ports=ordered,
        arc_partner=tuple(partner),
        fixed_terminal_index=tuple([-1] * port_count),
        crossing_for_port=tuple(crossing_for),
        plus_partner=plus,
        minus_partner=minus,
    )


def greedy_from_first(adjacency, factor_ports, first):
    count = len(adjacency)
    unprocessed = set(range(count))
    processed = {first}
    unprocessed.remove(first)
    order = [first]
    while unprocessed:
        def score(node):
            back = sum(mult for other, mult in adjacency[node].items() if other in processed)
            future = sum(mult for other, mult in adjacency[node].items() if other in unprocessed)
            disconnected = 1 if back == 0 and processed else 0
            return (disconnected, future - back, future, -back, len(factor_ports[node]), node)
        node = min(unprocessed, key=score)
        order.append(node)
        processed.add(node)
        unprocessed.remove(node)
    return order


def multistart_order(prepared):
    factor_ports, _port_factor, adjacency, _arcs = _factor_graph(prepared)
    best = None
    best_peak = None
    best_boundary = None
    for first in range(len(factor_ports)):
        order = greedy_from_first(adjacency, factor_ports, first)
        plan = plan_diagram_frontier(prepared, factor_order=order)
        key = (plan["peak_ports"], plan["max_boundary_ports"], tuple(order))
        if best is None or key < best:
            best = key
            best_peak = plan["peak_ports"]
            best_boundary = plan["max_boundary_ports"]
    return best_peak, best_boundary


def main():
    total = 0
    improved = 0
    made_eligible = 0
    deltas = []
    examples = []
    for crossings in (8, 10, 12, 14, 16):
        local = []
        for offset in range(80):
            prepared = random_crossing_diagram(220000 + 1000*crossings + offset, crossings)
            factor_ports, _port_factor, adjacency, _arcs = _factor_graph(prepared)
            current_order = _greedy_factor_order(adjacency, factor_ports)
            current = plan_diagram_frontier(prepared, factor_order=current_order)
            new_peak, new_boundary = multistart_order(prepared)
            delta = current["peak_ports"] - new_peak
            total += 1
            deltas.append(delta)
            improved += int(delta > 0)
            made_eligible += int(current["peak_ports"] > 10 and new_peak <= 10)
            local.append((delta, current["peak_ports"], new_peak, offset, current["max_boundary_ports"], new_boundary))
        local.sort(reverse=True)
        examples.extend((crossings, *row) for row in local[:3])
        avg = sum(row[0] for row in local)/len(local)
        print(f"crossings={crossings} improved={sum(row[0]>0 for row in local)}/{len(local)} avg_peak_drop={avg:.3f} best={local[0]}")
    print(f"TOTAL={total} IMPROVED={improved} ({100*improved/total:.2f}%) MADE_WIDTH10_ELIGIBLE={made_eligible}")
    print(f"MEAN_PEAK_DROP={sum(deltas)/len(deltas):.4f} MAX_PEAK_DROP={max(deltas)}")
    print("TOP_EXAMPLES")
    for row in sorted(examples, reverse=True)[:12]:
        print(row)


if __name__ == "__main__":
    main()
