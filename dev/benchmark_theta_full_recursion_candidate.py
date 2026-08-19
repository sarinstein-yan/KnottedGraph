"""Benchmark generic exact crossing-recursion memoization candidates.

The Dobrynin--Vesnin formula is used only after evaluation as a correctness
oracle. It is never called by the candidate evaluators.
"""

from __future__ import annotations

import json
import time

import benchmark_topoly_essential_torus_scaling as torus
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import (
    _skein_delta,
    invert_crossing,
    resolve_crossing,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def diagram_key(prepared):
    return (
        len(prepared.vertex_ids),
        prepared.ordered_ports,
        prepared.arc_partner,
        prepared.fixed_terminal_index,
        prepared.crossing_for_port,
    )


def first_reducing_inversion(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if moves:
            return crossing_index, reduced
    return None


def first_resolvable_crossing(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        try:
            children = tuple(
                resolve_crossing(prepared, crossing_index, spin)
                for spin in (0, 1, 2)
            )
        except ValueError:
            continue
        return crossing_index, children
    return None


def _relation_graph(prepared):
    """Return a finite directed colored graph encoding the prepared diagram.

    Node colors distinguish over/under crossing ports, graph-terminal ports and
    unlabeled spatial-graph vertices. Edge colors distinguish physical arcs,
    directed crossing cyclic order and terminal incidence. Thus graph
    isomorphism is exactly the relabeling freedom we want to quotient out.
    """
    port_count = len(prepared.arc_partner)
    parity = [-1] * port_count
    for ports in prepared.ordered_ports:
        for position, port in enumerate(ports):
            parity[port] = position % 2

    terminal_values = sorted(
        {terminal for terminal in prepared.fixed_terminal_index if terminal >= 0}
    )
    terminal_node = {
        terminal: port_count + index
        for index, terminal in enumerate(terminal_values)
    }

    labels = []
    for port in range(port_count):
        if prepared.crossing_for_port[port] >= 0:
            labels.append(parity[port])  # 0=over port, 1=under port
        else:
            labels.append(2)  # terminal port
    labels.extend([3] * len(terminal_values))  # unlabeled spatial-graph vertex

    edges = []
    # Physical arc pairing, encoded in both directions.
    for port, partner in enumerate(prepared.arc_partner):
        if port < partner:
            edges.append((port, 0, partner))
            edges.append((partner, 0, port))
    # Directed cyclic order at each crossing.
    for ports in prepared.ordered_ports:
        for left, right in zip(ports, ports[1:] + ports[:1]):
            edges.append((left, 1, right))
    # Unordered terminal incidence, encoded in both directions.
    for port, terminal in enumerate(prepared.fixed_terminal_index):
        if terminal >= 0:
            vnode = terminal_node[terminal]
            edges.append((vnode, 2, port))
            edges.append((port, 2, vnode))
    return tuple(labels), tuple(edges)


def _canonical_colored_digraph(labels, edges):
    """Exact canonical form by color refinement + individualization.

    This is a small nauty-style canonical-labeling search. Color refinement is
    deterministic and relabeling-invariant; if it is not discrete, every node
    in one ambiguous color class is individualized in turn and the
    lexicographically minimum recursively refined representation is selected.
    Hence equality of returned tuples is exact graph isomorphism, not a hash.
    """
    n = len(labels)
    outgoing = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    for left, kind, right in edges:
        outgoing[left].append((kind, right))
        incoming[right].append((kind, left))

    initial_values = sorted(set(labels))
    initial_map = {value: index for index, value in enumerate(initial_values)}
    initial = tuple(initial_map[value] for value in labels)
    search_memo = {}

    def refine(colors):
        colors = tuple(colors)
        while True:
            signatures = []
            for node in range(n):
                neighborhood = []
                neighborhood.extend(
                    (0, kind, colors[neighbor])
                    for kind, neighbor in outgoing[node]
                )
                neighborhood.extend(
                    (1, kind, colors[neighbor])
                    for kind, neighbor in incoming[node]
                )
                signatures.append((colors[node], tuple(sorted(neighborhood))))
            unique = sorted(set(signatures))
            mapping = {signature: index for index, signature in enumerate(unique)}
            new_colors = tuple(mapping[signature] for signature in signatures)
            if len(set(new_colors)) == len(set(colors)):
                return new_colors
            colors = new_colors

    def canonical(colors):
        colors = refine(colors)
        cached = search_memo.get(colors)
        if cached is not None:
            return cached

        classes = {}
        for node, color in enumerate(colors):
            classes.setdefault(color, []).append(node)
        ambiguous = [
            (len(nodes), color, nodes)
            for color, nodes in classes.items()
            if len(nodes) > 1
        ]
        if not ambiguous:
            order = sorted(range(n), key=colors.__getitem__)
            code = (
                tuple(labels[node] for node in order),
                tuple(sorted((colors[left], kind, colors[right]) for left, kind, right in edges)),
            )
            search_memo[colors] = code
            return code

        _size, _color, nodes = min(ambiguous)
        individualized_color = max(colors) + 1
        best = None
        for node in nodes:
            branch = list(colors)
            branch[node] = individualized_color
            candidate = canonical(tuple(branch))
            if best is None or candidate < best:
                best = candidate
        search_memo[colors] = best
        return best

    return canonical(initial)


def canonical_diagram_key(prepared):
    labels, edges = _relation_graph(prepared)
    return _canonical_colored_digraph(labels, edges)


def full_recursive_laurent(prepared, evaluator, *, canonical=False, stats=None):
    """Exact global Yamada crossing recursion with shared partial-diagram memo."""
    memo = {}
    if stats is None:
        stats = {}
    stats.update(
        calls=0,
        memo_hits=0,
        rii_moves=0,
        inversions=0,
        resolutions=0,
        key_seconds=0.0,
    )

    def rec(current):
        stats["calls"] += 1
        current, moves = current.reduce_reidemeister_ii()
        stats["rii_moves"] += moves
        key_started = time.perf_counter()
        key = canonical_diagram_key(current) if canonical else diagram_key(current)
        stats["key_seconds"] += time.perf_counter() - key_started
        if key in memo:
            stats["memo_hits"] += 1
            return memo[key]

        crossing_count = len(current.crossing_ids)
        if crossing_count == 0:
            value = evaluator.compute_prepared_bulk_laurent(current)
        else:
            inversion = first_reducing_inversion(current)
            if inversion is not None:
                crossing_index, inverted_reduced = inversion
                stats["inversions"] += 1
                positive = rec(resolve_crossing(current, crossing_index, 0))
                negative = rec(resolve_crossing(current, crossing_index, 1))
                value = add(rec(inverted_reduced), _skein_delta(positive, negative))
            else:
                resolved = first_resolvable_crossing(current)
                if resolved is None:
                    value = evaluator.compute_prepared_bulk_laurent(current)
                else:
                    _crossing_index, (plus, minus, vertex) = resolved
                    stats["resolutions"] += 1
                    value = add(
                        add(shift(rec(plus), 1), shift(rec(minus), -1)),
                        rec(vertex),
                    )

        memo[key] = value
        return value

    value = rec(prepared)
    stats["memo_size"] = len(memo)
    return value


def prepared_theta(n):
    _graph, processor, _pdcode = torus.prepare_essential_torus(n)
    yamada = Yamada.from_PDCode(processor)
    return PreparedCompactStateBuilder.prepare(
        yamada.vertices,
        yamada.crossings,
        yamada.arcs,
        _ordered_crossing_ports,
    )


def run_candidate(label, n, canonical):
    prepared = prepared_theta(n)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = tuple(sorted(torus.independent_theta_terms(n).items()))
    stats = {}
    started = time.perf_counter()
    actual = full_recursive_laurent(
        prepared,
        evaluator,
        canonical=canonical,
        stats=stats,
    )
    elapsed = time.perf_counter() - started
    if actual != expected:
        raise AssertionError(
            f"{label} disagrees with external theorem oracle at n={n}"
        )
    print(
        json.dumps(
            {
                "candidate": label,
                "n": n,
                "seconds": elapsed,
                "stats": stats,
                "correctness": "PASS",
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def main():
    for n in (9, 11):
        run_candidate("labeled_global_recursion", n, False)
    for n in (9, 11, 13, 15, 17):
        run_candidate("exact_canonical_global_recursion", n, True)


if __name__ == "__main__":
    main()
