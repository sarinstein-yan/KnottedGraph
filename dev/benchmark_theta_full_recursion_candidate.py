"""Benchmark exact generic crossing recursion with fast isomorphism memoization.

The Dobrynin--Vesnin formula is used only after evaluation as a correctness
oracle. It is never used by the candidate evaluator.
"""

from __future__ import annotations

import json
import time

import networkx as nx

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


def _vf2_graph(prepared):
    """Node-labeled undirected gadget graph exactly encoding a prepared diagram.

    Physical arcs and terminal incidence are ordinary gadgets. Directed cyclic
    crossing order is encoded by source/edge/target nodes with distinct labels,
    so undirected VF2++ still preserves crossing orientation. Rotating a crossing
    by two ports remains an isomorphism, while reflection does not.
    """
    graph = nx.Graph()
    port_count = len(prepared.arc_partner)
    parity = [-1] * port_count
    for ports in prepared.ordered_ports:
        for position, port in enumerate(ports):
            parity[port] = position % 2

    for port in range(port_count):
        if prepared.crossing_for_port[port] >= 0:
            label = "port_over" if parity[port] == 0 else "port_under"
        else:
            label = "port_terminal"
        graph.add_node(("p", port), label=label)

    # Physical arc pairing.
    for port, partner in enumerate(prepared.arc_partner):
        if port < partner:
            arc = ("a", port, partner)
            graph.add_node(arc, label="physical_arc")
            graph.add_edge(("p", port), arc)
            graph.add_edge(arc, ("p", partner))

    # Directed crossing cyclic order as an asymmetric labeled path.
    relation_index = 0
    for ports in prepared.ordered_ports:
        for left, right in zip(ports, ports[1:] + ports[:1]):
            src = ("cs", relation_index)
            mid = ("cm", relation_index)
            dst = ("ct", relation_index)
            relation_index += 1
            graph.add_node(src, label="cycle_source")
            graph.add_node(mid, label="cycle_relation")
            graph.add_node(dst, label="cycle_target")
            graph.add_edge(("p", left), src)
            graph.add_edge(src, mid)
            graph.add_edge(mid, dst)
            graph.add_edge(dst, ("p", right))

    terminal_ports = {}
    for port, terminal in enumerate(prepared.fixed_terminal_index):
        if terminal >= 0:
            terminal_ports.setdefault(terminal, []).append(port)
    for terminal, ports in terminal_ports.items():
        vertex = ("v", terminal)
        graph.add_node(vertex, label="spatial_vertex")
        for port in ports:
            graph.add_edge(vertex, ("p", port))
    return graph


class ExactVF2Memo:
    """WL-bucketed memo whose hits are certified by exact VF2++ isomorphism."""

    def __init__(self):
        self.buckets = {}
        self.size = 0
        self.hits = 0
        self.comparisons = 0
        self.graph_seconds = 0.0
        self.iso_seconds = 0.0

    def get(self, prepared):
        started = time.perf_counter()
        graph = _vf2_graph(prepared)
        fingerprint = nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="label",
            iterations=4,
        )
        self.graph_seconds += time.perf_counter() - started
        bucket_key = (len(prepared.crossing_ids), len(graph), fingerprint)
        for other, value in self.buckets.get(bucket_key, ()):
            self.comparisons += 1
            started = time.perf_counter()
            equivalent = nx.vf2pp_is_isomorphic(
                graph,
                other,
                node_label="label",
            )
            self.iso_seconds += time.perf_counter() - started
            if equivalent:
                self.hits += 1
                return True, value, bucket_key, graph
        return False, None, bucket_key, graph

    def put(self, bucket_key, graph, value):
        self.buckets.setdefault(bucket_key, []).append((graph, value))
        self.size += 1


def full_recursive_laurent(prepared, evaluator, stats=None):
    """Exact generic Yamada recursion with global exact isomorphism memo."""
    memo = ExactVF2Memo()
    if stats is None:
        stats = {}
    stats.update(calls=0, memo_hits=0, rii_moves=0, inversions=0, resolutions=0)

    def rec(current):
        stats["calls"] += 1
        current, moves = current.reduce_reidemeister_ii()
        stats["rii_moves"] += moves
        hit, cached, bucket_key, graph = memo.get(current)
        if hit:
            stats["memo_hits"] += 1
            return cached

        if not current.crossing_ids:
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
                    _index, (plus, minus, vertex) = resolved
                    stats["resolutions"] += 1
                    value = add(
                        add(shift(rec(plus), 1), shift(rec(minus), -1)),
                        rec(vertex),
                    )
        memo.put(bucket_key, graph, value)
        return value

    value = rec(prepared)
    stats.update(
        memo_size=memo.size,
        iso_hits=memo.hits,
        iso_comparisons=memo.comparisons,
        graph_seconds=memo.graph_seconds,
        iso_seconds=memo.iso_seconds,
        buckets=len(memo.buckets),
    )
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


def main():
    for n in (9, 11, 13, 15, 17):
        prepared = prepared_theta(n)
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        expected = tuple(sorted(torus.independent_theta_terms(n).items()))
        stats = {}
        started = time.perf_counter()
        actual = full_recursive_laurent(prepared, evaluator, stats=stats)
        elapsed = time.perf_counter() - started
        if actual != expected:
            raise AssertionError(
                f"VF2++ generic recursion disagrees with external theorem oracle at n={n}"
            )
        print(json.dumps({
            "candidate": "exact_vf2pp_global_recursion",
            "n": n,
            "seconds": elapsed,
            "stats": stats,
            "correctness": "PASS",
        }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
