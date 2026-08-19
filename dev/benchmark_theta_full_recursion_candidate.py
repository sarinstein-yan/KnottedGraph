"""Benchmark generic exact crossing-recursion and isomorphism memo candidates.

The Dobrynin--Vesnin formula is used only after evaluation as a correctness
oracle. It is never called by either candidate evaluator.
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


def _iso_graph(prepared):
    """Exact combinatorial diagram graph, invariant under internal relabeling.

    Directed local crossing cycles preserve cyclic order. Port labels distinguish
    over/under parity but not arbitrary crossing/terminal identifiers. Physical
    arcs and terminal incidence are encoded as typed directed edges.
    """
    graph = nx.DiGraph()
    for port in range(len(prepared.arc_partner)):
        crossing = prepared.crossing_for_port[port]
        if crossing >= 0:
            positions = prepared.ordered_ports[crossing]
            position = positions.index(port)
            label = f"port:{position % 2}"
        else:
            label = "port:terminal"
        graph.add_node(("p", port), label=label)

    # Physical arc pairing is undirected, represented by two directed edges.
    for port, partner in enumerate(prepared.arc_partner):
        if port < partner:
            graph.add_edge(("p", port), ("p", partner), kind="arc")
            graph.add_edge(("p", partner), ("p", port), kind="arc")

    # A directed four-cycle preserves crossing cyclic orientation. Rotation by
    # two is allowed automatically and corresponds to the same crossing data.
    for ports in prepared.ordered_ports:
        for left, right in zip(ports, ports[1:] + ports[:1]):
            graph.add_edge(("p", left), ("p", right), kind="cycle")

    # Spatial-graph vertices are unlabeled combinatorial terminals.
    terminal_ports = {}
    for port, terminal in enumerate(prepared.fixed_terminal_index):
        if terminal >= 0:
            terminal_ports.setdefault(terminal, []).append(port)
    for terminal, ports in terminal_ports.items():
        vnode = ("v", terminal)
        graph.add_node(vnode, label="vertex")
        for port in ports:
            graph.add_edge(vnode, ("p", port), kind="terminal")
            graph.add_edge(("p", port), vnode, kind="terminal")
    return graph


_NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("label", "")
_EDGE_MATCH = nx.algorithms.isomorphism.categorical_edge_match("kind", "")


class ExactIsoMemo:
    """Hash-bucketed exact diagram-isomorphism memo.

    Weisfeiler--Lehman hashes only select a bucket. Every apparent hit is
    confirmed by exact directed graph isomorphism, so hash collisions cannot
    change the result.
    """

    def __init__(self):
        self.buckets = {}
        self.size = 0
        self.comparisons = 0
        self.hits = 0

    def _token(self, prepared):
        graph = _iso_graph(prepared)
        fingerprint = nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="label",
            edge_attr="kind",
            iterations=5,
        )
        return (len(prepared.crossing_ids), fingerprint), graph

    def get(self, prepared):
        bucket_key, graph = self._token(prepared)
        for other, value in self.buckets.get(bucket_key, ()):
            self.comparisons += 1
            if nx.is_isomorphic(
                graph,
                other,
                node_match=_NODE_MATCH,
                edge_match=_EDGE_MATCH,
            ):
                self.hits += 1
                return True, value, bucket_key, graph
        return False, None, bucket_key, graph

    def put_token(self, bucket_key, graph, value):
        self.buckets.setdefault(bucket_key, []).append((graph, value))
        self.size += 1


def full_recursive_laurent(prepared, evaluator, *, use_iso_memo=False, stats=None):
    """Exact global Yamada crossing recursion with shared partial-diagram memo."""
    labeled_memo = {}
    iso_memo = ExactIsoMemo() if use_iso_memo else None
    if stats is None:
        stats = {}
    stats.update(calls=0, memo_hits=0, rii_moves=0, inversions=0, resolutions=0)

    def rec(current):
        stats["calls"] += 1
        current, moves = current.reduce_reidemeister_ii()
        stats["rii_moves"] += moves

        if iso_memo is None:
            key = diagram_key(current)
            cached = labeled_memo.get(key)
            if cached is not None:
                stats["memo_hits"] += 1
                return cached
            token = key
            token_graph = None
        else:
            hit, cached, token, token_graph = iso_memo.get(current)
            if hit:
                stats["memo_hits"] += 1
                return cached

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

        if iso_memo is None:
            labeled_memo[token] = value
        else:
            iso_memo.put_token(token, token_graph, value)
        return value

    value = rec(prepared)
    stats["memo_size"] = len(labeled_memo) if iso_memo is None else iso_memo.size
    if iso_memo is not None:
        stats["iso_comparisons"] = iso_memo.comparisons
        stats["iso_hits"] = iso_memo.hits
        stats["iso_buckets"] = len(iso_memo.buckets)
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


def run_candidate(label, n, use_iso_memo):
    prepared = prepared_theta(n)
    evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected = tuple(sorted(torus.independent_theta_terms(n).items()))
    stats = {}
    started = time.perf_counter()
    actual = full_recursive_laurent(
        prepared,
        evaluator,
        use_iso_memo=use_iso_memo,
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
    # Small baseline establishes the cost of exact graph-isomorphism memoization.
    for n in (9, 11):
        run_candidate("labeled_global_recursion", n, False)
    # If symmetry collapsing works, the high-n cases should become much smaller.
    for n in (9, 11, 13, 15, 17):
        run_candidate("exact_isomorphism_global_recursion", n, True)


if __name__ == "__main__":
    main()
