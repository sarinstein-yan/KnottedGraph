"""Discover/verify exact local curl factors in emergency Yamada leaves.

The expensive legacy prepared evaluator supplies the value of each complete
leaf. Candidate crossing removal is purely local. We accept a rule only when
leaf and reduced values differ by one exact signed Laurent monomial across all
sampled surrounding graph contexts.
"""
from __future__ import annotations

import json

import benchmark_theta_full_recursion_candidate as base
import knotted_graph.invariants.yamada.skein_hybrid as sh
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.state_compact import _MINUS_PAIRS, _PLUS_PAIRS


def monomial_ratio(full, reduced):
    if not full or not reduced or len(full) != len(reduced):
        return None
    shift = full[0][0] - reduced[0][0]
    first = reduced[0][1]
    if first == 0 or full[0][1] % first:
        return None
    scale = full[0][1] // first
    if scale not in (-1, 1):
        return None
    expected = tuple((power + shift, scale * coeff) for power, coeff in reduced)
    return (scale, shift) if expected == full else None


def collect_emergency_leaves(n):
    old = sh._smooth_crossing
    sh._smooth_crossing = base.ORIGINAL_SMOOTH
    memo = base.Memo()
    leaves = []

    def rec(p):
        p, _ = p.reduce_reidemeister_ii()
        hit, _value, key, idx = memo.get(p)
        if hit:
            return
        memo.put(key, idx, ())
        if len(p.crossing_ids) <= 4:
            return
        inv = base.trial_inversion(p)
        if inv is not None:
            crossing, reduced = inv
            rec(sh.resolve_crossing(p, crossing, 0))
            rec(sh.resolve_crossing(p, crossing, 1))
            rec(reduced)
            return
        rr = base.first_resolution(p)
        if rr is None:
            leaves.append(p)
            return
        _crossing, children = rr
        for child in children:
            rec(child)

    rec(base.prepared(n))
    sh._smooth_crossing = old
    return leaves


def self_pair_pattern(p, crossing):
    ports = p.ordered_ports[crossing]
    position = {port: i for i, port in enumerate(ports)}
    pairs = []
    for i, port in enumerate(ports):
        partner = p.arc_partner[port]
        j = position.get(partner)
        if j is not None and i < j:
            pairs.append((i, j))
    return tuple(pairs)


def main():
    leaves = collect_emergency_leaves(13)
    print("EMERGENCY_LEAVES", len(leaves), flush=True)
    rules = {}
    checked = 0
    for leaf_index, leaf in enumerate(leaves):
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        full = evaluator.compute_prepared_bulk_laurent(leaf)
        for crossing in range(len(leaf.crossing_ids)):
            pattern = self_pair_pattern(leaf, crossing)
            if not pattern:
                continue
            for channel, pairs in (("plus", _PLUS_PAIRS), ("minus", _MINUS_PAIRS)):
                reduced = base.general_smooth(leaf, crossing, pairs)
                # A curl-removal channel should not manufacture a detached circle.
                if len(reduced.vertex_ids) != len(leaf.vertex_ids):
                    continue
                reduced_value = evaluator.compute_prepared_bulk_laurent(reduced)
                ratio = monomial_ratio(full, reduced_value)
                if ratio is None:
                    continue
                key = (pattern, channel)
                rules.setdefault(key, set()).add(ratio)
                checked += 1
                print(json.dumps({
                    "leaf": leaf_index,
                    "crossings": len(leaf.crossing_ids),
                    "pattern": pattern,
                    "channel": channel,
                    "ratio": ratio,
                }, separators=(",", ":")), flush=True)
    print("RULE_SUMMARY=" + json.dumps({
        repr(key): sorted(values) for key, values in rules.items()
    }, separators=(",", ":")), flush=True)
    print("MATCHES", checked, flush=True)
    if not rules:
        raise AssertionError("no exact local monomial curl rule found")
    if any(len(values) != 1 for values in rules.values()):
        raise AssertionError("curl factor depends on surrounding context")


if __name__ == "__main__":
    main()
