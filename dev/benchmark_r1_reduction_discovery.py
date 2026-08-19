"""Discover/verify exact local curl factors in emergency Yamada leaves.

The legacy prepared evaluator supplies each complete value. Both crossing
orientations are tested by also inverting every sampled curl. A rule is accepted
only if complete and crossing-removed values differ by one exact signed Laurent
monomial across every surrounding graph context.
"""
from __future__ import annotations
import json
import benchmark_theta_full_recursion_candidate as base
import knotted_graph.invariants.yamada.skein_hybrid as sh
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.state_compact import _MINUS_PAIRS, _PLUS_PAIRS


def monomial_ratio(full, reduced):
    if not full or not reduced or len(full) != len(reduced): return None
    exponent = full[0][0] - reduced[0][0]
    if reduced[0][1] == 0 or full[0][1] % reduced[0][1]: return None
    sign = full[0][1] // reduced[0][1]
    if sign not in (-1, 1): return None
    expected = tuple((power + exponent, sign * coefficient) for power, coefficient in reduced)
    return (sign, exponent) if expected == full else None


def collect_emergency_leaves(n):
    old = sh._smooth_crossing; sh._smooth_crossing = base.ORIGINAL_SMOOTH
    memo = base.Memo(); leaves = []
    def rec(p):
        p, _ = p.reduce_reidemeister_ii()
        hit, _value, key, idx = memo.get(p)
        if hit: return
        memo.put(key, idx, ())
        if len(p.crossing_ids) <= 4: return
        inv = base.trial_inversion(p)
        if inv is not None:
            crossing, reduced = inv
            rec(sh.resolve_crossing(p, crossing, 0)); rec(sh.resolve_crossing(p, crossing, 1)); rec(reduced)
            return
        rr = base.first_resolution(p)
        if rr is None: leaves.append(p); return
        for child in rr[1]: rec(child)
    rec(base.prepared(n)); sh._smooth_crossing = old
    return leaves


def self_pair_pattern(p, crossing):
    ports = p.ordered_ports[crossing]; position = {port: i for i, port in enumerate(ports)}; pairs = []
    for i, port in enumerate(ports):
        j = position.get(p.arc_partner[port])
        if j is not None and i < j: pairs.append((i, j))
    return tuple(pairs)


def inspect_diagram(diagram, crossing, evaluator, rules, orientation):
    pattern = self_pair_pattern(diagram, crossing)
    if len(pattern) != 1: return 0
    full = evaluator.compute_prepared_bulk_laurent(diagram)
    found = 0
    for channel, pairs in (("plus", _PLUS_PAIRS), ("minus", _MINUS_PAIRS)):
        reduced = base.general_smooth(diagram, crossing, pairs)
        if len(reduced.vertex_ids) != len(diagram.vertex_ids): continue
        ratio = monomial_ratio(full, evaluator.compute_prepared_bulk_laurent(reduced))
        if ratio is None: continue
        rules.setdefault((pattern, channel), set()).add(ratio); found += 1
        print(json.dumps({"orientation":orientation,"pattern":pattern,"channel":channel,"ratio":ratio},separators=(",",":")),flush=True)
    return found


def main():
    leaves = collect_emergency_leaves(13); print("EMERGENCY_LEAVES", len(leaves), flush=True)
    rules = {}; checked = 0
    for leaf in leaves:
        evaluator = NativeCompactEvaluator(PythonCompactYamadaEvaluator)
        for crossing in range(len(leaf.crossing_ids)):
            if len(self_pair_pattern(leaf, crossing)) != 1: continue
            checked += inspect_diagram(leaf, crossing, evaluator, rules, "original")
            mirrored = sh.invert_crossing(leaf, crossing)
            checked += inspect_diagram(mirrored, crossing, evaluator, rules, "inverted")
    summary = {repr(key): sorted(values) for key, values in rules.items()}
    print("RULE_SUMMARY=" + json.dumps(summary,separators=(",",":")),flush=True)
    print("MATCHES",checked,flush=True)
    required = {
        (((0,1),),"plus"):{(1,-2)},
        (((2,3),),"plus"):{(1,-2)},
        (((0,3),),"minus"):{(1,2)},
        (((1,2),),"minus"):{(1,2)},
    }
    for key, expected in required.items():
        if rules.get(key) != expected: raise AssertionError((key,rules.get(key),expected))
    if any(len(values) != 1 for values in rules.values()): raise AssertionError("curl factor depends on context")

if __name__ == "__main__": main()
