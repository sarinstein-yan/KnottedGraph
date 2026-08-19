"""Validate and benchmark an exact trial-equivalent inversion selector.

The new selector determines the same first inversion/RII result as the old
try-every-crossing selector directly from prepared port adjacency. The
Dobrynin--Vesnin formula remains an external post-computation oracle only.
"""
from __future__ import annotations
import json, time
import benchmark_topoly_essential_torus_scaling as torus
from knotted_graph.invariants.yamada import _yamada_iso
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add, shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import _skein_delta, invert_crossing, resolve_crossing
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder


def diagram_key(prepared):
    return (prepared.vertex_ids, prepared.crossing_ids, prepared.ordered_ports,
            prepared.arc_partner, prepared.fixed_terminal_index, prepared.crossing_for_port)


def trial_reducing_inversion(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        inverted = invert_crossing(prepared, crossing_index)
        reduced, moves = inverted.reduce_reidemeister_ii()
        if moves:
            return crossing_index, reduced
    return None


def _near_rii_pairs(prepared):
    """Yield all crossing pairs exactly one inversion away from conservative RII."""
    arc_partner = prepared.arc_partner
    crossing_for_port = prepared.crossing_for_port
    ordered_ports = prepared.ordered_ports
    for first, first_ports in enumerate(ordered_ports):
        by_second = {}
        for first_position, first_port in enumerate(first_ports):
            partner = arc_partner[first_port]
            second = crossing_for_port[partner]
            if second < 0 or second >= first:
                continue
            by_second.setdefault(second, []).append((first_position, partner))
        for second in sorted(by_second):
            links = by_second[second]
            if len(links) != 2:
                continue
            second_ports = ordered_ports[second]
            second_position = {port: i for i, port in enumerate(second_ports)}
            shared = []
            for first_position, partner in links:
                pos = second_position.get(partner)
                if pos is None:
                    break
                shared.append((first_position, pos))
            if len(shared) != 2:
                continue
            if (shared[0][0] - shared[1][0]) % 4 not in (1, 3):
                continue
            if (shared[0][1] - shared[1][1]) % 4 not in (1, 3):
                continue
            # Existing RII would have equal endpoint parity. Since every rec()
            # entry is already RII-reduced, the only useful candidates have both
            # shared arcs parity-reversing; flipping either endpoint crossing
            # toggles both to the exact production RII condition.
            if any((a % 2) == (b % 2) for a, b in shared):
                continue
            removed = set(first_ports) | set(second_ports)
            splices = []
            valid = True
            for first_position, second_position_index in shared:
                first_external = first_ports[(first_position + 2) % 4]
                second_external = second_ports[(second_position_index + 2) % 4]
                left = arc_partner[first_external]
                right = arc_partner[second_external]
                if left in removed or right in removed or left == right:
                    valid = False
                    break
                splices.append((left, right))
            if valid and len({p for pair in splices for p in pair}) == 4:
                yield first, second, tuple(splices)


def trial_equivalent_direct_inversion(prepared):
    """Return exactly the old first-successful trial inversion without trials."""
    candidates = list(_near_rii_pairs(prepared))
    if not candidates:
        return None
    # Old code tries crossing indices 0,1,... . Thus the first success is the
    # smallest endpoint of any near-RII pair. After that inversion, production
    # _find_reidemeister_ii_pair orders pairs lexicographically by (first,second),
    # where first > second. Reproduce both choices exactly.
    crossing_index = min(min(first, second) for first, second, _ in candidates)
    involving = [item for item in candidates if crossing_index in item[:2]]
    first, second, splices = min(involving, key=lambda item: (item[0], item[1]))
    inverted = invert_crossing(prepared, crossing_index)
    reduced_once = inverted._remove_reidemeister_ii_pair(first, second, splices)
    # The old reducer continues cancelling any RII pairs exposed by that first
    # cancellation. Preserve that behavior exactly.
    reduced, _moves = reduced_once.reduce_reidemeister_ii()
    return crossing_index, reduced


def first_resolvable_crossing(prepared):
    for crossing_index in range(len(prepared.crossing_ids)):
        try:
            return crossing_index, tuple(resolve_crossing(prepared, crossing_index, spin) for spin in (0,1,2))
        except ValueError:
            continue
    return None


def native_index(prepared):
    return _yamada_iso.PreparedDiagramIndex(len(prepared.vertex_ids), [list(x) for x in prepared.ordered_ports],
        list(prepared.arc_partner), list(prepared.fixed_terminal_index), list(prepared.crossing_for_port))


class ExactNativeIsoMemo:
    def __init__(self): self.buckets={}; self.size=self.hits=self.comparisons=0
    def get(self,p):
        idx=native_index(p); key=(len(p.crossing_ids),idx.node_count,idx.fingerprint)
        for other,value in self.buckets.get(key,()):
            self.comparisons+=1
            if idx.isomorphic(other): self.hits+=1; return True,value,key,idx
        return False,None,key,idx
    def put(self,key,idx,value): self.buckets.setdefault(key,[]).append((idx,value)); self.size+=1


def evaluate(prepared,evaluator,selector,*,cutoff=4,verify_selector=False,stats=None):
    memo=ExactNativeIsoMemo(); stats={} if stats is None else stats
    stats.update(calls=0,hits=0,bulk=0,inversions=0,resolutions=0,selector_checks=0)
    def rec(current):
        stats["calls"]+=1
        current,_=current.reduce_reidemeister_ii()
        hit,cached,key,idx=memo.get(current)
        if hit: stats["hits"]+=1; return cached
        c=len(current.crossing_ids)
        if c<=cutoff:
            stats["bulk"]+=1; value=evaluator.compute_prepared_bulk_laurent(current)
        else:
            inversion=selector(current)
            if verify_selector:
                reference=trial_reducing_inversion(current)
                stats["selector_checks"]+=1
                if (inversion is None)!=(reference is None):
                    raise AssertionError("selector existence mismatch")
                if inversion is not None:
                    if inversion[0]!=reference[0] or diagram_key(inversion[1])!=diagram_key(reference[1]):
                        raise AssertionError(("selector state mismatch",c,inversion[0],reference[0]))
            if inversion is not None:
                crossing,reduced=inversion; stats["inversions"]+=1
                plus=rec(resolve_crossing(current,crossing,0)); minus=rec(resolve_crossing(current,crossing,1))
                value=add(rec(reduced),_skein_delta(plus,minus))
            else:
                resolved=first_resolvable_crossing(current)
                if resolved is None:
                    stats["bulk"]+=1; value=evaluator.compute_prepared_bulk_laurent(current)
                else:
                    _,(plus,minus,vertex)=resolved; stats["resolutions"]+=1
                    value=add(add(shift(rec(plus),1),shift(rec(minus),-1)),rec(vertex))
        memo.put(key,idx,value); return value
    value=rec(prepared); stats.update(memo_size=memo.size,iso_hits=memo.hits,iso_comparisons=memo.comparisons)
    return value


def prepared_theta(n):
    _g,p,_pd=torus.prepare_essential_torus(n); y=Yamada.from_PDCode(p)
    return PreparedCompactStateBuilder.prepare(y.vertices,y.crossings,y.arcs,_ordered_crossing_ports)


def run(n,label,selector,verify=False):
    prepared=prepared_theta(n); evaluator=NativeCompactEvaluator(PythonCompactYamadaEvaluator)
    expected=tuple(sorted(torus.independent_theta_terms(n).items())); stats={}; start=time.perf_counter()
    actual=evaluate(prepared,evaluator,selector,cutoff=4,verify_selector=verify,stats=stats); elapsed=time.perf_counter()-start
    if actual!=expected: raise AssertionError((n,label,actual,expected))
    print(json.dumps({"n":n,"candidate":label,"seconds":elapsed,"stats":stats,"correctness":"PASS"},separators=(",",":")),flush=True)


def main():
    # Verification mode compares selector outputs state-for-state with the old
    # trial selector on every unique recursion state at representative sizes.
    for n in (9,13,17): run(n,"direct_trial_equivalent_verified",trial_equivalent_direct_inversion,True)
    # Performance-only high-end cases after equivalence has been established.
    for n in (15,17,19):
        run(n,"trial_reference",trial_reducing_inversion,False)
        run(n,"direct_trial_equivalent",trial_equivalent_direct_inversion,False)

if __name__=="__main__": main()
