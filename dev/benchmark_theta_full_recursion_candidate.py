"""Validate exact adjacency-indexed RII search in generic Yamada recursion."""
from __future__ import annotations
import json,time
import benchmark_topoly_essential_torus_scaling as torus
from knotted_graph.invariants.yamada import _yamada_iso
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add,shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada,_ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import _skein_delta,invert_crossing,resolve_crossing
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder

ORIGINAL_RII=PreparedCompactStateBuilder._find_reidemeister_ii_pair


def adjacency_rii(self):
    """Same conservative RII candidate/order as production, using arc adjacency."""
    arc_partner=self.arc_partner
    crossing_for_port=self.crossing_for_port
    ordered_ports=self.ordered_ports
    for first,first_ports in enumerate(ordered_ports):
        by_second={}
        for first_position,first_port in enumerate(first_ports):
            partner=arc_partner[first_port]
            second=crossing_for_port[partner]
            if second<0 or second>=first: continue
            by_second.setdefault(second,[]).append((first_position,partner))
        # Preserve the old nested-loop order exactly: first ascending, then second ascending.
        for second in sorted(by_second):
            links=by_second[second]
            if len(links)!=2: continue
            second_ports=ordered_ports[second]
            second_position={port:i for i,port in enumerate(second_ports)}
            shared=[]
            for first_position,partner in links:
                pos=second_position.get(partner)
                if pos is None: break
                shared.append((first_position,pos,first_ports[first_position],partner))
            if len(shared)!=2: continue
            if (shared[0][0]-shared[1][0])%4 not in (1,3): continue
            if (shared[0][1]-shared[1][1])%4 not in (1,3): continue
            if any((a%2)!=(b%2) for a,b,_,_ in shared): continue
            removed=set(first_ports)|set(second_ports); splices=[]; valid=True
            for first_position,second_position_index,_,_ in shared:
                first_external=first_ports[(first_position+2)%4]
                second_external=second_ports[(second_position_index+2)%4]
                left=arc_partner[first_external]; right=arc_partner[second_external]
                if left in removed or right in removed or left==right:
                    valid=False; break
                splices.append((left,right))
            if valid and len({p for pair in splices for p in pair})==4:
                return first,second,tuple(splices)
    return None


def verified_adjacency_rii(self):
    expected=ORIGINAL_RII(self); actual=adjacency_rii(self)
    if actual!=expected: raise AssertionError(("RII finder mismatch",actual,expected))
    return actual


def near_rii_pairs(prepared):
    arc_partner=prepared.arc_partner; crossing_for_port=prepared.crossing_for_port; ordered_ports=prepared.ordered_ports
    for first,first_ports in enumerate(ordered_ports):
        by_second={}
        for first_position,first_port in enumerate(first_ports):
            partner=arc_partner[first_port]; second=crossing_for_port[partner]
            if second<0 or second>=first: continue
            by_second.setdefault(second,[]).append((first_position,partner))
        for second in sorted(by_second):
            links=by_second[second]
            if len(links)!=2: continue
            second_ports=ordered_ports[second]; positions={p:i for i,p in enumerate(second_ports)}
            shared=[]
            for fp,partner in links:
                pos=positions.get(partner)
                if pos is None: break
                shared.append((fp,pos))
            if len(shared)!=2 or (shared[0][0]-shared[1][0])%4 not in (1,3) or (shared[0][1]-shared[1][1])%4 not in (1,3): continue
            if any((a%2)==(b%2) for a,b in shared): continue
            removed=set(first_ports)|set(second_ports); splices=[]; valid=True
            for fp,sp in shared:
                left=arc_partner[first_ports[(fp+2)%4]]; right=arc_partner[second_ports[(sp+2)%4]]
                if left in removed or right in removed or left==right: valid=False; break
                splices.append((left,right))
            if valid and len({p for pair in splices for p in pair})==4: yield first,second,tuple(splices)


def direct_trial_equivalent(prepared):
    candidates=list(near_rii_pairs(prepared))
    if not candidates:return None
    crossing=min(min(a,b) for a,b,_ in candidates)
    first,second,splices=min((x for x in candidates if crossing in x[:2]),key=lambda x:(x[0],x[1]))
    inv=invert_crossing(prepared,crossing)
    once=inv._remove_reidemeister_ii_pair(first,second,splices)
    reduced,_=once.reduce_reidemeister_ii()
    return crossing,reduced


def first_resolution(p):
    for i in range(len(p.crossing_ids)):
        try:return i,tuple(resolve_crossing(p,i,s) for s in (0,1,2))
        except ValueError:pass
    return None


def native_index(p):return _yamada_iso.PreparedDiagramIndex(len(p.vertex_ids),[list(x) for x in p.ordered_ports],list(p.arc_partner),list(p.fixed_terminal_index),list(p.crossing_for_port))
class Memo:
    def __init__(self):self.b={};self.size=self.hits=0
    def get(self,p):
        i=native_index(p);k=(len(p.crossing_ids),i.node_count,i.fingerprint)
        for o,v in self.b.get(k,()):
            if i.isomorphic(o):self.hits+=1;return True,v,k,i
        return False,None,k,i
    def put(self,k,i,v):self.b.setdefault(k,[]).append((i,v));self.size+=1


def evaluate(prepared,evaluator,stats):
    memo=Memo();stats.update(calls=0,hits=0,bulk=0,inversions=0,resolutions=0)
    def rec(p):
        stats["calls"]+=1;p,_=p.reduce_reidemeister_ii();hit,val,k,idx=memo.get(p)
        if hit:stats["hits"]+=1;return val
        if len(p.crossing_ids)<=4:stats["bulk"]+=1;val=evaluator.compute_prepared_bulk_laurent(p)
        else:
            inv=direct_trial_equivalent(p)
            if inv is not None:
                c,r=inv;stats["inversions"]+=1
                plus=rec(resolve_crossing(p,c,0));minus=rec(resolve_crossing(p,c,1));val=add(rec(r),_skein_delta(plus,minus))
            else:
                rr=first_resolution(p)
                if rr is None:stats["bulk"]+=1;val=evaluator.compute_prepared_bulk_laurent(p)
                else:
                    _,(plus,minus,vertex)=rr;stats["resolutions"]+=1;val=add(add(shift(rec(plus),1),shift(rec(minus),-1)),rec(vertex))
        memo.put(k,idx,val);return val
    out=rec(prepared);stats.update(memo_size=memo.size,iso_hits=memo.hits);return out


def prepared_theta(n):
    _,p,_=torus.prepare_essential_torus(n);y=Yamada.from_PDCode(p);return PreparedCompactStateBuilder.prepare(y.vertices,y.crossings,y.arcs,_ordered_crossing_ports)

def run(n,label,finder):
    PreparedCompactStateBuilder._find_reidemeister_ii_pair=finder
    p=prepared_theta(n);e=NativeCompactEvaluator(PythonCompactYamadaEvaluator);expected=tuple(sorted(torus.independent_theta_terms(n).items()));stats={};t=time.perf_counter();got=evaluate(p,e,stats);secs=time.perf_counter()-t
    if got!=expected:raise AssertionError((n,label,got,expected))
    print(json.dumps({"n":n,"candidate":label,"seconds":secs,"stats":stats,"correctness":"PASS"},separators=(",",":")),flush=True)

def main():
    # Exhaustively compare finder outputs on every state reached by this recursion.
    for n in (9,13,17):run(n,"adjacency_RII_verified",verified_adjacency_rii)
    for n in (15,17,19):
        run(n,"quadratic_RII_reference",ORIGINAL_RII)
        run(n,"adjacency_RII",adjacency_rii)
    PreparedCompactStateBuilder._find_reidemeister_ii_pair=ORIGINAL_RII
if __name__=="__main__":main()
