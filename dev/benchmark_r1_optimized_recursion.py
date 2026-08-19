"""Benchmark generic exact Yamada recursion with local R1/curl reductions.

No family formula is used by evaluation. Dobrynin--Vesnin terms are consulted
only after computation as an external correctness oracle.
"""
from __future__ import annotations
import json,time
import benchmark_theta_full_recursion_candidate as base
import knotted_graph.invariants.yamada.skein_hybrid as sh
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add,shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.state_compact import _MINUS_PAIRS,_PLUS_PAIRS


def find_r1(p):
    """Return (reduced, exponent) for the first exact Reidemeister-I curl."""
    for crossing,ports in enumerate(p.ordered_ports):
        position={port:i for i,port in enumerate(ports)}
        self_pairs=[]
        for i,port in enumerate(ports):
            j=position.get(p.arc_partner[port])
            if j is not None and i<j:self_pairs.append((i,j))
        if len(self_pairs)!=1:continue
        pattern=self_pairs[0]
        if pattern in ((0,1),(2,3)):
            reduced=base.general_smooth(p,crossing,_PLUS_PAIRS); exponent=-2
        elif pattern in ((0,3),(1,2)):
            reduced=base.general_smooth(p,crossing,_MINUS_PAIRS); exponent=2
        else:
            continue
        if len(reduced.vertex_ids)!=len(p.vertex_ids):
            raise AssertionError("R1 removal unexpectedly created a detached loop")
        return reduced,exponent
    return None


def reduce_r1_chain(p):
    exponent=0;moves=0
    while True:
        p,_=p.reduce_reidemeister_ii()
        found=find_r1(p)
        if found is None:return p,exponent,moves
        p,delta=found;exponent+=delta;moves+=1


def evaluate(p,e,stats):
    memo=base.Memo();stats.update(calls=0,r1_moves=0,inversions=0,resolutions=0,bulk=0,bulk_seconds=0.0,max_bulk_crossings=0)
    def bulk(q):
        t=time.perf_counter();v=e.compute_prepared_bulk_laurent(q);stats["bulk_seconds"]+=time.perf_counter()-t;stats["bulk"]+=1;stats["max_bulk_crossings"]=max(stats["max_bulk_crossings"],len(q.crossing_ids));return v
    def rec(q):
        stats["calls"]+=1
        q,_=q.reduce_reidemeister_ii()
        r1=find_r1(q)
        if r1 is not None:
            reduced,exponent=r1;stats["r1_moves"]+=1
            return shift(rec(reduced),exponent)
        hit,v,key,idx=memo.get(q)
        if hit:return v
        if len(q.crossing_ids)<=4:v=bulk(q)
        else:
            inv=base.trial_inversion(q)
            if inv is not None:
                i,reduced=inv;stats["inversions"]+=1
                plus=rec(sh.resolve_crossing(q,i,0));minus=rec(sh.resolve_crossing(q,i,1));v=add(rec(reduced),sh._skein_delta(plus,minus))
            else:
                rr=base.first_resolution(q)
                if rr is None:v=bulk(q)
                else:
                    _,(plus,minus,vertex)=rr;stats["resolutions"]+=1;v=add(add(shift(rec(plus),1),shift(rec(minus),-1)),rec(vertex))
        memo.put(key,idx,v);return v
    out=rec(p);stats.update(iso_hits=memo.hits,memo_size=sum(map(len,memo.b.values())),native_graph_memo_size=e.memo_size);return out


def run(n):
    p=base.prepared(n);e=NativeCompactEvaluator(PythonCompactYamadaEvaluator);stats={};expected=tuple(sorted(base.torus.independent_theta_terms(n).items()));t=time.perf_counter();got=evaluate(p,e,stats);elapsed=time.perf_counter()-t
    if got!=expected:raise AssertionError((n,got,expected))
    print(json.dumps({"n":n,"candidate":"R1_exact_iso_generic","seconds":elapsed,"stats":stats,"correctness":"PASS"},separators=(",",":")),flush=True)


def main():
    for n in (9,11,13,15,17,19):run(n)

if __name__=="__main__":main()
