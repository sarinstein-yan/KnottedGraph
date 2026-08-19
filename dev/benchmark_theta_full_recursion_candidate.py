"""Profile the residual native abstract-graph cost after isomorphism collapse."""
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


def trial_inversion(p):
    for i in range(len(p.crossing_ids)):
        inv=invert_crossing(p,i);reduced,moves=inv.reduce_reidemeister_ii()
        if moves:return i,reduced
    return None

def first_resolution(p):
    for i in range(len(p.crossing_ids)):
        try:return i,tuple(resolve_crossing(p,i,s) for s in (0,1,2))
        except ValueError:pass
    return None

def index(p):return _yamada_iso.PreparedDiagramIndex(len(p.vertex_ids),[list(x) for x in p.ordered_ports],list(p.arc_partner),list(p.fixed_terminal_index),list(p.crossing_for_port))
class Memo:
    def __init__(self):self.b={};self.hits=0
    def get(self,p):
        x=index(p);k=(len(p.crossing_ids),x.node_count,x.fingerprint)
        for old,v in self.b.get(k,()):
            if x.isomorphic(old):self.hits+=1;return True,v,k,x
        return False,None,k,x
    def put(self,k,x,v):self.b.setdefault(k,[]).append((x,v))

def evaluate(p,e,stats):
    m=Memo();worst=[];stats.update(calls=0,bulk_calls=0,bulk_seconds=0.0,inversions=0,resolutions=0)
    def bulk(q):
        before=e.memo_size;t=time.perf_counter();v=e.compute_prepared_bulk_laurent(q);dt=time.perf_counter()-t;after=e.memo_size
        stats["bulk_calls"]+=1;stats["bulk_seconds"]+=dt
        worst.append((dt,len(q.crossing_ids),len(q.vertex_ids),len(q.arc_partner),after-before,after))
        return v
    def rec(q):
        stats["calls"]+=1;q,_=q.reduce_reidemeister_ii();hit,v,k,x=m.get(q)
        if hit:return v
        if len(q.crossing_ids)<=4:v=bulk(q)
        else:
            inv=trial_inversion(q)
            if inv is not None:
                i,r=inv;stats["inversions"]+=1
                pl=rec(resolve_crossing(q,i,0));mi=rec(resolve_crossing(q,i,1));v=add(rec(r),_skein_delta(pl,mi))
            else:
                rr=first_resolution(q)
                if rr is None:v=bulk(q)
                else:
                    _,(pl,mi,ve)=rr;stats["resolutions"]+=1;v=add(add(shift(rec(pl),1),shift(rec(mi),-1)),rec(ve))
        m.put(k,x,v);return v
    result=rec(p);stats["iso_hits"]=m.hits;stats["memo_size"]=sum(map(len,m.b.values()));stats["native_graph_memo_size"]=e.memo_size
    stats["worst_bulk"]=[{"seconds":a,"crossings":b,"vertices":c,"ports":d,"memo_growth":g,"memo_after":h} for a,b,c,d,g,h in sorted(worst,reverse=True)[:12]]
    return result

def prepared(n):
    _,pd,_=torus.prepare_essential_torus(n);y=Yamada.from_PDCode(pd);return PreparedCompactStateBuilder.prepare(y.vertices,y.crossings,y.arcs,_ordered_crossing_ports)
def main():
    for n in (17,19):
        p=prepared(n);e=NativeCompactEvaluator(PythonCompactYamadaEvaluator);stats={};expected=tuple(sorted(torus.independent_theta_terms(n).items()));t=time.perf_counter();got=evaluate(p,e,stats);secs=time.perf_counter()-t
        assert got==expected
        print(json.dumps({"n":n,"seconds":secs,"stats":stats,"correctness":"PASS"},separators=(",",":")),flush=True)
if __name__=="__main__":main()
