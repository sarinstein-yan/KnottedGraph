"""Benchmark generalized exact smoothing for self-adjacent crossings.

The generalized smoother represents any detached circle created while removing
one crossing exactly as the same dummy-vertex/self-loop component used by
PreparedCompactStateBuilder.build(). No theorem formula is used by evaluation.
"""
from __future__ import annotations
import json,time
import benchmark_topoly_essential_torus_scaling as torus
import knotted_graph.invariants.yamada.skein_hybrid as sh
from knotted_graph.invariants.yamada import _yamada_iso
from knotted_graph.invariants.yamada.compact import PythonCompactYamadaEvaluator
from knotted_graph.invariants.yamada.fast import add,shift
from knotted_graph.invariants.yamada.native import NativeCompactEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada,_ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder,_PLUS_PAIRS,_MINUS_PAIRS

ORIGINAL_SMOOTH=sh._smooth_crossing


def resolution_tables(ordered_ports,port_count):
    plus=[-1]*port_count;minus=[-1]*port_count
    for ports in ordered_ports:
        for a,b in _PLUS_PAIRS:
            pa,pb=ports[a],ports[b];plus[pa]=pb;plus[pb]=pa
        for a,b in _MINUS_PAIRS:
            pa,pb=ports[a],ports[b];minus[pa]=pb;minus[pb]=pa
    return tuple(plus),tuple(minus)


def general_smooth(prepared,crossing_index,pairs):
    ports=prepared.ordered_ports[crossing_index]
    removed=set(ports);partner=prepared.arc_partner

    # Build the local degree-two graph consisting of physical arcs incident on
    # the removed crossing plus the requested smoothing pairings.
    adjacency={p:set() for p in ports}
    for p in ports:
        q=partner[p];adjacency.setdefault(p,set()).add(q);adjacency.setdefault(q,set()).add(p)
    for a,b in pairs:
        p,q=ports[a],ports[b];adjacency[p].add(q);adjacency[q].add(p)

    seen=set();splices=[];closed_loops=0
    for start in list(adjacency):
        if start in seen:continue
        stack=[start];component=set()
        while stack:
            node=stack.pop()
            if node in component:continue
            component.add(node);seen.add(node);stack.extend(adjacency.get(node,()))
        external=sorted(node for node in component if node not in removed)
        if len(external)==2:
            splices.append(tuple(external))
        elif len(external)==0:
            closed_loops+=1
        else:
            raise RuntimeError(("malformed smoothing component",external,component))

    active=[p for p in range(len(partner)) if p not in removed]
    old_to_new={old:new for new,old in enumerate(active)}
    updated=list(partner)
    for left,right in splices:
        updated[left]=right;updated[right]=left
    new_partner=[]
    for old in active:
        q=updated[old]
        if q not in old_to_new:raise RuntimeError("smoothing left edge attached to removed port")
        new_partner.append(old_to_new[q])

    surviving=[i for i in range(len(prepared.crossing_ids)) if i!=crossing_index]
    remap={old:new for new,old in enumerate(surviving)}
    new_crossing_for=[]
    for old in active:
        c=prepared.crossing_for_port[old]
        if c<0:new_crossing_for.append(-1)
        elif c==crossing_index:raise RuntimeError("removed crossing port survived")
        else:new_crossing_for.append(remap[c])
    new_ordered=tuple(tuple(old_to_new[p] for p in prepared.ordered_ports[i]) for i in surviving)
    new_fixed=[prepared.fixed_terminal_index[p] for p in active]
    new_vertex_ids=list(prepared.vertex_ids)

    # Match build()'s exact representation of a terminal-free closed component:
    # one fresh graph vertex carrying one self-loop edge (two paired ports).
    next_id=max((*prepared.vertex_ids,*prepared.crossing_ids),default=-1)+1
    for loop_index in range(closed_loops):
        vertex_index=len(new_vertex_ids);new_vertex_ids.append(next_id+loop_index)
        left=len(new_partner);right=left+1
        new_partner.extend((right,left));new_fixed.extend((vertex_index,vertex_index));new_crossing_for.extend((-1,-1))

    plus,minus=resolution_tables(new_ordered,len(new_partner))
    return PreparedCompactStateBuilder(
        vertex_ids=tuple(new_vertex_ids),
        crossing_ids=tuple(prepared.crossing_ids[i] for i in surviving),
        ordered_ports=new_ordered,
        arc_partner=tuple(new_partner),
        fixed_terminal_index=tuple(new_fixed),
        crossing_for_port=tuple(new_crossing_for),
        plus_partner=plus,minus_partner=minus,
    )


def index(p):return _yamada_iso.PreparedDiagramIndex(len(p.vertex_ids),[list(x) for x in p.ordered_ports],list(p.arc_partner),list(p.fixed_terminal_index),list(p.crossing_for_port))
class Memo:
    def __init__(self):self.b={};self.hits=0
    def get(self,p):
        x=index(p);k=(len(p.crossing_ids),x.node_count,x.fingerprint)
        for old,v in self.b.get(k,()):
            if x.isomorphic(old):self.hits+=1;return True,v,k,x
        return False,None,k,x
    def put(self,k,x,v):self.b.setdefault(k,[]).append((x,v))

def trial_inversion(p):
    for i in range(len(p.crossing_ids)):
        inv=sh.invert_crossing(p,i);reduced,moves=inv.reduce_reidemeister_ii()
        if moves:return i,reduced
    return None

def first_resolution(p):
    for i in range(len(p.crossing_ids)):
        try:return i,tuple(sh.resolve_crossing(p,i,s) for s in (0,1,2))
        except ValueError:pass
    return None

def evaluate(p,e,stats):
    m=Memo();stats.update(calls=0,bulk=0,bulk_seconds=0.0,inversions=0,resolutions=0,max_bulk_crossings=0)
    def bulk(q):
        t=time.perf_counter();v=e.compute_prepared_bulk_laurent(q);stats["bulk_seconds"]+=time.perf_counter()-t;stats["bulk"]+=1;stats["max_bulk_crossings"]=max(stats["max_bulk_crossings"],len(q.crossing_ids));return v
    def rec(q):
        stats["calls"]+=1;q,_=q.reduce_reidemeister_ii();hit,v,k,x=m.get(q)
        if hit:return v
        if len(q.crossing_ids)<=4:v=bulk(q)
        else:
            inv=trial_inversion(q)
            if inv is not None:
                i,r=inv;stats["inversions"]+=1;pl=rec(sh.resolve_crossing(q,i,0));mi=rec(sh.resolve_crossing(q,i,1));v=add(rec(r),sh._skein_delta(pl,mi))
            else:
                rr=first_resolution(q)
                if rr is None:v=bulk(q)
                else:
                    _,(pl,mi,ve)=rr;stats["resolutions"]+=1;v=add(add(shift(rec(pl),1),shift(rec(mi),-1)),rec(ve))
        m.put(k,x,v);return v
    out=rec(p);stats.update(iso_hits=m.hits,memo_size=sum(map(len,m.b.values())),native_graph_memo_size=e.memo_size);return out

def prepared(n):
    _,pd,_=torus.prepare_essential_torus(n);y=Yamada.from_PDCode(pd);return PreparedCompactStateBuilder.prepare(y.vertices,y.crossings,y.arcs,_ordered_crossing_ports)

def run(n,label,smoother):
    sh._smooth_crossing=smoother;p=prepared(n);e=NativeCompactEvaluator(PythonCompactYamadaEvaluator);stats={};expected=tuple(sorted(torus.independent_theta_terms(n).items()));t=time.perf_counter();got=evaluate(p,e,stats);secs=time.perf_counter()-t
    if got!=expected:raise AssertionError((n,label,got,expected))
    print(json.dumps({"n":n,"candidate":label,"seconds":secs,"stats":stats,"correctness":"PASS"},separators=(",",":")),flush=True)

def main():
    # Old smoother provides a timing/control baseline. General smoothing must
    # reproduce the external theorem while eliminating >4-crossing emergency leaves.
    for n in (13,17,19):
        run(n,"legacy_smoothing",ORIGINAL_SMOOTH)
        run(n,"general_self_adjacent_smoothing",general_smooth)
    sh._smooth_crossing=ORIGINAL_SMOOTH
if __name__=="__main__":main()
