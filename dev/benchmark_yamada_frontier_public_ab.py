from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time

import networkx as nx
import numpy as np
import sympy as sp

A = sp.Symbol("A")
HERE = Path(__file__).resolve().parent


def _line(a, b, samples=2):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    t=np.linspace(0.0,1.0,int(samples))[:,None]
    return (1.0-t)*a+t*b


def _resample(points,samples=121):
    points=np.asarray(points,dtype=float); delta=np.diff(points,axis=0)
    lengths=np.linalg.norm(delta,axis=1); cumulative=np.concatenate([[0.0],np.cumsum(lengths)])
    target=np.linspace(0.0,cumulative[-1],int(samples)); out=np.empty((len(target),3),dtype=float)
    for dim in range(3): out[:,dim]=np.interp(target,cumulative,points[:,dim])
    return out


def _infinity_plus_pair(base_path,width,depth,samples=120):
    base=_resample(base_path,samples=samples); xy=base[:,:2]
    tangent=np.gradient(xy,axis=0); norm=np.linalg.norm(tangent,axis=1); norm[norm<1e-12]=1.0
    tangent=tangent/norm[:,None]; normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    t=np.linspace(0.0,1.0,len(base)); lateral=np.sin(2*np.pi*t)*np.sin(np.pi*t)
    z=np.exp(-((t-0.5)/0.070)**2)*np.sin(np.pi*t)**2
    over=base.copy(); under=base.copy(); over[:,:2]+=width*lateral[:,None]*normal; under[:,:2]-=width*lateral[:,None]*normal
    over[:,2]+=depth*z; under[:,2]-=depth*z; over[0]=under[0]=base[0]; over[-1]=under[-1]=base[-1]
    return over,under


def cycle_graph(n):
    radius=2.05; angles=np.pi/4+2*np.pi*np.arange(n)/n
    pos={k:np.array([radius*np.cos(a),radius*np.sin(a),0.0]) for k,a in enumerate(angles)}
    graph=nx.MultiGraph()
    for k,p in pos.items(): graph.add_node(k,pos=p.copy())
    side=float(np.linalg.norm(pos[1]-pos[0])); width=min(0.26,0.13*side); depth=min(0.18,0.085*side)
    for k in range(n):
        j=(k+1)%n; over,under=_infinity_plus_pair(_line(pos[k],pos[j]),width,depth)
        graph.add_edge(k,j,pts=over,crossing_role="over"); graph.add_edge(k,j,pts=under,crossing_role="under")
    return graph


def _load_torus():
    path=HERE/"benchmark_topoly_essential_torus_scaling.py"
    spec=importlib.util.spec_from_file_location("frontier_ab_torus",path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod


def _terms(expr):
    out={}
    for term in sp.expand(expr).as_ordered_terms():
        coeff,power=term.as_coeff_exponent(A); out[int(power)]=out.get(int(power),0)+int(coeff)
    return {str(k):int(v) for k,v in sorted(out.items()) if v}


def _prepare(case):
    from knotted_graph.projection import PDCode
    if case.startswith("cycle"):
        n=int(case.replace("cycle","")); processor=PDCode(cycle_graph(n)); pd=processor.compute(rotation_angles=(0.0,0.0,0.0))
        assert len(processor.crossings)==n; return processor,pd
    if case=="torus11":
        _g,processor,pd=_load_torus().prepare_essential_torus(11); return processor,pd
    raise ValueError(case)


def worker(case,queue):
    try:
        from knotted_graph.invariants.yamada.native import native_available
        from knotted_graph.invariants.yamada.polynomial import Yamada
        if not native_available(): raise RuntimeError("native unavailable")
        processor,pd=_prepare(case); vertices=list(processor.vertices.values()); crossings=list(processor.crossings.values()); arcs=list(processor.arcs.values())
        start=time.perf_counter(); ans=Yamada(vertices,crossings,arcs).compute(A,normalize=False,n_jobs=1,method="negami"); elapsed=time.perf_counter()-start
        terms=_terms(ans); blob=json.dumps(terms,sort_keys=True,separators=(",",":")).encode()
        queue.put({"time_s":elapsed,"pd_hash":hashlib.sha256(pd.encode()).hexdigest(),"terms_hash":hashlib.sha256(blob).hexdigest(),"terms":terms})
    except BaseException as exc: queue.put({"error":f"{type(exc).__name__}: {exc}"})


def measure(case,repeats=3):
    ctx=mp.get_context("spawn"); rows=[]
    for _ in range(repeats):
        q=ctx.Queue(); p=ctx.Process(target=worker,args=(case,q)); p.start(); p.join(300)
        if p.is_alive(): p.terminate(); p.join(); raise TimeoutError(case)
        row=q.get();
        if "error" in row: raise RuntimeError(row["error"])
        rows.append(row)
    assert len({r["pd_hash"] for r in rows})==1 and len({r["terms_hash"] for r in rows})==1
    return {"median_s":statistics.median(r["time_s"] for r in rows),"times_s":[r["time_s"] for r in rows],"pd_hash":rows[0]["pd_hash"],"terms_hash":rows[0]["terms_hash"],"terms":rows[0]["terms"]}


def run(label,output):
    result={"label":label,"cases":{case:measure(case) for case in ("cycle10","cycle11","torus11")}}
    Path(output).write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,sort_keys=True))


def compare(base,candidate):
    b=json.loads(Path(base).read_text()); c=json.loads(Path(candidate).read_text())
    for case in b["cases"]:
        old=b["cases"][case]; new=c["cases"][case]
        assert old["pd_hash"]==new["pd_hash"]; assert old["terms_hash"]==new["terms_hash"] and old["terms"]==new["terms"]
        ratio=old["median_s"]/new["median_s"]
        print(f"{case} BASE={old['median_s']:.9f}s CANDIDATE={new['median_s']:.9f}s SPEEDUP={ratio:.6f}x IMPROVEMENT={(ratio-1)*100:+.2f}% EXACT=PASS")
        print(f"  base_times={old['times_s']}"); print(f"  candidate_times={new['times_s']}")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    r=sub.add_parser("run"); r.add_argument("label"); r.add_argument("output")
    c=sub.add_parser("compare"); c.add_argument("base"); c.add_argument("candidate")
    args=p.parse_args(); run(args.label,args.output) if args.cmd=="run" else compare(args.base,args.candidate)
