from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time

import sympy as sp

A = sp.Symbol("A")
HERE = Path(__file__).resolve().parent


def _load_helper():
    path = HERE / "benchmark_topoly_paper_scaling.py"
    spec = importlib.util.spec_from_file_location("ab40_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _terms(expr):
    out = {}
    for term in sp.expand(expr).as_ordered_terms():
        coeff, exponent = term.as_coeff_exponent(A)
        out[int(exponent)] = out.get(int(exponent), 0) + int(coeff)
    return {str(k): int(v) for k, v in sorted(out.items()) if v}


def worker(queue):
    try:
        from knotted_graph.invariants.yamada.polynomial import Yamada
        from knotted_graph.invariants.yamada.native import native_available
        if not native_available():
            raise RuntimeError("native backend unavailable")
        mod = _load_helper()
        _graph, processor, pdcode = mod._prepare_crossing(40, 4)
        vertices = list(processor.vertices.values())
        crossings = list(processor.crossings.values())
        arcs = list(processor.arcs.values())
        assert len(crossings) == 40
        start = time.perf_counter()
        ans = Yamada(vertices, crossings, arcs).compute(A, normalize=False, n_jobs=1, method="negami")
        elapsed = time.perf_counter() - start
        terms = _terms(ans)
        payload = json.dumps(terms, sort_keys=True, separators=(",", ":")).encode()
        queue.put({"time_s": elapsed, "pd_hash": hashlib.sha256(pdcode.encode()).hexdigest(), "terms_hash": hashlib.sha256(payload).hexdigest(), "terms": terms})
    except BaseException as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def run(label: str, output: str):
    ctx = mp.get_context("spawn")
    rows = []
    for _ in range(5):
        q = ctx.Queue(); p = ctx.Process(target=worker, args=(q,)); p.start(); p.join(300)
        if p.is_alive():
            p.terminate(); p.join(); raise TimeoutError("40-crossing case timed out")
        row = q.get()
        if "error" in row: raise RuntimeError(row["error"])
        rows.append(row)
    assert len({r["pd_hash"] for r in rows}) == 1
    assert len({r["terms_hash"] for r in rows}) == 1
    result = {"label": label, "median_s": statistics.median(r["time_s"] for r in rows), "times_s": [r["time_s"] for r in rows], "pd_hash": rows[0]["pd_hash"], "terms_hash": rows[0]["terms_hash"], "terms": rows[0]["terms"]}
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


def compare(base: str, latest: str):
    b=json.loads(Path(base).read_text()); n=json.loads(Path(latest).read_text())
    assert b["pd_hash"] == n["pd_hash"]
    assert b["terms_hash"] == n["terms_hash"] and b["terms"] == n["terms"]
    ratio=b["median_s"]/n["median_s"]
    print(f"BASE_MEDIAN_S={b['median_s']:.9f}")
    print(f"LATEST_MEDIAN_S={n['median_s']:.9f}")
    print(f"SPEEDUP={ratio:.6f}x")
    print(f"IMPROVEMENT={(ratio-1)*100:+.2f}%")
    print("EXACT_OUTPUT_MATCH=PASS")


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd", required=True)
    r=sub.add_parser("run"); r.add_argument("label"); r.add_argument("output")
    c=sub.add_parser("compare"); c.add_argument("base"); c.add_argument("latest")
    a=p.parse_args(); run(a.label,a.output) if a.cmd=="run" else compare(a.base,a.latest)
