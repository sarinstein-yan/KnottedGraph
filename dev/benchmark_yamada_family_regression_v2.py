from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import sympy as sp

A = sp.Symbol("A")
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "User_guide" / "benchmarks" / "03_knottedgraph_vs_topoly_scaling_final_push.ipynb"


def _load_reference_constructors():
    nb = json.loads(NOTEBOOK.read_text())
    namespace = {}
    target = None
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if (
            "def reference_dv_theta_graph" in source
            and "def reference_lllv_cycle_graph" in source
            and "def reference_lllv_theta_graph" in source
        ):
            target = source
            break
    if target is None:
        raise RuntimeError("paper-reference constructor cell not found")
    exec(compile(target, str(NOTEBOOK), "exec"), namespace)
    return {
        "THETA_N_VALUES": namespace["reference_dv_theta_graph"],
        "LLLV_CYCLE_N_VALUES": namespace["reference_lllv_cycle_graph"],
        "LLLV_THETA_S_VALUES": namespace["reference_lllv_theta_graph"],
    }


def _terms(expr):
    out = {}
    for term in sp.expand(expr).as_ordered_terms():
        coeff, power = term.as_coeff_exponent(A)
        out[int(power)] = out.get(int(power), 0) + int(coeff)
    return {str(k): int(v) for k, v in sorted(out.items()) if v}


def _compute(graph):
    from knotted_graph.invariants.yamada.polynomial import Yamada
    from knotted_graph.projection import PDCode

    processor = PDCode(graph)
    pd = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calc = Yamada.from_PDCode(processor)
    start = time.perf_counter()
    expr = calc.compute(A, normalize=False, n_jobs=1, method="negami")
    elapsed = time.perf_counter() - start
    terms = _terms(expr)
    return elapsed, {
        "crossings": len(processor.crossings),
        "pd_hash": hashlib.sha256(pd.encode()).hexdigest(),
        "terms_hash": hashlib.sha256(json.dumps(terms, sort_keys=True).encode()).hexdigest(),
        "terms": terms,
    }


def measure_case(label: str, family: str, n: int, output: str):
    constructor = _load_reference_constructors()[family]
    graph = constructor(int(n))
    _compute(graph)
    times = []
    meta = None
    for _ in range(3):
        elapsed, current = _compute(graph)
        times.append(elapsed)
        if meta is None:
            meta = current
        else:
            assert current == meta
    result = {
        "label": label,
        "family": family,
        "n": int(n),
        **meta,
        "median_s": statistics.median(times),
        "times_s": times,
    }
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)


def compare_case(base_path: str, new_path: str):
    old = json.loads(Path(base_path).read_text())
    cur = json.loads(Path(new_path).read_text())
    assert old["family"] == cur["family"] and old["n"] == cur["n"]
    assert old["pd_hash"] == cur["pd_hash"]
    assert old["terms_hash"] == cur["terms_hash"]
    assert old["terms"] == cur["terms"]
    ratio = old["median_s"] / cur["median_s"]
    change = (cur["median_s"] / old["median_s"] - 1.0) * 100.0
    tag = "IMPROVE" if ratio > 1.05 else "REGRESS" if ratio < 0.95 else "NEUTRAL"
    print(
        f"{cur['family']} n={cur['n']:2d} c={cur['crossings']:2d} "
        f"old={old['median_s']:.6f}s new={cur['median_s']:.6f}s "
        f"old/new={ratio:.3f}x change={change:+.1f}% {tag} EXACT=PASS"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure-case")
    m.add_argument("label")
    m.add_argument("family")
    m.add_argument("n", type=int)
    m.add_argument("output")
    c = sub.add_parser("compare-case")
    c.add_argument("base")
    c.add_argument("new")
    args = p.parse_args()
    if args.cmd == "measure-case":
        measure_case(args.label, args.family, args.n, args.output)
    else:
        compare_case(args.base, args.new)
