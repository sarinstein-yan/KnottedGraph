from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time

import sympy as sp

A = sp.Symbol("A")
HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _terms(expr):
    out = {}
    for term in sp.expand(expr).as_ordered_terms():
        coeff, exponent = term.as_coeff_exponent(A)
        out[int(exponent)] = out.get(int(exponent), 0) + int(coeff)
    return {str(k): int(v) for k, v in sorted(out.items()) if v}


def _mirror_graph(graph):
    mirrored = graph.copy()
    for _node, data in mirrored.nodes(data=True):
        if "pos" in data:
            data["pos"] = data["pos"].copy()
            data["pos"][2] *= -1.0
    for _u, _v, _key, data in mirrored.edges(keys=True, data=True):
        if "pts" in data:
            data["pts"] = data["pts"].copy()
            data["pts"][:, 2] *= -1.0
    return mirrored


def _prepare(case: str):
    from knotted_graph.projection import PDCode

    if case.startswith("torus"):
        mod = _load_module(
            "ab_torus_helper",
            HERE / "benchmark_topoly_essential_torus_scaling.py",
        )
        mirror = case.endswith("_mirror")
        n = int(case.split("_")[0].replace("torus", ""))
        if mirror:
            graph = _mirror_graph(mod.essential_torus_graph(n))
            processor = PDCode(graph)
            pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        else:
            _graph, processor, pdcode = mod.prepare_essential_torus(n)
        return processor, pdcode

    if case.startswith("random20_s"):
        sample_index = int(case.rsplit("s", 1)[1])
        mod = _load_module(
            "ab_random_helper",
            HERE / "benchmark_topoly_random_cubic_ensemble.py",
        )
        ensemble = mod.topology_ensemble(20, sample_index + 1, mod.DEFAULT_SEED)
        sample, abstract = ensemble[sample_index]
        _embedded, processor, pdcode, _embedding_attempt = mod.prepare_sample(
            sample, abstract, mod.DEFAULT_SEED
        )
        return processor, pdcode

    if case.startswith("controlled"):
        left, graph_part = case.split("_g")
        crossings = int(left.replace("controlled", ""))
        graph_index = int(graph_part)
        mod = _load_module(
            "ab_controlled_helper",
            HERE / "benchmark_topoly_paper_scaling.py",
        )
        _graph, processor, pdcode = mod._prepare_crossing(crossings, graph_index)
        return processor, pdcode

    raise ValueError(case)


def _single_worker(case: str, queue):
    try:
        from knotted_graph.invariants.yamada.polynomial import Yamada
        from knotted_graph.invariants.yamada.native import native_available

        if not native_available():
            raise RuntimeError("native Yamada backend unavailable")
        processor, pdcode = _prepare(case)
        vertices = list(processor.vertices.values())
        crossings = list(processor.crossings.values())
        arcs = list(processor.arcs.values())

        # Preparation/projection/imports are intentionally excluded from timing.
        start = time.perf_counter()
        result = Yamada(vertices, crossings, arcs).compute(
            A,
            normalize=False,
            n_jobs=1,
            method="negami",
        )
        elapsed = time.perf_counter() - start
        terms = _terms(result)
        payload = json.dumps(terms, sort_keys=True, separators=(",", ":")).encode()
        queue.put(
            {
                "status": "ok",
                "case": case,
                "crossings": len(crossings),
                "pd_hash": hashlib.sha256(pdcode.encode()).hexdigest(),
                "terms": terms,
                "terms_hash": hashlib.sha256(payload).hexdigest(),
                "time_s": elapsed,
            }
        )
    except BaseException as exc:
        queue.put({"status": "error", "case": case, "error": f"{type(exc).__name__}: {exc}"})


def _measure(case: str, repeats: int, timeout_s: float):
    rows = []
    ctx = mp.get_context("spawn")
    for _ in range(repeats):
        queue = ctx.Queue()
        proc = ctx.Process(target=_single_worker, args=(case, queue))
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            raise TimeoutError(f"{case} exceeded {timeout_s}s")
        if queue.empty():
            raise RuntimeError(f"{case}: worker exited {proc.exitcode} without result")
        row = queue.get()
        if row["status"] != "ok":
            raise RuntimeError(row["error"])
        rows.append(row)

    pd_hashes = {row["pd_hash"] for row in rows}
    term_hashes = {row["terms_hash"] for row in rows}
    if len(pd_hashes) != 1 or len(term_hashes) != 1:
        raise AssertionError(f"non-deterministic result for {case}")
    times = [row["time_s"] for row in rows]
    return {
        "case": case,
        "crossings": rows[0]["crossings"],
        "pd_hash": rows[0]["pd_hash"],
        "terms_hash": rows[0]["terms_hash"],
        "terms": rows[0]["terms"],
        "times_s": times,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
    }


def run(label: str, output: Path, repeats: int, timeout_s: float):
    cases = [
        "torus9",
        "torus11",
        "torus11_mirror",
        "random20_s1",
        "controlled16_g4",
        "controlled32_g4",
    ]
    results = []
    for case in cases:
        row = _measure(case, repeats, timeout_s)
        results.append(row)
        print(
            f"{label} {case}: c={row['crossings']} median={row['median_s']:.9f}s "
            f"times={row['times_s']}",
            flush=True,
        )
    payload = {"label": label, "repeats": repeats, "cases": results}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compare(base_path: Path, latest_path: Path):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    base_rows = {row["case"]: row for row in base["cases"]}
    latest_rows = {row["case"]: row for row in latest["cases"]}
    if set(base_rows) != set(latest_rows):
        raise AssertionError("case sets differ")

    ratios = []
    base_total = 0.0
    latest_total = 0.0
    print("\nA/B RESULTS (baseline / latest):")
    for case in base_rows:
        old = base_rows[case]
        new = latest_rows[case]
        if old["pd_hash"] != new["pd_hash"]:
            raise AssertionError(f"{case}: PD differs between branches")
        if old["terms_hash"] != new["terms_hash"] or old["terms"] != new["terms"]:
            raise AssertionError(f"{case}: polynomial differs between branches")
        ratio = old["median_s"] / new["median_s"]
        pct = (ratio - 1.0) * 100.0
        ratios.append(ratio)
        base_total += old["median_s"]
        latest_total += new["median_s"]
        print(
            f"{case:20s} c={old['crossings']:2d} old={old['median_s']:.9f}s "
            f"new={new['median_s']:.9f}s speedup={ratio:.4f}x improvement={pct:+.2f}%"
        )

    geometric = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    median_ratio = statistics.median(ratios)
    total_ratio = base_total / latest_total
    print(f"GEOMEAN_SPEEDUP={geometric:.6f}x ({(geometric - 1)*100:+.2f}%)")
    print(f"MEDIAN_CASE_SPEEDUP={median_ratio:.6f}x ({(median_ratio - 1)*100:+.2f}%)")
    print(f"SUM_OF_CASE_MEDIANS_SPEEDUP={total_ratio:.6f}x ({(total_ratio - 1)*100:+.2f}%)")
    print(f"BASE_SUM_MEDIANS_S={base_total:.9f}")
    print(f"LATEST_SUM_MEDIANS_S={latest_total:.9f}")
    print("EXACT_OUTPUT_MATCH=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--repeats", type=int, default=5)
    run_parser.add_argument("--timeout", type=float, default=120.0)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("latest", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        run(args.label, args.output, args.repeats, args.timeout)
    else:
        compare(args.base, args.latest)
