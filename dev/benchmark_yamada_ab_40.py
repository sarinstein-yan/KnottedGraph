from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

A = sp.Symbol("A")


def _line(a, b, samples: int = 2) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = np.linspace(0.0, 1.0, int(samples))[:, None]
    return (1.0 - t) * a + t * b


def _resample_polyline(points: np.ndarray, samples: int = 121) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total <= 1e-12:
        raise ValueError("base path has zero length")
    target = np.linspace(0.0, total, int(samples))
    out = np.empty((len(target), 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(target, cumulative, points[:, dim])
    return out


def _infinity_plus_pair(base_path: np.ndarray, *, width: float, depth: float, samples: int = 120):
    base = _resample_polyline(base_path, samples=samples)
    xy = base[:, :2]
    tangent = np.gradient(xy, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm < 1e-12] = 1.0
    tangent = tangent / norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    t = np.linspace(0.0, 1.0, len(base))
    lateral = np.sin(2.0 * np.pi * t) * np.sin(np.pi * t)
    z_profile = np.exp(-((t - 0.5) / 0.070) ** 2) * np.sin(np.pi * t) ** 2
    over = base.copy(); under = base.copy()
    over[:, :2] += float(width) * lateral[:, None] * normal
    under[:, :2] -= float(width) * lateral[:, None] * normal
    over[:, 2] += float(depth) * z_profile
    under[:, 2] -= float(depth) * z_profile
    over[0] = under[0] = base[0]
    over[-1] = under[-1] = base[-1]
    return over, under


def reference_lllv_cycle_graph(n: int) -> nx.MultiGraph:
    radius = 2.05
    angles = np.pi / 4 + 2.0 * np.pi * np.arange(n) / n
    positions = {k: np.array([radius * np.cos(a), radius * np.sin(a), 0.0]) for k, a in enumerate(angles)}
    graph = nx.MultiGraph()
    for k, pos in positions.items():
        graph.add_node(k, pos=pos.copy())
    side = float(np.linalg.norm(positions[1] - positions[0]))
    width = min(0.26, 0.13 * side)
    depth = min(0.18, 0.085 * side)
    for k in range(n):
        j = (k + 1) % n
        over, under = _infinity_plus_pair(_line(positions[k], positions[j], 2), width=width, depth=depth)
        common = {"replacement_piece": "infinity_plus", "base_edge_index": int(k)}
        graph.add_edge(k, j, pts=over, role="infinity_plus_over", crossing_role="over", **common)
        graph.add_edge(k, j, pts=under, role="infinity_plus_under", crossing_role="under", **common)
    return graph


def _terms(expr):
    out = {}
    for term in sp.expand(expr).as_ordered_terms():
        coeff, exponent = term.as_coeff_exponent(A)
        out[int(exponent)] = out.get(int(exponent), 0) + int(coeff)
    return {str(k): int(v) for k, v in sorted(out.items()) if v}


def worker(n: int, queue):
    try:
        from knotted_graph.invariants.yamada.polynomial import Yamada
        from knotted_graph.invariants.yamada.native import native_available
        from knotted_graph.projection import PDCode
        if not native_available():
            raise RuntimeError("native backend unavailable")
        graph = reference_lllv_cycle_graph(n)
        processor = PDCode(graph)
        pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        crossings = list(processor.crossings.values())
        if len(crossings) != n:
            raise AssertionError(f"cycle:{n}: expected {n} crossings, got {len(crossings)}")
        vertices = list(processor.vertices.values())
        arcs = list(processor.arcs.values())
        start = time.perf_counter()
        ans = Yamada(vertices, crossings, arcs).compute(A, normalize=False, n_jobs=1, method="negami")
        elapsed = time.perf_counter() - start
        terms = _terms(ans)
        payload = json.dumps(terms, sort_keys=True, separators=(",", ":")).encode()
        queue.put({"time_s": elapsed, "pd_hash": hashlib.sha256(pdcode.encode()).hexdigest(), "terms_hash": hashlib.sha256(payload).hexdigest(), "terms": terms})
    except BaseException as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def measure(n: int, repeats: int = 3):
    ctx = mp.get_context("spawn")
    rows = []
    for _ in range(repeats):
        q = ctx.Queue(); p = ctx.Process(target=worker, args=(n, q)); p.start(); p.join(300)
        if p.is_alive():
            p.terminate(); p.join(); raise TimeoutError(f"cycle:{n} timed out")
        row = q.get()
        if "error" in row: raise RuntimeError(row["error"])
        rows.append(row)
    assert len({r["pd_hash"] for r in rows}) == 1
    assert len({r["terms_hash"] for r in rows}) == 1
    return {"n": n, "median_s": statistics.median(r["time_s"] for r in rows), "times_s": [r["time_s"] for r in rows], "pd_hash": rows[0]["pd_hash"], "terms_hash": rows[0]["terms_hash"], "terms": rows[0]["terms"]}


def run(label: str, output: str):
    result = {"label": label, "cases": {str(n): measure(n) for n in (10, 11)}}
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


def compare(base: str, latest: str):
    b = json.loads(Path(base).read_text()); n = json.loads(Path(latest).read_text())
    for key in ("10", "11"):
        old = b["cases"][key]; new = n["cases"][key]
        assert old["pd_hash"] == new["pd_hash"]
        assert old["terms_hash"] == new["terms_hash"] and old["terms"] == new["terms"]
        ratio = old["median_s"] / new["median_s"]
        print(f"cycle:{key}:infinity_plus BASE_MEDIAN_S={old['median_s']:.9f}")
        print(f"cycle:{key}:infinity_plus LATEST_MEDIAN_S={new['median_s']:.9f}")
        print(f"cycle:{key}:infinity_plus SPEEDUP={ratio:.6f}x")
        print(f"cycle:{key}:infinity_plus IMPROVEMENT={(ratio-1)*100:+.2f}%")
        print(f"cycle:{key}:infinity_plus EXACT_OUTPUT_MATCH=PASS")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("label"); r.add_argument("output")
    c = sub.add_parser("compare"); c.add_argument("base"); c.add_argument("latest")
    a = p.parse_args(); run(a.label, a.output) if a.cmd == "run" else compare(a.base, a.latest)
