from __future__ import annotations

import ast
import json
import math
import re
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial, select_projection

A = sp.Symbol("A")


def coords(n: int) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    # Break circular symmetry so independent chord intersections are generic.
    r = 5.0 * (1.0 + 0.07 * np.sin(3 * t + 0.31) + 0.025 * np.cos(5 * t))
    z = 0.55 * np.sin(2 * t + 0.27) + 0.23 * np.cos(5 * t - 0.11)
    return np.c_[r * np.cos(t), r * np.sin(t), z]


def adjacent(a: int, b: int, n: int) -> bool:
    d = (a - b) % n
    return d in (1, n - 1)


def chord_cross(e1, e2):
    a, b = sorted(e1)
    c, d = sorted(e2)
    if len({a, b, c, d}) < 4:
        return False
    return (a < c < b < d) or (c < a < d < b)


def crossing_score(edges) -> int:
    return sum(
        chord_cross(edges[i], edges[j])
        for i in range(len(edges))
        for j in range(i + 1, len(edges))
    )


def random_matching(n: int, rng) -> list[tuple[int, int]] | None:
    p = list(map(int, rng.permutation(n)))
    edges = []
    for k in range(0, n, 2):
        a, b = p[k], p[k + 1]
        if adjacent(a, b, n):
            return None
        edges.append(tuple(sorted((a, b))))
    return sorted(edges)


def matching_near(n: int, target: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    best = None
    best_error = math.inf
    for _ in range(20000):
        candidate = random_matching(n, rng)
        if candidate is None:
            continue
        error = abs(crossing_score(candidate) - target)
        if error < best_error:
            best = candidate
            best_error = error
        if error == 0:
            return candidate
    if best is None:
        raise RuntimeError(f"matching failure n={n}")
    return best


def make_graph(xyz, bridges):
    n = len(xyz)
    graph = nx.MultiGraph()
    for i, p in enumerate(xyz):
        graph.add_node(i, pos=np.asarray(p, float))
    for i in range(n):
        j = (i + 1) % n
        graph.add_edge(i, j, pts=np.vstack([xyz[i], xyz[j]]))
    for a, b in bridges:
        graph.add_edge(a, b, pts=np.vstack([xyz[a], xyz[b]]))
    return graph


def kg_signature(expr):
    expr = sp.expand(sp.cancel(expr))
    if expr == 0:
        return [0]
    terms = sp.Add.make_args(expr)
    lo = min(int(term.as_powers_dict().get(A, 0)) for term in terms)
    poly = sp.Poly(sp.expand(expr * A ** (-lo)), A)
    values = [int(poly.nth(i)) for i in range(poly.degree() + 1)]
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values


def parse_topoly(raw):
    s = str(raw).strip()
    try:
        value = ast.literal_eval(s)
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
    except Exception:
        pass
    q = s
    for char in "[](),;":
        q = q.replace(char, " ")
    tokens = q.split()
    if tokens and all(re.fullmatch(r"[-+]?\d+", token) for token in tokens):
        return [int(token) for token in tokens]
    return None


def transform(values, reverse, alternating, sign):
    out = list(values)
    if reverse:
        out.reverse()
    if alternating:
        out = [value * ((-1) ** i) for i, value in enumerate(out)]
    out = [sign * value for value in out]
    while len(out) > 1 and out[0] == 0:
        out.pop(0)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def run_kg(xyz, bridges, repeats=3):
    graph = make_graph(xyz, bridges)
    # Verify the shared XY projection is nondegenerate before timing.
    projection = select_projection(graph, rotation_angles=(0.0, 0.0, 0.0))
    times = []
    answer = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        answer = compute_yamada_polynomial(
            graph,
            A,
            rotation_angles=(0.0, 0.0, 0.0),
            normalize=True,
            n_jobs=1,
            method="negami",
            return_result=True,
        )
        times.append(time.perf_counter() - t0)
    return {
        "runtime_s": statistics.median(times),
        "crossings": projection.num_crossings,
        "polynomial": str(sp.expand(answer.polynomial)),
        "signature": kg_signature(answer.polynomial),
    }


def run_topoly(xyz, bridges, offset, repeats=3):
    from topoly import yamada
    from topoly.params import Closure, ReduceMethod, Translate

    times = []
    value = None
    shifted = [(a + offset, b + offset) for a, b in bridges]
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = yamada(
            np.asarray(xyz, float).tolist(),
            closure=Closure.CLOSED,
            tries=1,
            reduce_method=ReduceMethod.NO,
            max_cross=200,
            poly_reduce=True,
            translate=Translate.NO,
            hide_trivial=False,
            hide_rare=False,
            minimal=False,
            cuda=False,
            run_parallel=False,
            parallel_workers=1,
            bridges=shifted,
            breaks=[],
        )
        times.append(time.perf_counter() - t0)
    return {
        "runtime_s": statistics.median(times),
        "raw": str(value),
        "signature": parse_topoly(value),
    }


def calibration_cases():
    return [(6, 0), (8, 1), (10, 2)]


def detect_offset_and_convention():
    candidates = []
    for offset in (0, 1):
        observations = []
        valid = True
        for n, target in calibration_cases():
            xyz = coords(n)
            bridges = matching_near(n, target, 9000 + n)
            kg = run_kg(xyz, bridges, repeats=1)
            try:
                tp = run_topoly(xyz, bridges, offset, repeats=1)
            except Exception as exc:
                print("CALIBRATION_ERROR", offset, n, type(exc).__name__, str(exc))
                valid = False
                break
            if tp["signature"] is None:
                valid = False
                break
            observations.append((kg["signature"], tp["signature"]))
        if not valid:
            continue
        for reverse in (False, True):
            for alternating in (False, True):
                for sign in (1, -1):
                    score = sum(
                        transform(tp, reverse, alternating, sign) == kg
                        for kg, tp in observations
                    )
                    candidates.append(
                        (score, len(observations), offset, reverse, alternating, sign)
                    )
    if not candidates:
        raise RuntimeError("No parseable Topoly calibration convention.")
    candidates.sort(key=lambda item: (-item[0], item[2], item[3], item[4], item[5] == -1))
    best = candidates[0]
    print("CALIBRATION_CANDIDATES", json.dumps(candidates))
    if best[0] != best[1]:
        raise RuntimeError(f"No global convention matches all calibration cases: {best}")
    return {
        "offset": best[2],
        "reverse": best[3],
        "alternating": best[4],
        "sign": best[5],
    }


def main():
    convention = detect_offset_and_convention()
    print("CONVENTION", json.dumps(convention, sort_keys=True))

    rows = []
    # Held-out edge scaling. Target one bridge-projection crossing so E varies
    # without a rapidly growing crossing-state burden.
    for n in (12, 14, 16, 18):
        xyz = coords(n)
        bridges = matching_near(n, 1, 12000 + n)
        kg = run_kg(xyz, bridges)
        tp = run_topoly(xyz, bridges, convention["offset"])
        transformed = transform(
            tp["signature"],
            convention["reverse"],
            convention["alternating"],
            convention["sign"],
        )
        agree = transformed == kg["signature"]
        row = {
            "n": n,
            "V": n,
            "E": n + n // 2,
            "crossings": kg["crossings"],
            "kg_runtime_s": kg["runtime_s"],
            "topoly_runtime_s": tp["runtime_s"],
            "topoly_over_kg": tp["runtime_s"] / kg["runtime_s"],
            "agree": agree,
            "kg_signature": kg["signature"],
            "topoly_raw": tp["raw"],
        }
        rows.append(row)
        print("RESULT", json.dumps(row, separators=(",", ":")))
        if not agree:
            raise AssertionError(f"Held-out polynomial mismatch at n={n}")

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
