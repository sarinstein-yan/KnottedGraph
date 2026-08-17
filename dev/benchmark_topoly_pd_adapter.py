from __future__ import annotations

import ast
import json
import re
import statistics
import time

import numpy as np

from benchmark_topoly_common import (
    calibration_cases,
    coords,
    matching_near,
    run_kg,
    transform,
)


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


def run_topoly_via_pd(xyz, bridges, offset, repeats=3):
    """Use Topoly's own coordinate/bridge parser and PD generator, then Yamada.

    Topoly 1.1.0's one-call spatial path drops bridge metadata in
    ``Invariant.calculate_spatial``. This adapter retains the intended graph by
    materializing the PD code that Topoly itself generated before calling its
    Yamada engine on that code.
    """
    from topoly import yamada
    from topoly.invariants import Invariant
    from topoly.params import Translate

    shifted = [(a + offset, b + offset) for a, b in bridges]
    times = []
    raw = None
    pdcode = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        parsed = Invariant(
            np.asarray(xyz, float).tolist(),
            bridges=shifted,
            breaks=[],
        )
        pdcode = parsed.pdcode
        if not pdcode:
            raise RuntimeError(
                "Topoly coordinate/bridge parser did not generate a PD code."
            )
        raw = yamada(
            pdcode,
            max_cross=200,
            poly_reduce=True,
            translate=Translate.NO,
            hide_trivial=False,
            hide_rare=False,
            minimal=False,
            cuda=False,
            run_parallel=False,
            parallel_workers=1,
        )
        times.append(time.perf_counter() - t0)

    return {
        "runtime_s": statistics.median(times),
        "raw": str(raw),
        "signature": parse_topoly(raw),
        "pdcode": pdcode,
    }


def detect_convention():
    candidates = []
    observations = []

    # The diagnostic showed zero-based bridge indexing is the meaningful mode:
    # it produced all intended bridge arcs and a nonempty spatial-graph PD code.
    offset = 0

    for n, target in calibration_cases():
        xyz = coords(n)
        bridges = matching_near(n, target, 9000 + n)
        kg = run_kg(xyz, bridges, repeats=1)
        tp = run_topoly_via_pd(xyz, bridges, offset, repeats=1)
        print(
            "CALIBRATION_OBS",
            json.dumps(
                {
                    "n": n,
                    "kg_signature": kg["signature"],
                    "topoly_raw": tp["raw"],
                    "topoly_signature": tp["signature"],
                    "topoly_pdcode": tp["pdcode"],
                },
                separators=(",", ":"),
            ),
        )
        if tp["signature"] is None:
            raise RuntimeError(f"Unparseable Topoly output at n={n}: {tp['raw']}")
        observations.append((kg["signature"], tp["signature"]))

    for reverse in (False, True):
        for alternating in (False, True):
            for sign in (1, -1):
                score = sum(
                    transform(tp, reverse, alternating, sign) == kg
                    for kg, tp in observations
                )
                candidates.append(
                    dict(
                        score=score,
                        total=len(observations),
                        reverse=reverse,
                        alternating=alternating,
                        sign=sign,
                        offset=offset,
                    )
                )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["reverse"],
            item["alternating"],
            item["sign"] == -1,
        )
    )
    best = candidates[0]
    print("CONVENTION_CANDIDATES", json.dumps(candidates, separators=(",", ":")))
    if best["score"] != best["total"]:
        raise RuntimeError(
            "Topoly-via-PD still does not agree under the tested global "
            f"Laurent conventions. Best={best}"
        )
    return best


def main():
    convention = detect_convention()
    print("CONVENTION", json.dumps(convention, separators=(",", ":")))

    rows = []
    for n in (12, 14, 16, 18):
        xyz = coords(n)
        bridges = matching_near(n, 1, 12000 + n)
        kg = run_kg(xyz, bridges, repeats=3)
        tp = run_topoly_via_pd(
            xyz,
            bridges,
            convention["offset"],
            repeats=3,
        )
        transformed = transform(
            tp["signature"],
            convention["reverse"],
            convention["alternating"],
            convention["sign"],
        )
        agree = transformed == kg["signature"]
        row = {
            "V": n,
            "E": n + n // 2,
            "crossings": kg["crossings"],
            "kg_runtime_s": kg["runtime_s"],
            "topoly_runtime_s": tp["runtime_s"],
            "topoly_over_kg": tp["runtime_s"] / kg["runtime_s"],
            "agree": agree,
            "kg_signature": kg["signature"],
            "topoly_raw": tp["raw"],
            "topoly_pdcode": tp["pdcode"],
        }
        rows.append(row)
        print("RESULT", json.dumps(row, separators=(",", ":")))
        if not agree:
            raise AssertionError(f"Held-out polynomial mismatch at n={n}")

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
