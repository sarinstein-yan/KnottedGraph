from __future__ import annotations

import argparse
import json
import statistics

from benchmark_topoly_essential_torus_scaling import prepare_essential_torus
import benchmark_topoly_paper_scaling as base

DEFAULT_N_VALUES = (3, 5, 7, 9, 11, 13, 15, 17, 19)
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_S = 30.0


def _one(framework, processor, pdcode, timeout_s):
    result = base._run_with_timeout(framework, processor, pdcode, timeout_s)
    if result["status"] != "ok":
        raise AssertionError(
            f"{framework} did not finish within {timeout_s}s: {result}"
        )
    return result


def benchmark_n(n: int, repeats: int, timeout_s: float):
    _graph, processor, pdcode = prepare_essential_torus(n)
    kg_times = []
    tp_times = []
    convention = None

    for repeat in range(repeats):
        kg = _one("knottedgraph", processor, pdcode, timeout_s)
        tp = _one("topoly", processor, pdcode, timeout_s)
        sign, orientation, shift = base._validate_laurent_unit(
            kg["terms"], tp["terms"]
        )
        current = (sign, orientation, shift)
        if convention is None:
            convention = current
        elif current != convention:
            raise AssertionError(
                f"T(2,{n}) convention changed across repeats: "
                f"{convention} -> {current}"
            )
        kg_times.append(float(kg["time_s"]))
        tp_times.append(float(tp["time_s"]))
        print(
            json.dumps(
                {
                    "n": n,
                    "repeat": repeat,
                    "knottedgraph_s": kg_times[-1],
                    "topoly_s": tp_times[-1],
                    "correctness": "PASS",
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    kg_median = statistics.median(kg_times)
    tp_median = statistics.median(tp_times)
    row = {
        "n": n,
        "crossings": n,
        "repeats": repeats,
        "knottedgraph_median_s": kg_median,
        "topoly_median_s": tp_median,
        "topoly_over_knottedgraph": tp_median / kg_median,
        "knottedgraph_faster": kg_median < tp_median,
        "unit_sign_topoly_over_kg": convention[0],
        "variable_orientation": convention[1],
        "monomial_shift_topoly_minus_kg": convention[2],
        "correctness": "PASS",
    }
    print("MEDIAN=" + json.dumps(row, separators=(",", ":")), flush=True)
    return row


def main(n_values, repeats: int, timeout_s: float, require_faster: bool):
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rows = [benchmark_n(n, repeats, timeout_s) for n in n_values]

    if require_faster:
        failures = [row for row in rows if not row["knottedgraph_faster"]]
        if failures:
            raise AssertionError(
                "KnottedGraph did not beat Topoly at every tested certified "
                f"T(2,n) crossing count: {failures}"
            )

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-values",
        default=",".join(map(str, DEFAULT_N_VALUES)),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--require-faster", action="store_true")
    args = parser.parse_args()
    values = [int(value) for value in args.n_values.split(",") if value.strip()]
    if values != sorted(set(values)):
        raise ValueError("n values must be unique and increasing")
    if any(n < 3 or n % 2 == 0 for n in values):
        raise ValueError("T(2,n) gate requires odd n >= 3")
    main(values, args.repeats, args.timeout, args.require_faster)
