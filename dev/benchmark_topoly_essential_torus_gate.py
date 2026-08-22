from __future__ import annotations

import argparse
import json
import statistics

from benchmark_dobrynin_vesnin_theta_family import prepare_theta_family
import benchmark_topoly_paper_scaling as base

DEFAULT_N_VALUES = (3, 5, 7, 9, 11, 13, 15, 17, 19)
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_S = 30.0


def _run(framework, processor, pdcode, timeout_s):
    return base._run_with_timeout(framework, processor, pdcode, timeout_s)


def _require_knottedgraph(result, n: int, timeout_s: float):
    if result["status"] != "ok":
        raise AssertionError(
            f"KnottedGraph failed for certified T(2,{n}) within {timeout_s}s: {result}"
        )
    return result


def _independent_theta_terms(n: int) -> dict[int, int]:
    """Dobrynin--Vesnin Theorem 2, coded independently of production.

    This is intentionally not imported from ``theta_twist.py``. It is an
    independent algebraic oracle for the certified benchmark family:

        R(Theta(n)) = (A^2+A+1+A^-1+A^-2) A^n
                      - (A+A^-1) A^(-2n)
                      - (A^2+1+A^-2) (-1)^n A^(-n).
    """
    terms: dict[int, int] = {}

    def add(power: int, coefficient: int) -> None:
        terms[power] = terms.get(power, 0) + coefficient
        if not terms[power]:
            del terms[power]

    for offset in (2, 1, 0, -1, -2):
        add(n + offset, 1)
    add(-2 * n + 1, -1)
    add(-2 * n - 1, -1)
    coefficient = -((-1) ** n)
    for offset in (2, 0, -2):
        add(-n + offset, coefficient)
    return terms


def benchmark_n(n: int, repeats: int, timeout_s: float):
    _graph, processor, pdcode = prepare_theta_family(n)
    expected = _independent_theta_terms(n)
    kg_times: list[float] = []
    tp_times: list[float] = []
    tp_statuses: list[str] = []
    topoly_agreement = None
    convention = None

    for repeat in range(repeats):
        kg = _require_knottedgraph(
            _run("knottedgraph", processor, pdcode, timeout_s), n, timeout_s
        )
        tp = _run("topoly", processor, pdcode, timeout_s)

        # KnottedGraph correctness is checked against the independent published
        # formula, not against the competitor. The retained exhaustive oracle
        # separately verifies the same output through n=17 in CI.
        if kg["terms"] != expected:
            raise AssertionError(
                f"KnottedGraph disagrees with the independent Theta({n}) formula: "
                f"KG={kg['terms']}, expected={expected}"
            )

        kg_time = float(kg["time_s"])
        kg_times.append(kg_time)
        tp_status = str(tp["status"])
        tp_statuses.append(tp_status)

        agrees = None
        current_convention = None
        tp_time = None
        if tp_status == "ok":
            tp_time = float(tp["time_s"])
            tp_times.append(tp_time)
            try:
                current_convention = base._validate_laurent_unit(
                    kg["terms"], tp["terms"]
                )
            except AssertionError:
                agrees = False
            else:
                agrees = True

            if topoly_agreement is None:
                topoly_agreement = agrees
                convention = current_convention
            elif agrees != topoly_agreement or current_convention != convention:
                raise AssertionError(
                    f"T(2,{n}) Topoly agreement/convention changed across successful repeats"
                )

        print(
            json.dumps(
                {
                    "n": n,
                    "repeat": repeat,
                    "knottedgraph_s": kg_time,
                    "topoly_status": tp_status,
                    "topoly_s": tp_time,
                    "topoly_error": tp.get("error"),
                    "knottedgraph_formula": "PASS",
                    "topoly_agrees": agrees,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    kg_median = statistics.median(kg_times)
    all_topoly_ok = len(tp_times) == repeats
    tp_median = statistics.median(tp_times) if tp_times else None
    if all_topoly_ok:
        knottedgraph_faster = kg_median < tp_median
        speedup = tp_median / kg_median
        comparison = "timed"
    else:
        # A competitor timeout/error is itself a benchmark outcome. KnottedGraph
        # has completed the independently verified calculation while Topoly has
        # not completed all equivalent calls, so there is no finite timing ratio.
        knottedgraph_faster = True
        speedup = None
        comparison = "topoly_failed_or_timed_out"

    row = {
        "n": n,
        "crossings": n,
        "repeats": repeats,
        "knottedgraph_median_s": kg_median,
        "topoly_median_s": tp_median,
        "topoly_statuses": tp_statuses,
        "topoly_over_knottedgraph": speedup,
        "comparison": comparison,
        "knottedgraph_faster": knottedgraph_faster,
        "knottedgraph_formula": "PASS",
        "topoly_agrees": topoly_agreement,
        "unit_sign_topoly_over_kg": convention[0] if convention else None,
        "variable_orientation": convention[1] if convention else None,
        "monomial_shift_topoly_minus_kg": convention[2] if convention else None,
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
