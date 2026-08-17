from __future__ import annotations

import json
import statistics
import time

import sympy as sp

from benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def _kg_terms(poly: sp.Expr) -> dict[int, int]:
    out: dict[int, int] = {}
    for term in sp.expand(poly).as_ordered_terms():
        coeff, exponent = term.as_coeff_exponent(A)
        out[int(exponent)] = out.get(int(exponent), 0) + int(coeff)
    return {k: v for k, v in out.items() if v}


def _topoly_terms(poly) -> dict[int, int]:
    out: dict[int, int] = {}
    for term in poly.term:
        degree = getattr(term, "degree", {})
        exponent = 0
        if degree:
            # Yamada is univariate. Topoly stores the variable under its own
            # internal key, so do not rely on that key being literally "x".
            exponent = int(next(iter(degree.values())))
        coeff = int(term.coef)
        out[exponent] = out.get(exponent, 0) + coeff
    return {k: v for k, v in out.items() if v}


def _coefficient_sequence(terms: dict[int, int]) -> list[int]:
    if not terms:
        return [0]
    lo = min(terms)
    hi = max(terms)
    return [terms.get(exp, 0) for exp in range(lo, hi + 1)]


def _validate_same_laurent_up_to_monomial(kg_terms, topoly_terms):
    kg_sequence = _coefficient_sequence(kg_terms)
    topoly_sequence = _coefficient_sequence(topoly_terms)
    if kg_sequence != topoly_sequence:
        raise AssertionError(
            "Topoly and KnottedGraph differ beyond a global Laurent monomial shift: "
            f"KG={kg_sequence}, Topoly={topoly_sequence}"
        )
    if not kg_terms and not topoly_terms:
        return 0
    return min(topoly_terms) - min(kg_terms)


def _median_time(fn, repeats: int):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def main():
    from topoly.invariants import Invariant, YamadaGraph

    rows = []
    for crossing_count in range(1, 8):
        spatial_graph = multi_crossing_theta(crossing_count)
        processor = PDCode(spatial_graph)
        pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        actual_crossings = len(processor.crossings)
        if actual_crossings != crossing_count:
            raise AssertionError(
                f"fixture requested c={crossing_count} but produced {actual_crossings} crossings"
            )

        # Correctness is checked from fresh, uncached evaluators first.
        kg_calculator = Yamada.from_PDCode(processor)
        kg_poly = kg_calculator.compute(A, normalize=False, n_jobs=1, method="negami")
        Invariant.known["Yamada"] = {}
        topoly_poly = YamadaGraph(pdcode).point(max_cross=200)
        shift = _validate_same_laurent_up_to_monomial(
            _kg_terms(kg_poly), _topoly_terms(topoly_poly)
        )

        def run_kg():
            return Yamada.from_PDCode(processor).compute(
                A, normalize=False, n_jobs=1, method="negami"
            )

        def run_topoly():
            # Topoly keeps a class-global memo dictionary. Clear it before every
            # run so repeated benchmark iterations are cold, matching the new
            # KnottedGraph evaluator constructed on every run.
            Invariant.known["Yamada"] = {}
            return YamadaGraph(pdcode).point(max_cross=200)

        # Warm imports and Python dispatch, but not invariant memo state.
        run_kg()
        run_topoly()
        Invariant.known["Yamada"] = {}

        repeats = 7 if crossing_count <= 4 else 3
        kg_time, kg_answer = _median_time(run_kg, repeats)
        topoly_time, topoly_answer = _median_time(run_topoly, repeats)

        final_shift = _validate_same_laurent_up_to_monomial(
            _kg_terms(kg_answer), _topoly_terms(topoly_answer)
        )
        if final_shift != shift:
            raise AssertionError("Topoly normalization shift changed across repetitions")

        row = {
            "crossings": crossing_count,
            "pd_length": len(pdcode),
            "monomial_shift_topoly_minus_kg": shift,
            "knottedgraph_s": kg_time,
            "topoly_s": topoly_time,
            "kg_over_topoly": kg_time / topoly_time,
            "topoly_over_kg": topoly_time / kg_time,
            "coefficient_count": len(_coefficient_sequence(_kg_terms(kg_answer))),
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
