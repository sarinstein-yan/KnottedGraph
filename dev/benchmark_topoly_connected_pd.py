from __future__ import annotations

import json
import statistics
import time

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode, generate_isotopy_angles

A = sp.Symbol("A")


def spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def kg_terms(poly: sp.Expr) -> dict[int, int]:
    out = {}
    for term in sp.expand(poly).as_ordered_terms():
        coeff, exponent = term.as_coeff_exponent(A)
        out[int(exponent)] = out.get(int(exponent), 0) + int(coeff)
    return {k: v for k, v in out.items() if v}


def topoly_terms(poly) -> dict[int, int]:
    out = {}
    for term in poly.term:
        degree = getattr(term, "degree", {})
        exponent = int(next(iter(degree.values()))) if degree else 0
        out[exponent] = out.get(exponent, 0) + int(term.coef)
    return {k: v for k, v in out.items() if v}


def sequence(terms: dict[int, int]) -> list[int]:
    if not terms:
        return [0]
    return [terms.get(i, 0) for i in range(min(terms), max(terms) + 1)]


def validate(kg, topoly):
    kg_t = kg_terms(kg)
    tp_t = topoly_terms(topoly)
    if sequence(kg_t) != sequence(tp_t):
        raise AssertionError(
            "Topoly and KnottedGraph differ beyond monomial normalization: "
            f"KG={sequence(kg_t)}, Topoly={sequence(tp_t)}"
        )
    return min(tp_t) - min(kg_t) if kg_t and tp_t else 0


def median_time(fn, repeats):
    values = []
    answer = None
    for _ in range(repeats):
        start = time.perf_counter()
        answer = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), answer


def select_views(embedded, *, max_views=96, max_crossings=8):
    by_crossings = {}
    for angles in generate_isotopy_angles(max_views):
        rotation = tuple(float(x) for x in angles)
        processor = PDCode(embedded)
        try:
            processor.compute(rotation_angles=rotation, rotation_order="ZYX")
        except (ValueError, RuntimeError):
            continue
        c = len(processor.crossings)
        if 1 <= c <= max_crossings and c not in by_crossings:
            calculator = Yamada.from_PDCode(processor)
            if len(calculator._diagram_blocks()) != 1:
                raise AssertionError("connected benchmark unexpectedly factorized")
            by_crossings[c] = (rotation, processor)
    return [by_crossings[c] for c in sorted(by_crossings)]


def main():
    from topoly.invariants import Invariant, YamadaGraph

    abstract_cases = [
        ("K4", nx.complete_graph(4), 7),
        ("K3_3", nx.complete_bipartite_graph(3, 3), 1),
        ("cube", nx.cubical_graph(), 5),
        ("petersen", nx.petersen_graph(), 9),
        ("cubic8", nx.random_regular_graph(3, 8, seed=11), 11),
    ]

    rows = []
    seen_pd = set()
    for name, graph, seed in abstract_cases:
        if not nx.is_connected(graph):
            raise AssertionError(f"{name} must be connected")
        embedded = spring_embedding(graph, seed)
        for rotation, processor in select_views(embedded):
            pdcode = processor._generate_pd_code()
            if pdcode in seen_pd:
                continue
            seen_pd.add(pdcode)
            crossings = len(processor.crossings)
            calculator = Yamada.from_PDCode(processor)

            kg_check = calculator.compute(A, normalize=False, n_jobs=1, method="negami")
            Invariant.known["Yamada"] = {}
            tp_check = YamadaGraph(pdcode).point(max_cross=200)
            shift = validate(kg_check, tp_check)

            def run_kg():
                return Yamada.from_PDCode(processor).compute(
                    A, normalize=False, n_jobs=1, method="negami"
                )

            def run_topoly():
                Invariant.known["Yamada"] = {}
                return YamadaGraph(pdcode).point(max_cross=200)

            run_kg()
            run_topoly()
            Invariant.known["Yamada"] = {}

            repeats = 5 if crossings <= 4 else 3
            kg_time, kg_answer = median_time(run_kg, repeats)
            tp_time, tp_answer = median_time(run_topoly, repeats)
            if validate(kg_answer, tp_answer) != shift:
                raise AssertionError("normalization shift changed between runs")

            row = {
                "graph": name,
                "crossings": crossings,
                "rotation": list(rotation),
                "pd_length": len(pdcode),
                "monomial_shift_topoly_minus_kg": shift,
                "knottedgraph_s": kg_time,
                "topoly_s": tp_time,
                "kg_over_topoly": kg_time / tp_time,
                "topoly_over_kg": tp_time / kg_time,
                "coefficient_count": len(sequence(kg_terms(kg_answer))),
            }
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")))

    if not rows:
        raise AssertionError("No connected crossing-containing benchmark views found")
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
