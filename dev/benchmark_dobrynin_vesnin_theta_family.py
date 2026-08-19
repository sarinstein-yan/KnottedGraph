from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np

DEV = Path(__file__).resolve().parent
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

import benchmark_topoly_paper_scaling as base  # noqa: E402
from knotted_graph.projection import PDCode  # noqa: E402


PAPER = (
    "A.A. Dobrynin and A.Yu. Vesnin, The Yamada polynomial for graphs, "
    "embedded knot-wise into three-dimensional space, Vychisl. Sistemy 155 "
    "(1996) 37-86, Theorem 2"
)
FORMULA = (
    "R(Theta(n))(A)=(A^2+A+1+A^-1+A^-2)A^n"
    "-(A+A^-1)A^(-2n)"
    "-(A^2+1+A^-2)(-1)^n A^(-n)"
)


def validate_n(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError("Dobrynin-Vesnin Theta(n) requires n >= 0")
    return n


def published_theta_terms(n: int) -> dict[int, int]:
    """Independent literal expansion of Dobrynin--Vesnin Theorem 2."""
    n = validate_n(n)
    terms: dict[int, int] = {}

    def add(power: int, coefficient: int) -> None:
        terms[power] = terms.get(power, 0) + coefficient
        if not terms[power]:
            del terms[power]

    for offset in (2, 1, 0, -1, -2):
        add(n + offset, 1)
    add(-2 * n + 1, -1)
    add(-2 * n - 1, -1)
    third = -((-1) ** n)
    for offset in (2, 0, -2):
        add(-n + offset, third)
    return terms


def _braid(n: int, samples_per_crossing: int):
    point_count = max(2, n * samples_per_crossing + 1)
    s = np.linspace(0.0, 1.0, point_count)
    x = np.linspace(-2.5, 2.5, point_count)
    phase = n * np.pi * s
    y = np.cos(phase)
    z = 0.55 * np.sin(phase)
    return np.column_stack([x, y, z]), np.column_stack([x, -y, -z])


def theta_family_graph(n: int, samples_per_crossing: int = 18) -> nx.MultiGraph:
    """Geometric Dobrynin--Vesnin Theta(n) diagram for every n >= 0.

    Theorem 2 defines Theta(n) from the canonical two-strand torus diagram plus
    one exterior arc. For odd n the abstract graph is a theta graph. For even n
    it is the pince-nez/handcuff graph: two loop edges joined by one bridge.
    """
    n = validate_n(n)
    samples_per_crossing = int(samples_per_crossing)
    if samples_per_crossing < 6:
        raise ValueError("samples_per_crossing must be >= 6")

    braid_a, braid_b = _braid(n, samples_per_crossing)
    U = np.array([0.0, 3.0, 0.0])
    V = np.array([0.0, -3.0, 0.0])
    left_x = -3.5
    right_x = 3.5

    outer = np.asarray(
        [
            U,
            [0.0, 4.5, 0.0],
            [-5.0, 4.5, 0.0],
            [-5.0, -4.5, 0.0],
            [0.0, -4.5, 0.0],
            V,
        ],
        dtype=float,
    )

    graph = nx.MultiGraph()
    graph.add_node("u", pos=U.copy())
    graph.add_node("v", pos=V.copy())

    if n % 2:
        # The two braid strands exchange endpoints. Each becomes one of the
        # three U--V theta edges; the second is traversed in reverse so both
        # graph edges are oriented U -> V in the stored polyline.
        edge_a = np.vstack(
            [
                U,
                [left_x, 3.0, 0.0],
                [left_x, 1.0, 0.0],
                braid_a[0],
                braid_a[1:-1],
                braid_a[-1],
                [right_x, -1.0, 0.0],
                [right_x, -3.0, 0.0],
                V,
            ]
        )
        edge_b = np.vstack(
            [
                U,
                [right_x, 3.0, 0.0],
                [right_x, 1.0, 0.0],
                braid_b[-1],
                braid_b[-2:0:-1],
                braid_b[0],
                [left_x, -1.0, 0.0],
                [left_x, -3.0, 0.0],
                V,
            ]
        )
        graph.add_edge("u", "v", pts=edge_a, role="torus_a")
        graph.add_edge("u", "v", pts=edge_b, role="torus_b")
        graph.add_edge("u", "v", pts=outer, role="exterior_arc")
        expected_abstract = "theta"
    else:
        # The two braid strands return to the same side of the closure. They
        # therefore form the two loops of the handcuff/pince-nez graph, with the
        # added exterior arc as the U--V bridge.
        loop_u = np.vstack(
            [
                U,
                [left_x, 3.0, 0.0],
                [left_x, 1.0, 0.0],
                braid_a[0],
                braid_a[1:-1],
                braid_a[-1],
                [right_x, 1.0, 0.0],
                [right_x, 3.0, 0.0],
                U,
            ]
        )
        loop_v = np.vstack(
            [
                V,
                [left_x, -3.0, 0.0],
                [left_x, -1.0, 0.0],
                braid_b[0],
                braid_b[1:-1],
                braid_b[-1],
                [right_x, -1.0, 0.0],
                [right_x, -3.0, 0.0],
                V,
            ]
        )
        graph.add_edge("u", "u", pts=loop_u, role="torus_component_u")
        graph.add_edge("v", "v", pts=loop_v, role="torus_component_v")
        graph.add_edge("u", "v", pts=outer, role="exterior_arc")
        expected_abstract = "handcuff"

    if graph.number_of_nodes() != 2 or graph.number_of_edges() != 3:
        raise AssertionError("Theta(n) benchmark must have V=2,E=3")
    degrees = dict(graph.degree())
    if any(degree != 3 for degree in degrees.values()):
        raise AssertionError(f"Theta({n}) graph is not trivalent: {degrees}")
    graph.graph["dobrynin_vesnin_abstract_type"] = expected_abstract
    return graph


def prepare_theta_family(n: int):
    n = validate_n(n)
    graph = theta_family_graph(n)
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    projected = len(processor.crossings)
    if projected != n:
        raise AssertionError(
            f"Dobrynin-Vesnin Theta({n}) canonical projection should have "
            f"exactly {n} crossings; PDCode detected {projected}"
        )
    return graph, processor, pdcode


def _run(framework: str, processor, pdcode: str, timeout_s: float):
    return base._run_with_timeout(framework, processor, pdcode, timeout_s)


def compare_one(n: int, timeout_s: float) -> dict:
    graph, processor, pdcode = prepare_theta_family(n)
    expected = published_theta_terms(n)
    kg = _run("knottedgraph", processor, pdcode, timeout_s)
    tp = _run("topoly", processor, pdcode, timeout_s)

    row = {
        "n": n,
        "crossings": len(processor.crossings),
        "abstract_graph": graph.graph["dobrynin_vesnin_abstract_type"],
        "paper": PAPER,
        "theorem": "Theorem 2",
        "formula": FORMULA,
        "knottedgraph_status": kg["status"],
        "topoly_status": tp["status"],
        "knottedgraph_s": kg.get("time_s"),
        "topoly_s": tp.get("time_s"),
    }

    if kg["status"] != "ok":
        raise AssertionError(f"KnottedGraph failed for Theta({n}): {kg}")
    if kg["terms"] != expected:
        raise AssertionError(
            f"KnottedGraph disagrees with published Theta({n}) formula: "
            f"KG={kg['terms']}, published={expected}"
        )
    row["knottedgraph_vs_published"] = "PASS"

    if tp["status"] == "ok":
        try:
            sign, orientation, shift = base._validate_laurent_unit(
                expected, tp["terms"]
            )
        except AssertionError:
            row["topoly_vs_published"] = "FAIL"
        else:
            row["topoly_vs_published"] = "PASS"
            row["topoly_unit_sign"] = sign
            row["topoly_variable_orientation"] = orientation
            row["topoly_monomial_shift"] = shift
        row["topoly_over_knottedgraph"] = tp["time_s"] / kg["time_s"]
    else:
        row["topoly_vs_published"] = "ERROR"
        row["topoly_error"] = tp.get("error")

    print(json.dumps(row, separators=(",", ":")), flush=True)
    return row


def main(start: int, stop: int, timeout_s: float) -> None:
    if start < 0 or stop < start:
        raise ValueError("require 0 <= start <= stop")
    print("PAPER=" + PAPER, flush=True)
    print("THEOREM=Theorem 2", flush=True)
    print("FORMULA=" + FORMULA, flush=True)
    rows = [compare_one(n, timeout_s) for n in range(start, stop + 1)]
    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    main(args.start, args.stop, args.timeout)
