from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np

# This driver reuses the exact same timing and Laurent-polynomial comparison
# machinery as benchmark_topoly_paper_scaling.py. Keep dev/ importable when
# the file is executed directly from the repository root.
DEV = Path(__file__).resolve().parent
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

import benchmark_topoly_paper_scaling as base  # noqa: E402
from knotted_graph.projection import PDCode  # noqa: E402

FAMILY = "essential_torus_constituent"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_CENSOR_FRONTIER = 2
DEFAULT_N_VALUES = (3, 5, 9, 17, 33)


def _validate_n(n: int) -> int:
    n = int(n)
    if n < 3 or n % 2 == 0:
        raise ValueError("T(2,n) benchmark requires odd n >= 3")
    return n


def essential_torus_graph(n: int, samples_per_crossing: int = 18) -> nx.MultiGraph:
    """Return a theta graph whose e1+e2 cycle is the closure of sigma_1^n.

    For odd n >= 3, the closure of the two-strand braid sigma_1^n is the
    torus knot T(2,n) (or its mirror if the projection orientation is reversed).
    Its crossing number is exactly n. The third theta edge is routed outside
    the braid diagram and therefore does not create extra projected crossings.

    The graph itself has two degree-3 vertices and three edges. The constituent
    cycle formed by roles ``torus_a`` and ``torus_b`` supplies the topological
    crossing-number certificate used by this benchmark.
    """
    n = _validate_n(n)
    samples_per_crossing = int(samples_per_crossing)
    if samples_per_crossing < 6:
        raise ValueError("samples_per_crossing must be >= 6")

    # A geometric realization of the two-strand braid sigma_1^n. As x grows,
    # the two points execute n same-handed half-turns in the yz plane. In the
    # xy projection their y coordinates cross exactly n times, while z fixes
    # the over/under information.
    point_count = n * samples_per_crossing + 1
    s = np.linspace(0.0, 1.0, point_count)
    x = np.linspace(-2.5, 2.5, point_count)
    phase = n * np.pi * s
    y = np.cos(phase)
    z = 0.55 * np.sin(phase)
    braid_a = np.column_stack([x, y, z])
    braid_b = np.column_stack([x, -y, -z])

    # For odd n, braid_a runs top-left -> bottom-right and braid_b runs
    # bottom-left -> top-right. Closing equal-height endpoints outside the
    # braid gives the standard closure of sigma_1^n. Split the two closure arcs
    # at U and V so the closed knot becomes two U--V edges of a theta graph.
    U = np.array([0.0, 3.0, 0.0])
    V = np.array([0.0, -3.0, 0.0])
    left_x = -3.5
    right_x = 3.5

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

    # A third U--V edge routed wholly outside the closed-braid rectangle.
    # Its xy projection is disjoint from the torus-knot cycle except at U,V.
    edge_c = np.asarray(
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
    graph.add_edge("u", "v", pts=edge_a, role="torus_a")
    graph.add_edge("u", "v", pts=edge_b, role="torus_b")
    graph.add_edge("u", "v", pts=edge_c, role="outer_theta_edge")

    degrees = dict(graph.degree())
    if graph.number_of_nodes() != 2 or graph.number_of_edges() != 3:
        raise AssertionError("essential torus benchmark must be a theta graph V=2,E=3")
    if any(degree != 3 for degree in degrees.values()):
        raise AssertionError(f"essential torus graph is not trivalent: {degrees}")
    return graph


def prepare_essential_torus(n: int):
    """Construct the graph and certify that the supplied projection is minimal."""
    n = _validate_n(n)
    graph = essential_torus_graph(n)
    processor = PDCode(graph)
    pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    projected = len(processor.crossings)

    # The e1+e2 constituent is T(2,n), whose crossing number is n. The chosen
    # standard braid projection has exactly n crossings, so it attains that
    # lower bound and is crossing-minimal rather than merely a complicated
    # Reidemeister-reducible diagram.
    if projected != n:
        raise AssertionError(
            f"T(2,{n}) theta projection should have exactly {n} crossings; "
            f"PDCode detected {projected}"
        )
    return graph, processor, pdcode


def essential_torus_row(
    n: int,
    timeout_s: float,
    active: dict[str, bool],
) -> dict:
    graph, processor, pdcode = prepare_essential_torus(n)
    row = {
        "family": FAMILY,
        "sample_kind": "certified_topological_crossing_number",
        "size": int(n),
        "sample": 0,
        "constituent_knot": f"T(2,{n})",
        "braid_word": f"sigma_1^{n}",
        "certified_min_crossings": int(n),
        "crossings": len(processor.crossings),
        "crossing_minimal_diagram": len(processor.crossings) == int(n),
        "certificate": "constituent cycle is closure of sigma_1^n = T(2,n)",
        "embedding_hash": base._embedding_hash(graph),
        "pd_hash": hashlib.sha256(pdcode.encode()).hexdigest(),
        "pd_length": len(pdcode),
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "connected": True,
        "regular_degree": 3,
        "timeout_s": float(timeout_s),
    }
    return base._finalize_row(
        row,
        base._evaluate_pair(processor, pdcode, timeout_s, active),
    )


def parse_n_values(text: str) -> list[int]:
    values = [_validate_n(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one odd n >= 3 is required")
    if values != sorted(set(values)):
        raise ValueError("n values must be unique and strictly increasing")
    return values


def main(timeout_s: float, n_values: list[int], censor_frontier: int) -> None:
    if censor_frontier < 1:
        raise ValueError("censor_frontier must be >= 1")

    active = {"knottedgraph": True, "topoly": True}
    consecutive = {"knottedgraph": 0, "topoly": 0}
    rows: list[dict] = []

    print(
        "CONFIG="
        + json.dumps(
            {
                "family": FAMILY,
                "n_values": n_values,
                "timeout_s": timeout_s,
                "censor_frontier": censor_frontier,
                "topological_certificate": "cr(T(2,n)) = n for odd n >= 3",
                "timed_repeats_per_sample": 1,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    print(f"FAMILY={FAMILY}", flush=True)

    for n in n_values:
        row = essential_torus_row(n, timeout_s, active)
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")), flush=True)
        base._update_censoring(
            FAMILY,
            n,
            [row],
            active,
            consecutive,
            censor_frontier,
        )
        if not any(active.values()):
            break

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--n-values",
        default=",".join(map(str, DEFAULT_N_VALUES)),
        help="comma-separated odd n values >= 3; default: 3,5,9,17,33",
    )
    parser.add_argument(
        "--censor-frontier",
        type=int,
        default=DEFAULT_CENSOR_FRONTIER,
    )
    args = parser.parse_args()
    main(
        timeout_s=args.timeout,
        n_values=parse_n_values(args.n_values),
        censor_frontier=args.censor_frontier,
    )
