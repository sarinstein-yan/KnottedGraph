from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp
import topoly
from topoly.params import Closure

from knotted_graph.projection import compute_yamada_polynomial

import discover_yamada_theta_collisions as core

A = sp.Symbol("A")
TARGETS = [
    ("pair6_left", 32, 58, 0.12),
    ("pair6_right", 39, 153, 0.05),
    ("pair9_left", 32, 197, 0.12),
    ("pair9_right", 39, 102, 0.05),
]


def cycle_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.vstack([a, b[-2:0:-1]])


def invariant(value):
    if isinstance(value, dict):
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    return str(value)


def constituent_data(edge_points: list[np.ndarray]) -> list[dict]:
    out = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cycle = cycle_points(edge_points[i], edge_points[j])
        kwargs = dict(
            closure=Closure.CLOSED,
            chiral=True,
            run_parallel=False,
        )
        out.append(
            {
                "edge_pair": [i, j],
                "homfly": invariant(topoly.homfly(cycle.tolist(), **kwargs)),
                "jones": invariant(topoly.jones(cycle.tolist(), **kwargs)),
            }
        )
    return out


def two_terminal_subgraph(graph: nx.MultiGraph, keep_roles: tuple[int, int]) -> nx.MultiGraph:
    sub = nx.MultiGraph()
    sub.add_node("u", pos=np.asarray(graph.nodes["u"]["pos"], dtype=float))
    sub.add_node("v", pos=np.asarray(graph.nodes["v"]["pos"], dtype=float))
    for _, _, data in graph.edges(data=True):
        if int(data["role"]) in keep_roles:
            sub.add_edge(
                "u",
                "v",
                role=int(data["role"]),
                pts=np.asarray(data["pts"], dtype=float),
            )
    if sub.number_of_edges() != 2:
        raise AssertionError("expected a two-edge two-terminal subgraph")
    return sub


def yamada(graph: nx.MultiGraph, *, normalize: bool) -> str:
    result = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=normalize,
        n_jobs=1,
        crossing_warning_threshold=None,
        return_result=True,
    )
    return str(sp.expand(result.polynomial))


def run(plantri: str, output: Path) -> dict:
    shadows = core.generate_shadows(plantri, 8)
    by_index = {shadow.index: shadow for shadow in shadows}
    records = []
    for label, shadow_index, bits, fraction in TARGETS:
        shadow = by_index[shadow_index]
        graph, edge_points = core.spatial_theta(shadow, bits, approach_fraction=fraction)
        traces = core.trace_theta_edges(shadow)
        direct_roles = [i for i, trace in enumerate(traces) if len(trace) == 2]
        if direct_roles != [2]:
            raise AssertionError(
                f"{label}: expected role 2 to be the direct edge, got {direct_roles}"
            )
        residual = two_terminal_subgraph(graph, (0, 1))
        records.append(
            {
                "label": label,
                "shadow": shadow_index,
                "bits": bits,
                "bitstring": format(bits, "08b"),
                "constituents": constituent_data(edge_points),
                "full_theta_yamada_normalized": yamada(graph, normalize=True),
                "full_theta_yamada_raw": yamada(graph, normalize=False),
                "two_terminal_residual_yamada_normalized": yamada(
                    residual, normalize=True
                ),
                "two_terminal_residual_yamada_raw": yamada(
                    residual, normalize=False
                ),
                "direct_edge_role": 2,
            }
        )

    payload = {"targets": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("STRUCTURE=" + json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output)


if __name__ == "__main__":
    main()
