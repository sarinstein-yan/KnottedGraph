from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from knotted_graph.invariants.yamada.diagram_frontier import _greedy_factor_order
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "User_guide" / "benchmarks" / "03_knottedgraph_vs_topoly_scaling_final_push.ipynb"


def constructors():
    notebook = json.loads(NOTEBOOK.read_text())
    ns = {}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if all(name in source for name in (
            "def reference_dv_theta_graph",
            "def reference_lllv_cycle_graph",
            "def reference_lllv_theta_graph",
        )):
            exec(compile(source, str(NOTEBOOK), "exec"), ns)
            return {
                "THETA_N_VALUES": ns["reference_dv_theta_graph"],
                "LLLV_CYCLE_N_VALUES": ns["reference_lllv_cycle_graph"],
                "LLLV_THETA_S_VALUES": ns["reference_lllv_theta_graph"],
            }
    raise RuntimeError("constructors not found")


def prepared(family, n):
    graph = constructors()[family](n)
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    y = Yamada.from_PDCode(pd)
    return PreparedCompactStateBuilder.prepare(
        y.vertices, y.crossings, y.arcs, _ordered_crossing_ports
    )


def raw_tables(state):
    nv = len(state.vertex_ids)
    nc = len(state.crossing_ids)
    factor_ports = [[] for _ in range(nv + nc)]
    port_factor = [-1] * len(state.arc_partner)
    for p in range(len(state.arc_partner)):
        f = state.fixed_terminal_index[p]
        if f < 0:
            f = nv + state.crossing_for_port[p]
        factor_ports[f].append(p)
        port_factor[p] = f
    return factor_ports, port_factor, list(state.arc_partner)


def factorized_tables(state):
    """Split every fixed equality tensor into an arity<=3 equality chain."""
    nv = len(state.vertex_ids)
    nc = len(state.crossing_ids)
    original_ports = len(state.arc_partner)
    vertex_ports = [[] for _ in range(nv)]
    crossing_ports = [[] for _ in range(nc)]
    for p in range(original_ports):
        v = state.fixed_terminal_index[p]
        if v >= 0:
            vertex_ports[v].append(p)
        else:
            crossing_ports[state.crossing_for_port[p]].append(p)

    def neighbor_key(p):
        q = state.arc_partner[p]
        v = state.fixed_terminal_index[q]
        if v >= 0:
            return (0, v, q)
        return (1, state.crossing_for_port[q], q)

    factors = []
    port_factor = [-1] * original_ports
    arc_partner = list(state.arc_partner)

    for ports in vertex_ports:
        ports = sorted(ports, key=neighbor_key)
        previous_right = None
        for i, p in enumerate(ports):
            factor = len(factors)
            local = [p]
            port_factor[p] = factor
            if previous_right is not None:
                left = len(arc_partner)
                arc_partner.append(previous_right)
                arc_partner[previous_right] = left
                port_factor.append(factor)
                local.append(left)
            if i + 1 < len(ports):
                right = len(arc_partner)
                arc_partner.append(-1)
                port_factor.append(factor)
                local.append(right)
                previous_right = right
            else:
                previous_right = None
            factors.append(local)

    for ports in crossing_ports:
        factor = len(factors)
        local = sorted(ports)
        for p in local:
            port_factor[p] = factor
        factors.append(local)

    if any(f < 0 for f in port_factor):
        raise RuntimeError("factorized port has no owner")
    if any(q < 0 for q in arc_partner):
        raise RuntimeError("unpaired virtual port")
    return factors, port_factor, arc_partner


def plan(factor_ports, port_factor, arc_partner):
    adjacency = [defaultdict(int) for _ in factor_ports]
    for p, q in enumerate(arc_partner):
        if p >= q:
            continue
        a, b = port_factor[p], port_factor[q]
        if a != b:
            adjacency[a][b] += 1
            adjacency[b][a] += 1
    order = _greedy_factor_order(adjacency, factor_ports)
    active = []
    processed = set()
    peak = 0
    boundary = 0
    for factor in order:
        active.extend(sorted(factor_ports[factor]))
        peak = max(peak, len(active))
        processed.add(factor)
        active = [p for p in active if port_factor[arc_partner[p]] not in processed]
        boundary = max(boundary, len(active))
    if active:
        raise RuntimeError("planner did not close")
    return order, peak, boundary


def report(family, n):
    state = prepared(family, n)
    raw_fp, raw_pf, raw_ap = raw_tables(state)
    fac_fp, fac_pf, fac_ap = factorized_tables(state)
    raw_order, raw_peak, raw_boundary = plan(raw_fp, raw_pf, raw_ap)
    fac_order, fac_peak, fac_boundary = plan(fac_fp, fac_pf, fac_ap)
    print(json.dumps({
        "family": family,
        "n": n,
        "crossings": len(state.crossing_ids),
        "raw_factors": len(raw_fp),
        "raw_ports": len(raw_ap),
        "raw_peak_ports": raw_peak,
        "raw_boundary_ports": raw_boundary,
        "factorized_factors": len(fac_fp),
        "factorized_ports": len(fac_ap),
        "factorized_peak_ports": fac_peak,
        "factorized_boundary_ports": fac_boundary,
        "raw_order": raw_order,
        "factorized_order": fac_order,
    }, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("family")
    p.add_argument("n", type=int)
    args = p.parse_args()
    report(args.family, args.n)
