from __future__ import annotations

import argparse
import json
from pathlib import Path

from knotted_graph.invariants.yamada.frontier_ordering import greedy_factor_order
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
    nf = nv + nc
    factor_ports = [[] for _ in range(nf)]
    port_factor = [-1] * len(state.arc_partner)
    for p in range(len(state.arc_partner)):
        f = state.fixed_terminal_index[p]
        if f < 0:
            f = nv + state.crossing_for_port[p]
        factor_ports[f].append(p)
        port_factor[p] = f
    return factor_ports, port_factor, list(state.arc_partner)


def factorized_tables(state):
    """Split each fixed equality tensor into a chain of arity <=3 tensors."""
    nv = len(state.vertex_ids)
    nc = len(state.crossing_ids)
    original_ports = len(state.arc_partner)

    # Group original physical ports by fixed vertex and crossing.
    vertex_ports = [[] for _ in range(nv)]
    crossing_ports = [[] for _ in range(nc)]
    for p in range(original_ports):
        v = state.fixed_terminal_index[p]
        if v >= 0:
            vertex_ports[v].append(p)
        else:
            crossing_ports[state.crossing_for_port[p]].append(p)

    # Sort each equality chain by the neighboring original factor. This is a
    # topology-only canonical choice; no benchmark/runtime information enters.
    def neighbor_key(p):
        q = state.arc_partner[p]
        v = state.fixed_terminal_index[q]
        if v >= 0:
            return (0, v, q)
        return (1, state.crossing_for_port[q], q)

    factors = []
    port_factor = [-1] * original_ports
    arc_partner = list(state.arc_partner)

    # Each chain segment owns one original port and at most two virtual ports.
    for v, ports in enumerate(vertex_ports):
        ports = sorted(ports, key=neighbor_key)
        if not ports:
            continue
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

    # Crossings remain four-port factors.
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


def report(family, n):
    state = prepared(family, n)
    raw_fp, raw_pf, raw_ap = raw_tables(state)
    fac_fp, fac_pf, fac_ap = factorized_tables(state)
    raw_order, raw_peak = greedy_factor_order(raw_fp, raw_pf, raw_ap)
    fac_order, fac_peak = greedy_factor_order(fac_fp, fac_pf, fac_ap)
    print(json.dumps({
        "family": family,
        "n": n,
        "crossings": len(state.crossing_ids),
        "raw_factors": len(raw_fp),
        "raw_ports": len(raw_ap),
        "raw_peak_ports": raw_peak,
        "factorized_factors": len(fac_fp),
        "factorized_ports": len(fac_ap),
        "factorized_peak_ports": fac_peak,
        "raw_order": raw_order,
        "factorized_order": fac_order,
    }, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("family")
    p.add_argument("n", type=int)
    args = p.parse_args()
    report(args.family, args.n)
