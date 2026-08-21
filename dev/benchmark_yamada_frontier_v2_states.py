from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from benchmark_yamada_frontier_v2_public_ab import _prepare


def measure(case: str, repeats: int = 3):
    from knotted_graph.invariants.yamada.diagram_frontier import (
        compute_diagram_frontier_laurent,
        plan_diagram_frontier,
    )
    from knotted_graph.invariants.yamada.polynomial import Yamada

    processor, _pd = _prepare(case)
    calculator = Yamada(
        list(processor.vertices.values()),
        list(processor.crossings.values()),
        list(processor.arcs.values()),
    )
    prepared = calculator._prepare_compact_state_builder()
    plan = plan_diagram_frontier(prepared)
    rows = []
    values = []
    for _ in range(repeats):
        stats = {}
        start = time.perf_counter()
        value = compute_diagram_frontier_laurent(
            prepared,
            factor_order=plan["factor_order"],
            stats=stats,
        )
        elapsed = time.perf_counter() - start
        rows.append({"time_s": elapsed, "stats": stats})
        values.append(value)
    assert all(value == values[0] for value in values)
    middle = sorted(rows, key=lambda row: row["time_s"])[len(rows) // 2]
    return {
        "median_s": statistics.median(row["time_s"] for row in rows),
        "times_s": [row["time_s"] for row in rows],
        "stats": middle["stats"],
        "value": [[int(p), int(c)] for p, c in values[0]],
    }


def run(label: str, output: str):
    data = {"label": label, "cases": {case: measure(case) for case in ("cycle10", "cycle11")}}
    Path(output).write_text(json.dumps(data, indent=2, sort_keys=True))
    print(json.dumps(data, sort_keys=True))


def compare(base: str, candidate: str):
    b = json.loads(Path(base).read_text())
    c = json.loads(Path(candidate).read_text())
    for case in b["cases"]:
        old = b["cases"][case]
        new = c["cases"][case]
        assert old["value"] == new["value"]
        old_states = int(old["stats"]["max_states"])
        new_states = int(new["stats"]["max_states"])
        old_transitions = int(old["stats"]["transitions"])
        new_transitions = int(new["stats"]["transitions"])
        print(
            f"{case} STATES={old_states}->{new_states} "
            f"STATE_REDUCTION={(1-new_states/old_states)*100:.2f}% "
            f"TRANSITIONS={old_transitions}->{new_transitions} "
            f"TRANSITION_REDUCTION={(1-new_transitions/old_transitions)*100:.2f}% "
            f"DIRECT_SPEEDUP={old['median_s']/new['median_s']:.6f}x EXACT=PASS"
        )
        print(f"  base_stats={old['stats']}")
        print(f"  candidate_stats={new['stats']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("output")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("candidate")
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.label, args.output)
    else:
        compare(args.base, args.candidate)
