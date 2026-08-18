from __future__ import annotations

import json
import multiprocessing as mp
import statistics
import time

import networkx as nx
import numpy as np

from benchmark_topoly_random_cubic_ensemble import (
    DEFAULT_SEED,
    prepare_sample,
    topology_ensemble,
)
from knotted_graph.invariants.yamada.compact import CompactYamadaEvaluator
from knotted_graph.invariants.yamada.polynomial import Yamada, _ordered_crossing_ports
from knotted_graph.invariants.yamada.skein_hybrid import (
    bulk_laurent,
    compute_hybrid_laurent,
)
from knotted_graph.invariants.yamada.state_compact import PreparedCompactStateBuilder
from knotted_graph.projection import PDCode


def _spring_embedding(graph: nx.Graph, seed: int) -> nx.MultiGraph:
    positions = nx.spring_layout(graph, dim=3, seed=seed, scale=3.0)
    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(node, pos=np.asarray(point, dtype=float))
    for u, v in graph.edges():
        embedded.add_edge(u, v, pts=np.vstack([positions[u], positions[v]]))
    return embedded


def _prepared_from_processor(processor: PDCode):
    calculator = Yamada.from_PDCode(processor)
    prepared = PreparedCompactStateBuilder.prepare(
        calculator.vertices,
        calculator.crossings,
        calculator.arcs,
        _ordered_crossing_ports,
    )
    reduced, moves = prepared.reduce_reidemeister_ii()
    return prepared, reduced, moves


def _worker(mode, vertices, crossings, arcs, queue):
    try:
        calculator = Yamada(list(vertices), list(crossings), list(arcs))
        prepared = PreparedCompactStateBuilder.prepare(
            calculator.vertices,
            calculator.crossings,
            calculator.arcs,
            _ordered_crossing_ports,
        )
        evaluator = CompactYamadaEvaluator()
        stats = {}
        start = time.perf_counter()
        if mode == "bulk":
            reduced, moves = prepared.reduce_reidemeister_ii()
            value = bulk_laurent(reduced, evaluator)
            stats = {
                "initial_rii_moves": moves,
                "remaining_crossings": len(reduced.crossing_ids),
            }
        elif mode == "hybrid":
            value = compute_hybrid_laurent(prepared, evaluator, stats=stats)
        else:
            raise ValueError(mode)
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "ok",
                "mode": mode,
                "time_s": elapsed,
                "value": value,
                "stats": stats,
            }
        )
    except BaseException as exc:  # pragma: no cover - benchmark diagnostics
        queue.put(
            {
                "status": "error",
                "mode": mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run(mode, processor, timeout_s):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            mode,
            list(processor.vertices.values()),
            list(processor.crossings.values()),
            list(processor.arcs.values()),
            queue,
        ),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {"status": "timeout", "mode": mode, "timeout_s": timeout_s}
    if not queue.empty():
        return queue.get()
    return {
        "status": "error",
        "mode": mode,
        "error": f"worker exited with code {process.exitcode} without returning data",
    }


def _benchmark_processor(name, processor, timeout_s):
    _, reduced, initial_rii_moves = _prepared_from_processor(processor)
    bulk = _run("bulk", processor, timeout_s)
    hybrid = _run("hybrid", processor, timeout_s)
    row = {
        "case": name,
        "V": len(processor.vertices),
        "E": len(processor.arcs) - 2 * len(processor.crossings),
        "crossings": len(processor.crossings),
        "initial_rii_moves": initial_rii_moves,
        "remaining_crossings_after_initial_rii": len(reduced.crossing_ids),
        "bulk_status": bulk["status"],
        "hybrid_status": hybrid["status"],
        "bulk_s": bulk.get("time_s"),
        "hybrid_s": hybrid.get("time_s"),
        "hybrid_stats": hybrid.get("stats"),
    }
    if bulk["status"] == "ok" and hybrid["status"] == "ok":
        if bulk["value"] != hybrid["value"]:
            raise AssertionError(
                f"hybrid changed exact Laurent output for {name}: "
                f"bulk={bulk['value']} hybrid={hybrid['value']}"
            )
        row["speedup"] = bulk["time_s"] / hybrid["time_s"]
        row["correctness"] = "PASS"
    else:
        row["correctness"] = "not-paired-after-timeout-or-error"
    print(json.dumps(row, separators=(",", ":")))
    return row


def _petersen_processor():
    embedded = _spring_embedding(nx.petersen_graph(), 9)
    processor = PDCode(embedded)
    processor.compute(
        rotation_angles=(-134.58074129795634, 55.40942502382338, 0.0)
    )
    if len(processor.crossings) != 6:
        raise AssertionError(f"expected Petersen c=6, got {len(processor.crossings)}")
    return processor


def _random_cubic_processors():
    for vertex_count in (10, 14, 20):
        ensemble = topology_ensemble(vertex_count, 2, DEFAULT_SEED)
        for sample, abstract in ensemble:
            _, processor, _, embedding_attempt = prepare_sample(
                sample,
                abstract,
                DEFAULT_SEED,
            )
            yield (
                f"random_cubic_V{vertex_count}_s{sample.sample_index}_e{embedding_attempt}",
                processor,
            )


def main():
    timeout_s = 30.0
    rows = [_benchmark_processor("petersen_c6", _petersen_processor(), timeout_s)]
    for name, processor in _random_cubic_processors():
        rows.append(_benchmark_processor(name, processor, timeout_s))

    paired = [row for row in rows if row.get("correctness") == "PASS"]
    if not paired:
        raise AssertionError("hybrid benchmark produced no paired exact comparisons")
    print(
        "SUMMARY="
        + json.dumps(
            {
                "paired": len(paired),
                "median_speedup": statistics.median(row["speedup"] for row in paired),
                "rows": rows,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
