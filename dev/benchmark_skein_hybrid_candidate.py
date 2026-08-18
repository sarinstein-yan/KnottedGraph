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
from knotted_graph.invariants.yamada.streaming_candidate import ChunkedEvaluatorProxy
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
        base_evaluator = CompactYamadaEvaluator()
        evaluator = (
            ChunkedEvaluatorProxy(base_evaluator, chunk_size=256)
            if mode in {"chunked", "hybrid_chunked"}
            else base_evaluator
        )
        stats = {}
        start = time.perf_counter()
        if mode in {"bulk", "chunked"}:
            reduced, moves = prepared.reduce_reidemeister_ii()
            value = bulk_laurent(reduced, evaluator)
            stats = {
                "initial_rii_moves": moves,
                "remaining_crossings": len(reduced.crossing_ids),
            }
        elif mode in {"hybrid", "hybrid_chunked"}:
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
                "memo_size": getattr(base_evaluator, "memo_size", None),
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
    results = {
        mode: _run(mode, processor, timeout_s)
        for mode in ("bulk", "chunked", "hybrid", "hybrid_chunked")
    }

    successful = [result for result in results.values() if result["status"] == "ok"]
    if successful:
        reference = successful[0]["value"]
        for result in successful[1:]:
            if result["value"] != reference:
                raise AssertionError(
                    f"candidate changed exact Laurent output for {name}: "
                    f"{successful[0]['mode']}={reference} "
                    f"{result['mode']}={result['value']}"
                )

    row = {
        "case": name,
        "V": len(processor.vertices),
        "crossings": len(processor.crossings),
        "initial_rii_moves": initial_rii_moves,
        "remaining_crossings_after_initial_rii": len(reduced.crossing_ids),
        "correctness": "PASS" if len(successful) >= 2 else "not-enough-paired-results",
    }
    bulk_time = results["bulk"].get("time_s")
    for mode, result in results.items():
        row[f"{mode}_status"] = result["status"]
        row[f"{mode}_s"] = result.get("time_s")
        row[f"{mode}_stats"] = result.get("stats")
        row[f"{mode}_memo_size"] = result.get("memo_size")
        if bulk_time is not None and result.get("time_s") is not None:
            row[f"{mode}_speedup_vs_bulk"] = bulk_time / result["time_s"]

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
    timeout_s = 20.0
    rows = [_benchmark_processor("petersen_c6", _petersen_processor(), timeout_s)]
    for name, processor in _random_cubic_processors():
        rows.append(_benchmark_processor(name, processor, timeout_s))

    speedups = []
    for row in rows:
        for mode in ("chunked", "hybrid", "hybrid_chunked"):
            key = f"{mode}_speedup_vs_bulk"
            if key in row:
                speedups.append(row[key])
    print(
        "SUMMARY="
        + json.dumps(
            {
                "paired_speedups": len(speedups),
                "median_candidate_speedup_vs_bulk": (
                    statistics.median(speedups) if speedups else None
                ),
                "rows": rows,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
