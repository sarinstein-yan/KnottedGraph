from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time

import sympy as sp

from benchmark_topoly_extended_scaling import (
    Case,
    _embedding_hash,
    _kg_terms,
    _prepare,
    _seed_for,
    _sequence,
    _topoly_terms,
    _validate_laurent_unit,
)
from knotted_graph.invariants.yamada.polynomial import Yamada

A = sp.Symbol("A")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_EMBEDDINGS = 10
DEFAULT_SEED = 20260818

PAPER_FAMILIES = ("crossings_fixed", "vertices_k4")


def paper_cases(profile: str) -> dict[str, list[Case]]:
    """Return only the two benchmark families used in the paper figures."""
    if profile == "smoke":
        crossings = [1, 4, 8, 12]
        vertices = [4, 16, 64, 256]
    elif profile == "paper":
        crossings = [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
            24, 28, 32, 36, 40, 48, 56, 64, 80,
        ]
        vertices = [
            4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
            384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        ]
    else:
        raise ValueError(profile)

    return {
        "crossings_fixed": [Case("crossings_fixed", value) for value in crossings],
        "vertices_k4": [Case("vertices_k4", value) for value in vertices],
    }


def _run_once(framework: str, vertices, crossings, arcs, pdcode: str):
    if framework == "knottedgraph":
        answer = Yamada(
            list(vertices),
            list(crossings),
            list(arcs),
        ).compute(A, normalize=False, n_jobs=1, method="negami")
        return answer, _kg_terms(answer)

    if framework == "topoly":
        from topoly.invariants import Invariant, YamadaGraph

        Invariant.known["Yamada"] = {}
        answer = YamadaGraph(pdcode).point(max_cross=5000)
        return answer, _topoly_terms(answer)

    raise ValueError(framework)


def _worker(
    framework: str,
    vertices,
    crossings,
    arcs,
    pdcode: str,
    case: Case,
    embedding: int,
    seed: int,
    queue,
):
    """Time exactly one evaluation for one independent embedding.

    Statistical replication comes from the independent embeddings at each x
    value.  Keeping exactly one timed evaluation per embedding at every x avoids
    the old repeat-threshold discontinuities without duplicating expensive
    Yamada calculations.
    """
    try:
        start = time.perf_counter()
        _, terms = _run_once(framework, vertices, crossings, arcs, pdcode)
        elapsed = time.perf_counter() - start

        queue.put(
            {
                "status": "ok",
                "framework": framework,
                "time_s": elapsed,
                "timed_repeats": 1,
                "terms": terms,
                "embedding": embedding,
                "embedding_seed": seed,
            }
        )
    except BaseException as exc:  # pragma: no cover - benchmark diagnostics
        queue.put(
            {
                "status": "error",
                "framework": framework,
                "embedding": embedding,
                "embedding_seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_with_timeout(
    framework: str,
    case: Case,
    embedding: int,
    seed: int,
    timeout_s: float,
    processor,
    pdcode: str,
):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            framework,
            list(processor.vertices.values()),
            list(processor.crossings.values()),
            list(processor.arcs.values()),
            pdcode,
            case,
            embedding,
            seed,
            queue,
        ),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {
            "status": "timeout",
            "framework": framework,
            "embedding": embedding,
            "embedding_seed": seed,
            "timeout_s": timeout_s,
        }
    if not queue.empty():
        return queue.get()
    return {
        "status": "error",
        "framework": framework,
        "embedding": embedding,
        "embedding_seed": seed,
        "error": f"worker exited with code {process.exitcode} without returning data",
    }


def _row(
    case: Case,
    embedding: int,
    seed: int,
    timeout_s: float,
    active: dict[str, bool],
):
    graph, processor, pdcode = _prepare(case, seed)
    embedding_hash = _embedding_hash(graph)
    pd_hash = hashlib.sha256(pdcode.encode()).hexdigest()

    results = {}
    for framework in ("knottedgraph", "topoly"):
        if active[framework]:
            results[framework] = _run_with_timeout(
                framework,
                case,
                embedding,
                seed,
                timeout_s,
                processor,
                pdcode,
            )
        else:
            results[framework] = {
                "status": "skipped_after_censor_frontier",
                "framework": framework,
                "embedding": embedding,
                "embedding_seed": seed,
            }

    kg = results["knottedgraph"]
    tp = results["topoly"]
    row = {
        "family": case.family,
        "size": case.size,
        "sample_kind": "embedding",
        "embedding": embedding,
        "embedding_seed": seed,
        "embedding_hash": embedding_hash,
        "pd_hash": pd_hash,
        "pd_length": len(pdcode),
        "V": graph.number_of_nodes(),
        "E": graph.number_of_edges(),
        "crossings": len(processor.crossings),
        "timeout_s": timeout_s,
        "knottedgraph_status": kg["status"],
        "topoly_status": tp["status"],
        "knottedgraph_s": kg.get("time_s"),
        "topoly_s": tp.get("time_s"),
        "knottedgraph_repeats": kg.get("timed_repeats"),
        "topoly_repeats": tp.get("timed_repeats"),
    }

    for result in (kg, tp):
        if result["status"] == "error":
            row[f"{result['framework']}_error"] = result.get("error")

    if kg["status"] == "ok" and tp["status"] == "ok":
        sign, orientation, shift = _validate_laurent_unit(kg["terms"], tp["terms"])
        row.update(
            {
                "unit_sign_topoly_over_kg": sign,
                "variable_orientation": orientation,
                "monomial_shift_topoly_minus_kg": shift,
                "topoly_over_kg": tp["time_s"] / kg["time_s"],
                "kg_over_topoly": kg["time_s"] / tp["time_s"],
                "coefficient_count": len(_sequence(kg["terms"])),
                "correctness": "PASS",
            }
        )
    else:
        row["correctness"] = "not-evaluated-after-timeout-error-or-skip"
    return row


def main(
    timeout_s: float,
    profile: str,
    embeddings: int,
    base_seed: int,
    censor_frontier: int,
):
    if embeddings < 1:
        raise ValueError("embeddings must be >= 1")
    if censor_frontier < 1:
        raise ValueError("censor_frontier must be >= 1")

    plan = paper_cases(profile)
    print(
        "CONFIG="
        + json.dumps(
            {
                "profile": profile,
                "families": list(plan),
                "embeddings_per_x": embeddings,
                "timeout_s": timeout_s,
                "base_seed": base_seed,
                "censor_frontier": censor_frontier,
                "timed_repeats_per_sample": 1,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    rows = []
    for family, cases in plan.items():
        active = {"knottedgraph": True, "topoly": True}
        consecutive_fully_censored = {"knottedgraph": 0, "topoly": 0}
        print(f"FAMILY={family}", flush=True)

        for case in cases:
            case_rows = []
            embedding_hashes = set()
            for embedding in range(embeddings):
                seed = _seed_for(base_seed, family, case.size, embedding)
                row = _row(case, embedding, seed, timeout_s, active)
                rows.append(row)
                case_rows.append(row)
                embedding_hashes.add(row["embedding_hash"])
                print(json.dumps(row, separators=(",", ":")), flush=True)

            if len(embedding_hashes) != embeddings:
                raise AssertionError(
                    f"{family}/{case.size}: duplicate embedding geometry detected"
                )

            for framework in ("knottedgraph", "topoly"):
                if not active[framework]:
                    continue
                statuses = [row[f"{framework}_status"] for row in case_rows]
                if all(status in {"timeout", "error"} for status in statuses):
                    consecutive_fully_censored[framework] += 1
                else:
                    consecutive_fully_censored[framework] = 0
                if consecutive_fully_censored[framework] >= censor_frontier:
                    active[framework] = False
                    print(
                        f"CENSOR_FRONTIER={family}:{framework}:{case.size}",
                        flush=True,
                    )

            if not any(active.values()):
                break

    print("SUMMARY=" + json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--profile",
        choices=("smoke", "paper"),
        default="paper",
    )
    parser.add_argument("--embeddings", type=int, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--censor-frontier", type=int, default=2)
    args = parser.parse_args()
    main(
        args.timeout,
        args.profile,
        args.embeddings,
        args.seed,
        args.censor_frontier,
    )
