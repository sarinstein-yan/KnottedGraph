from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "dev" / "benchmark_topoly_extended_scaling.py"


def _load_benchmark_module():
    name = "kg_topoly_scaling_benchmark"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_profile_extends_all_scaling_axes():
    bench = _load_benchmark_module()
    cases = bench._cases("paper")

    assert max(case.size for case in cases["crossings_fixed"]) >= 80
    assert max(case.size for case in cases["crossings_throughput"]) >= 1000
    assert max(case.size for case in cases["edges_theta"]) >= 5000
    assert max(case.size for case in cases["vertices_k4"]) >= 8192
    assert max(case.size for case in cases["connected_prism"]) >= 256

    assert bench.DEFAULT_EMBEDDINGS == 10
    assert bench.DEFAULT_TIMEOUT_S >= 120


def test_ten_embedding_seeds_and_geometries_are_distinct():
    bench = _load_benchmark_module()
    representatives = [
        bench.Case("crossings_fixed", 4),
        bench.Case("crossings_throughput", 3),
        bench.Case("edges_theta", 8),
        bench.Case("vertices_k4", 8),
        bench.Case("connected_prism", 4),
    ]

    for case in representatives:
        seeds = [
            bench._seed_for(bench.DEFAULT_SEED, case.family, case.size, embedding)
            for embedding in range(10)
        ]
        assert len(set(seeds)) == 10

        hashes = []
        for seed in seeds:
            graph, processor, _pdcode = bench._prepare(case, seed)
            hashes.append(bench._embedding_hash(graph))
            assert len(processor.crossings) == bench._expected_crossings(case)

            if case.family == "crossings_fixed":
                assert graph.number_of_nodes() == 2
                assert graph.number_of_edges() == 3
            elif case.family == "edges_theta":
                assert graph.number_of_nodes() == 2
                assert graph.number_of_edges() == case.size
            elif case.family == "vertices_k4":
                assert graph.number_of_nodes() == case.size
            elif case.family == "connected_prism":
                assert graph.number_of_nodes() == 2 * case.size
                assert graph.number_of_edges() == 3 * case.size

        assert len(set(hashes)) == 10
