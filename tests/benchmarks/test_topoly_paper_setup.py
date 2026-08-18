from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
CONTROLLED_SCRIPT = ROOT / "dev" / "benchmark_topoly_extended_scaling.py"
RANDOM_CUBIC_SCRIPT = ROOT / "dev" / "benchmark_topoly_random_cubic_ensemble.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _controlled():
    return _load_module(CONTROLLED_SCRIPT, "kg_topoly_scaling_benchmark")


def _random_cubic():
    return _load_module(RANDOM_CUBIC_SCRIPT, "kg_topoly_random_cubic_benchmark")


def test_paper_profile_extends_all_scaling_axes():
    bench = _controlled()
    cases = bench._cases("paper")

    assert max(case.size for case in cases["crossings_fixed"]) >= 80
    assert max(case.size for case in cases["crossings_throughput"]) >= 1000
    assert max(case.size for case in cases["edges_theta"]) >= 5000
    assert max(case.size for case in cases["vertices_k4"]) >= 8192
    assert max(case.size for case in cases["connected_prism"]) >= 256

    assert bench.DEFAULT_EMBEDDINGS == 10
    assert bench.DEFAULT_TIMEOUT_S >= 120


def test_ten_embedding_seeds_and_geometries_are_distinct():
    bench = _controlled()
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


def test_random_cubic_paper_profile_and_default_sample_count():
    bench = _random_cubic()
    vertices = bench.vertex_grid("paper")

    assert vertices[0] >= 10
    assert max(vertices) >= 200
    assert all(vertex_count % 2 == 0 for vertex_count in vertices)
    assert bench.DEFAULT_SAMPLES == 10
    assert bench.DEFAULT_TIMEOUT_S >= 120


def test_random_cubic_instances_are_connected_cubic_and_nonisomorphic():
    bench = _random_cubic()
    # Four samples make the structural test fast while exercising the exact
    # pairwise-isomorphism rejection used by the paper run (which requests ten).
    ensemble = bench.topology_ensemble(10, 4, bench.DEFAULT_SEED)
    assert len(ensemble) == 4

    hashes = []
    for sample, graph in ensemble:
        assert sample.vertex_count == 10
        assert nx.is_connected(graph)
        assert graph.number_of_nodes() == 10
        assert graph.number_of_edges() == 15
        assert all(degree == 3 for _, degree in graph.degree())
        hashes.append(bench._abstract_hash(graph))

    assert len(set(hashes)) == 4
    for i in range(len(ensemble)):
        for j in range(i):
            assert not nx.is_isomorphic(ensemble[i][1], ensemble[j][1])


def test_random_cubic_pd_preparation_preserves_abstract_topology():
    bench = _random_cubic()
    sample, abstract = bench.topology_ensemble(10, 1, bench.DEFAULT_SEED)[0]
    embedded, processor, pdcode, embedding_attempt = bench.prepare_sample(
        sample,
        abstract,
        bench.DEFAULT_SEED,
    )

    assert embedded.number_of_nodes() == abstract.number_of_nodes()
    assert embedded.number_of_edges() == abstract.number_of_edges()
    assert nx.is_isomorphic(nx.Graph(embedded), abstract)
    assert isinstance(pdcode, str) and pdcode
    assert embedding_attempt >= 0
    assert len(processor.vertices) == abstract.number_of_nodes()
