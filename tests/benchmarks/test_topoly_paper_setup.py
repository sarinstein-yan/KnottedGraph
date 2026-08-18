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
    profiles = []
    for sample, graph in ensemble:
        assert sample.vertex_count == 10
        assert nx.is_connected(graph)
        assert graph.number_of_nodes() == 10
        assert graph.number_of_edges() == 15
        assert all(degree == 3 for _, degree in graph.degree())
        hashes.append(bench._abstract_hash(graph))
        profiles.append(bench._isomorphism_profile(graph))

    assert len(set(hashes)) == 4
    for i in range(len(ensemble)):
        for j in range(i):
            assert not bench._are_isomorphic_exact(
                ensemble[i][1],
                ensemble[j][1],
                profiles[i],
                profiles[j],
            )


def test_random_cubic_isomorphism_prefilter_preserves_exactness():
    bench = _random_cubic()
    _sample, graph = bench.topology_ensemble(40, 1, bench.DEFAULT_SEED)[0]

    relabeling = {
        node: graph.number_of_nodes() - 1 - node
        for node in graph.nodes()
    }
    relabeled = nx.relabel_nodes(graph, relabeling, copy=True)

    left_profile = bench._isomorphism_profile(graph)
    right_profile = bench._isomorphism_profile(relabeled)
    assert left_profile[0] == right_profile[0]
    assert bench._are_isomorphic_exact(
        graph,
        relabeled,
        left_profile,
        right_profile,
    )

    other = bench.topology_ensemble(40, 2, bench.DEFAULT_SEED)[1][1]
    other_profile = bench._isomorphism_profile(other)
    assert not bench._are_isomorphic_exact(
        graph,
        other,
        left_profile,
        other_profile,
    )


def test_random_cubic_large_ensemble_uses_scalable_exact_filter():
    bench = _random_cubic()
    ensemble = bench.topology_ensemble(200, 10, bench.DEFAULT_SEED)
    assert len(ensemble) == 10

    profiles = [bench._isomorphism_profile(graph) for _, graph in ensemble]
    for i in range(len(ensemble)):
        for j in range(i):
            assert not bench._are_isomorphic_exact(
                ensemble[i][1],
                ensemble[j][1],
                profiles[i],
                profiles[j],
            )


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


def test_committed_random_cubic_corpus_is_complete_and_reconstructs_pd():
    bench = _random_cubic()
    corpus = bench.DEFAULT_CORPUS
    assert corpus.exists()
    rows = [
        __import__("json").loads(line)
        for line in corpus.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == len(bench.vertex_grid("paper")) * bench.DEFAULT_SAMPLES

    for vertex_count in bench.vertex_grid("paper"):
        group = [row for row in rows if int(row["V"]) == vertex_count]
        assert len(group) == bench.DEFAULT_SAMPLES
        assert len({row["graph6"] for row in group}) == bench.DEFAULT_SAMPLES
        assert len({row["pdcode"] for row in group}) == bench.DEFAULT_SAMPLES

    # Reconstruct representative small, middle and large committed instances.
    for vertex_count in (10, 64, 200):
        sample, abstract = bench.load_committed_ensemble(vertex_count, 1)[0]
        embedded, processor, pdcode, _ = bench.prepare_sample(
            sample, abstract, bench.DEFAULT_SEED
        )
        row = abstract.graph["_committed_benchmark"]
        assert pdcode == row["pdcode"]
        assert len(processor.crossings) == int(row["crossings"])
        assert embedded.number_of_nodes() == vertex_count
        assert embedded.number_of_edges() == 3 * vertex_count // 2
