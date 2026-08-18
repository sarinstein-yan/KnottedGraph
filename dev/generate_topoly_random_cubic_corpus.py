from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

import benchmark_topoly_random_cubic_ensemble as bench


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "benchmark_data"
    / "topoly_random_cubic_v1.jsonl"
)


def _canonical_embedded(abstract: nx.Graph, embedded: nx.MultiGraph):
    """Rebuild a benchmark embedding in graph6's deterministic edge order."""
    canonical = nx.MultiGraph()
    nodes = sorted(abstract.nodes())
    for node in nodes:
        canonical.add_node(node, pos=np.asarray(embedded.nodes[node]["pos"], dtype=float))

    edges = sorted((min(u, v), max(u, v)) for u, v in abstract.edges())
    for u, v in edges:
        canonical.add_edge(
            u,
            v,
            pts=np.vstack([canonical.nodes[u]["pos"], canonical.nodes[v]["pos"]]),
        )
    return canonical, edges


def _record(sample, abstract: nx.Graph, embedded, processor, pdcode: str, embedding_attempt: int):
    del processor, pdcode  # Stored PD data is derived from the canonical reconstruction.
    canonical, edges = _canonical_embedded(abstract, embedded)
    canonical_processor = bench.PDCode(canonical)
    canonical_pdcode = canonical_processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    graph6 = nx.to_graph6_bytes(abstract, header=False).decode("ascii").strip()

    # graph6 decoding inserts nodes in numerical order and produces this same
    # lexicographic undirected edge order. Assert that assumption here so the
    # committed PD representation cannot silently depend on NetworkX internals.
    reconstructed = nx.from_graph6_bytes(graph6.encode("ascii"))
    reconstructed_edges = list(reconstructed.edges())
    assert reconstructed_edges == edges, (reconstructed_edges, edges)

    return {
        "schema_version": 2,
        "family": "random_cubic",
        "V": abstract.number_of_nodes(),
        "E": abstract.number_of_edges(),
        "sample": sample.sample_index,
        "topology_seed": sample.topology_seed,
        "topology_attempt": sample.topology_attempt,
        "embedding_attempt": embedding_attempt,
        "graph6": graph6,
        "edge_list": [[int(u), int(v)] for u, v in edges],
        "embedding_edge_order": [[int(u), int(v)] for u, v in edges],
        "node_positions": {
            str(int(node)): [float(x) for x in canonical.nodes[node]["pos"]]
            for node in sorted(canonical.nodes())
        },
        "pdcode": canonical_pdcode,
        "crossings": len(canonical_processor.crossings),
        "topology_instance_hash": bench._abstract_hash(abstract),
        "embedding_hash": bench._embedding_hash(canonical),
    }


def generate(output: Path, *, base_seed: int, samples_per_v: int) -> list[dict]:
    rows: list[dict] = []
    for vertex_count in bench.vertex_grid("paper"):
        ensemble = bench.topology_ensemble(vertex_count, samples_per_v, base_seed)
        for sample, abstract in ensemble:
            embedded, processor, pdcode, embedding_attempt = bench.prepare_sample(
                sample,
                abstract,
                base_seed,
            )
            row = _record(
                sample,
                abstract,
                embedded,
                processor,
                pdcode,
                embedding_attempt,
            )
            rows.append(row)
            print(
                f"V={vertex_count} sample={sample.sample_index + 1}/{samples_per_v} "
                f"crossings={row['crossings']}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} committed benchmark instances to {output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=bench.DEFAULT_SEED)
    parser.add_argument("--samples-per-v", type=int, default=bench.DEFAULT_SAMPLES)
    args = parser.parse_args()
    generate(args.output, base_seed=args.seed, samples_per_v=args.samples_per_v)


if __name__ == "__main__":
    main()
