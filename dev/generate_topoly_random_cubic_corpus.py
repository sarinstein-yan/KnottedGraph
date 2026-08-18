from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

import benchmark_topoly_random_cubic_ensemble as bench


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "benchmark_data"
    / "topoly_random_cubic_v1.jsonl"
)


def _record(sample, abstract: nx.Graph, embedded, processor, pdcode: str, embedding_attempt: int):
    nodes = sorted(abstract.nodes())
    edges = sorted((min(u, v), max(u, v)) for u, v in abstract.edges())
    graph6 = nx.to_graph6_bytes(abstract, header=False).decode("ascii").strip()
    return {
        "schema_version": 1,
        "family": "random_cubic",
        "V": abstract.number_of_nodes(),
        "E": abstract.number_of_edges(),
        "sample": sample.sample_index,
        "topology_seed": sample.topology_seed,
        "topology_attempt": sample.topology_attempt,
        "embedding_attempt": embedding_attempt,
        "graph6": graph6,
        "edge_list": [[int(u), int(v)] for u, v in edges],
        "node_positions": {
            str(int(node)): [float(x) for x in embedded.nodes[node]["pos"]]
            for node in nodes
        },
        "pdcode": pdcode,
        "crossings": len(processor.crossings),
        "topology_instance_hash": bench._abstract_hash(abstract),
        "embedding_hash": bench._embedding_hash(embedded),
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
            rows.append(
                _record(
                    sample,
                    abstract,
                    embedded,
                    processor,
                    pdcode,
                    embedding_attempt,
                )
            )
            print(
                f"V={vertex_count} sample={sample.sample_index + 1}/{samples_per_v} "
                f"crossings={len(processor.crossings)}",
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
