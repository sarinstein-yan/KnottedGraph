import argparse
from pathlib import Path

from repulsive_layout import read_graph
from ctypes_layout import run_layout_ctypes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repulsive-curves through the shared library via ctypes.")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--dll", type=Path, default=Path("build/bin/librcurves_shared.dll"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples-per-edge", type=int, default=8)
    args = parser.parse_args()

    graph = read_graph(args.graph)
    result = run_layout_ctypes(
        graph,
        library_path=args.dll,
        steps=args.steps,
        samples_per_edge=args.samples_per_edge,
    )
    print("nodes:", len(result.node_positions_final))
    print("edges:", len(result.edge_polylines_final))
    first = result.node_order[0]
    print("first node:", first, result.node_positions_final[first].tolist())


if __name__ == "__main__":
    main()
