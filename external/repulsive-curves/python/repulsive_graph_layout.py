import argparse
from pathlib import Path

from ctypes_layout import run_layout_ctypes_workspace
from repulsive_layout import read_graph, run_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a repulsive-curves graph drawing scene and run the solver.")
    parser.add_argument("graph", type=Path, help="Graph file: .json, .edgelist/.txt, or .graphml")
    parser.add_argument("--workspace", type=Path, default=Path("generated"), help="Output workspace directory")
    parser.add_argument("--samples-per-edge", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--layout-mode", choices=["auto", "spring", "circular", "bipartite"], default="auto")
    parser.add_argument("--bend-strength", type=float, default=0.15)
    parser.add_argument("--solver", type=Path, default=Path("build/bin/rcurves_headless.exe"))
    parser.add_argument("--engine", choices=["subprocess", "ctypes"], default="subprocess")
    parser.add_argument(
        "--solver-arg",
        action="append",
        default=[],
        help="Extra arg passed through to rcurves_headless. Can be repeated.",
    )
    args = parser.parse_args()

    graph = read_graph(args.graph)
    if args.engine == "ctypes":
        result = run_layout_ctypes_workspace(
            graph,
            workspace=args.workspace,
            library_path=args.solver,
            steps=args.steps,
            seed=args.seed,
            jitter=args.jitter,
            scale=args.scale,
            layout_mode=args.layout_mode,
            samples_per_edge=args.samples_per_edge,
            bend_strength=args.bend_strength,
        )
    else:
        result = run_layout(
            graph,
            workspace=args.workspace,
            solver=args.solver,
            samples_per_edge=args.samples_per_edge,
            steps=args.steps,
            seed=args.seed,
            jitter=args.jitter,
            scale=args.scale,
            layout_mode=args.layout_mode,
            bend_strength=args.bend_strength,
            extra_solver_args=args.solver_arg,
        )
    print("Wrote:", result.artifacts.final_curve_path)
    print("Wrote:", result.artifacts.layout_json_path)


if __name__ == "__main__":
    main()
