from __future__ import annotations

import argparse
import json
from pathlib import Path

from .driver import DEFAULT_DRIVER_BINARY, DEFAULT_DRIVER_SOURCE, DEFAULT_REPULSOR_ROOT, DriverConfig, SolverOptions
from .pipeline import run_protein_example
from .protein_examples import available_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m knotted_graph.repulsive_layout",
        description="Run Repulsor safe-step spatial graph layout examples.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    examples = subparsers.add_parser("examples", help="Run a built-in protein-derived theta graph example.")
    examples.add_argument("--sample", choices=available_samples(), default="1aoc")
    examples.add_argument("--out", type=Path, default=None, help="Output directory.")
    examples.add_argument("--pdb-cache", type=Path, default=None)
    examples.add_argument("--pdb", type=Path, default=None, help="Use a local PDB file instead of downloading.")
    examples.add_argument("--steps", type=int, default=100)
    examples.add_argument("--q", type=float, default=4.0)
    examples.add_argument("--p", type=float, default=8.0)
    examples.add_argument("--threads", type=int, default=1)
    examples.add_argument("--max-time", type=float, default=1.0)
    examples.add_argument("--safe-fraction", type=float, default=0.95)
    examples.add_argument("--max-backtracks", type=int, default=12)
    examples.add_argument("--max-iter", type=int, default=60)
    examples.add_argument("--tolerance", type=float, default=1e-4)
    examples.add_argument("--free-special-vertices", action="store_true")
    examples.add_argument("--seed", type=int, default=7)
    examples.add_argument("--total-arc-points", type=int, default=None)
    examples.add_argument("--target-node-distance", type=float, default=None)
    examples.add_argument("--node-distance-scale", type=float, default=1.0)
    examples.add_argument("--repulsor-root", type=Path, default=DEFAULT_REPULSOR_ROOT)
    examples.add_argument("--driver-source", type=Path, default=DEFAULT_DRIVER_SOURCE)
    examples.add_argument("--driver-binary", type=Path, default=DEFAULT_DRIVER_BINARY)
    examples.add_argument("--force-build", action="store_true")
    examples.add_argument("--keep-workspace", action="store_true")
    examples.add_argument("--no-save-steps", action="store_true")
    examples.add_argument("--no-render", action="store_true")
    examples.add_argument("--use-wsl", action=argparse.BooleanOptionalAction, default=None)
    examples.add_argument("--quiet", action="store_true")
    return parser


def run_examples(args: argparse.Namespace) -> int:
    workspace = args.out
    if workspace is None:
        workspace = Path("build") / "repulsive_layout" / f"{args.sample}_{args.steps}steps"

    result = run_protein_example(
        sample=args.sample,
        workspace=workspace,
        pdb_cache=args.pdb_cache,
        pdb_path=args.pdb,
        total_arc_points=args.total_arc_points,
        seed=args.seed,
        target_node_distance=args.target_node_distance,
        node_distance_scale=args.node_distance_scale,
        solver_options=SolverOptions(
            steps=args.steps,
            q=args.q,
            p=args.p,
            threads=args.threads,
            max_time=args.max_time,
            safe_fraction=args.safe_fraction,
            max_backtracks=args.max_backtracks,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            free_special_vertices=args.free_special_vertices,
        ),
        driver_config=DriverConfig(
            repulsor_root=args.repulsor_root,
            driver_source=args.driver_source,
            driver_binary=args.driver_binary,
            use_wsl=args.use_wsl,
            verbose=not args.quiet,
        ),
        force_build=args.force_build,
        keep_workspace=args.keep_workspace,
        save_steps=not args.no_save_steps,
        render_html=not args.no_render,
    )

    summary = {
        "workspace": str(result.workspace),
        "initial_html": result.metadata.get("initial_html"),
        "final_html": result.metadata.get("final_html"),
        "metadata_json": str(result.workspace / "metadata.json"),
        "certificate": result.metadata.get("certificate"),
        "history_summary": result.metadata.get("history_summary"),
        "elapsed_seconds": result.metadata.get("elapsed_seconds"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "examples":
        return run_examples(args)
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
