from __future__ import annotations

import argparse
import json
from pathlib import Path

from .driver import DEFAULT_DRIVER_BINARY, DEFAULT_DRIVER_SOURCE, DEFAULT_REPULSOR_ROOT, DriverConfig, SolverOptions
from .pipeline import run_protein_example
from .protein_examples import available_samples
from .resampling import ResamplingOptions
from .topology import verify_obj_step_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m knotted_graph.layout.repulsive",
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
    examples.add_argument("--repulsion-weight", type=float, default=1.0)
    examples.add_argument("--length-weight", type=float, default=0.0)
    examples.add_argument("--curve-length-floor-weight", type=float, default=0.0)
    examples.add_argument("--bend-weight", type=float, default=0.0)
    examples.add_argument("--tube-radius", type=float, default=0.0)
    examples.add_argument("--tube-gap", type=float, default=0.0)
    examples.add_argument("--tube-weight", type=float, default=0.0)
    examples.add_argument("--topology-tolerance", type=float, default=1e-7)
    examples.add_argument("--no-topology-check", action="store_true")
    examples.add_argument("--resample-target-segment-length", type=float, default=None)
    examples.add_argument("--resample-points-per-edge", type=int, default=None)
    examples.add_argument("--resample-min-points-per-edge", type=int, default=2)
    examples.add_argument("--resample-max-points-per-edge", type=int, default=None)
    examples.add_argument("--resample-allow-downsample", action="store_true")
    examples.add_argument("--resample-downsample-min-clearance", type=float, default=None)
    examples.add_argument("--free-special-vertices", action="store_true")
    examples.add_argument("--seed", type=int, default=7)
    examples.add_argument("--total-arc-points", type=int, default=None)
    examples.add_argument("--target-node-distance", type=float, default=None)
    examples.add_argument("--node-distance-scale", type=float, default=1.0)
    examples.add_argument(
        "--pin-node-collar-points",
        type=int,
        default=0,
        help="Also pin this many interior points next to each graph node on every incident edge.",
    )
    examples.add_argument("--repulsor-root", type=Path, default=DEFAULT_REPULSOR_ROOT)
    examples.add_argument("--driver-source", type=Path, default=DEFAULT_DRIVER_SOURCE)
    examples.add_argument("--driver-binary", type=Path, default=DEFAULT_DRIVER_BINARY)
    examples.add_argument("--force-build", action="store_true")
    examples.add_argument("--keep-workspace", action="store_true")
    examples.add_argument(
        "--save-steps",
        action="store_true",
        help="Save accepted-step OBJ files under certified_steps/ for later independent verification.",
    )
    examples.add_argument(
        "--no-save-steps",
        action="store_false",
        dest="save_steps",
        help=argparse.SUPPRESS,
    )
    examples.add_argument(
        "--verify-topology",
        action="store_true",
        help="Run the independent Python swept-step verifier after optimization. Implies --save-steps.",
    )
    examples.add_argument(
        "--no-verify-topology",
        action="store_false",
        dest="verify_topology",
        help=argparse.SUPPRESS,
    )
    examples.add_argument("--no-render", action="store_true")
    examples.add_argument("--no-simplify", action="store_true")
    examples.add_argument("--use-wsl", action=argparse.BooleanOptionalAction, default=None)
    examples.add_argument("--quiet", action="store_true")
    examples.set_defaults(save_steps=False, verify_topology=False)

    verify = subparsers.add_parser("verify-steps", help="Verify swept topology over saved step OBJ files.")
    verify.add_argument("--steps-dir", type=Path, required=True)
    verify.add_argument("--epsilon", type=float, default=1e-7)
    verify.add_argument("--pattern", default="step_*.obj")
    verify.add_argument("--out", type=Path, default=None)
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
            repulsion_weight=args.repulsion_weight,
            length_weight=args.length_weight,
            curve_length_floor_weight=args.curve_length_floor_weight,
            bend_weight=args.bend_weight,
            tube_radius=args.tube_radius,
            tube_gap=args.tube_gap,
            tube_weight=args.tube_weight,
            topology_check=not args.no_topology_check,
            topology_tolerance=args.topology_tolerance,
            free_special_vertices=args.free_special_vertices,
        ),
        resampling_options=ResamplingOptions(
            target_segment_length=args.resample_target_segment_length,
            points_per_edge=args.resample_points_per_edge,
            min_points_per_edge=args.resample_min_points_per_edge,
            max_points_per_edge=args.resample_max_points_per_edge,
            allow_downsample=args.resample_allow_downsample,
            downsample_min_clearance=args.resample_downsample_min_clearance,
        )
        if (
            args.resample_target_segment_length is not None
            or args.resample_points_per_edge is not None
            or args.resample_min_points_per_edge != 2
            or args.resample_max_points_per_edge is not None
            or args.resample_allow_downsample
            or args.resample_downsample_min_clearance is not None
        )
        else None,
        driver_config=DriverConfig(
            repulsor_root=args.repulsor_root,
            driver_source=args.driver_source,
            driver_binary=args.driver_binary,
            use_wsl=args.use_wsl,
            verbose=not args.quiet,
        ),
        force_build=args.force_build,
        keep_workspace=args.keep_workspace,
        save_steps=args.save_steps or args.verify_topology,
        render_html=not args.no_render,
        simplify_after_layout=not args.no_simplify,
        pin_node_collar_points=args.pin_node_collar_points,
        verify_topology=args.verify_topology,
    )

    summary = {
        "workspace": str(result.workspace),
        "initial_html": result.metadata.get("initial_html"),
        "final_html": result.metadata.get("final_html"),
        "final_simplified_html": result.metadata.get("final_simplified_html"),
        "output_html": result.metadata.get("output_html"),
        "metadata_json": str(result.workspace / "metadata.json"),
        "certificate": result.metadata.get("certificate"),
        "history_summary": result.metadata.get("history_summary"),
        "decimation": result.metadata.get("decimation"),
        "elapsed_seconds": result.metadata.get("elapsed_seconds"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_verify_steps(args: argparse.Namespace) -> int:
    result = verify_obj_step_sequence(args.steps_dir, epsilon=args.epsilon, pattern=args.pattern)
    payload = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["verified"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "examples":
        return run_examples(args)
    if args.command == "verify-steps":
        return run_verify_steps(args)
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
