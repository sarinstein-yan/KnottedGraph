import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from ctypes_layout import run_layout_ctypes_workspace
from repulsive_layout import read_graph, run_layout


def iter_graph_files(paths: Iterable[Path], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in path.glob(pattern):
                if child.suffix.lower() in {".json", ".edgelist", ".txt", ".graphml"} and child.is_file():
                    files.append(child)
    return sorted(set(files))


def safe_name(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    parent = path.parent.name.replace(" ", "_")
    return stem if not parent else f"{parent}__{stem}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 3D graph layout with repulsive curves.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Graph files or directories")
    parser.add_argument("--workspace", type=Path, default=Path("build/batch"))
    parser.add_argument("--solver", type=Path, default=Path("build/bin/rcurves_headless.exe"))
    parser.add_argument("--engine", choices=["subprocess", "ctypes"], default="subprocess")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--samples-per-edge", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--layout-mode", choices=["auto", "spring", "circular", "bipartite"], default="auto")
    parser.add_argument("--bend-strength", type=float, default=0.15)
    parser.add_argument("--solver-arg", action="append", default=[])
    args = parser.parse_args()

    graph_files = iter_graph_files(args.inputs, recursive=args.recursive)
    args.workspace.mkdir(parents=True, exist_ok=True)
    summary_path = args.workspace / "summary.csv"

    rows = []
    for index, graph_file in enumerate(graph_files):
        workspace = args.workspace / safe_name(graph_file)
        graph = read_graph(graph_file)
        if args.engine == "ctypes":
            result = run_layout_ctypes_workspace(
                graph,
                workspace=workspace,
                library_path=args.solver,
                samples_per_edge=args.samples_per_edge,
                steps=args.steps,
                seed=args.seed + index,
                jitter=args.jitter,
                scale=args.scale,
                layout_mode=args.layout_mode,
                bend_strength=args.bend_strength,
            )
        else:
            result = run_layout(
                graph,
                workspace=workspace,
                solver=args.solver,
                samples_per_edge=args.samples_per_edge,
                steps=args.steps,
                seed=args.seed + index,
                jitter=args.jitter,
                scale=args.scale,
                layout_mode=args.layout_mode,
                bend_strength=args.bend_strength,
                extra_solver_args=args.solver_arg,
            )
        rows.append(
            {
                "input": str(graph_file),
                "workspace": str(workspace),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "layout_json": str(result.artifacts.layout_json_path),
                "final_obj": str(result.artifacts.final_curve_path),
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "workspace", "nodes", "edges", "layout_json", "final_obj"])
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
