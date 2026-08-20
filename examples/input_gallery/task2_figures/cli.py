"""Command-line entry point for the three current Task 2 figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

# The command-line renderer is always noninteractive. Importable modules do not
# set a backend, so notebook and GUI callers keep control of their session.
mpl.use("Agg")

from .common import DEFAULT_ASSET_ROOT, DEFAULT_OUTPUT_DIR, validate_assets
from .main import render_main
from .s1 import render_s1
from .s2 import render_s2
from .specs import FIGURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m examples.input_gallery.task2_figures",
        description="Validate accepted panels and build the current Main, S1, and S2 figures.",
    )
    parser.add_argument(
        "target",
        choices=("main", "s1", "s2", "all", "verify"),
        help="figure to build, all three figures, or input verification only",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help=f"accepted panel bundle root (default: {DEFAULT_ASSET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"generated output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asset_root = args.asset_root.expanduser()
    output_dir = args.output_dir.expanduser()

    if args.target == "verify":
        verified = {
            key: len(validate_assets(spec, asset_root))
            for key, spec in FIGURES.items()
        }
        print(json.dumps({"verified": verified, "asset_root": str(asset_root.resolve())}, indent=2))
        return 0

    builders = {
        "main": render_main,
        "s1": render_s1,
        "s2": render_s2,
    }
    targets = tuple(builders) if args.target == "all" else (args.target,)
    # Validate the complete selected input set before the first renderer is
    # allowed to write. Individual builders recheck their own inputs so the
    # public Python API remains fail closed as well.
    for target in targets:
        validate_assets(FIGURES[target], asset_root)
    results = []
    for target in targets:
        result = builders[target](asset_root=asset_root, output_dir=output_dir)
        results.append(
            {
                "figure": result.figure,
                "outputs": [str(path) for path in result.outputs],
                "summary": str(result.summary),
            }
        )
    print(json.dumps({"results": results}, indent=2))
    return 0
