from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPULSIVE_ROOT = REPO_ROOT / "external" / "repulsive-curves"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required tool: {name}. Install it first, then rerun this script."
        )


def find_binaries(build_dir: Path) -> list[Path]:
    patterns = [
        "**/rcurves_headless.exe",
        "**/rcurves_headless",
        "**/rcurves_shared.dll",
        "**/librcurves_shared.dll",
        "**/librcurves_shared.so",
        "**/librcurves_shared.dylib",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(build_dir.glob(pattern))
    return sorted({path.resolve() for path in found})


def in_virtualenv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install KnottedGraph repulsion dependencies and build vendored repulsive-curves."
    )
    parser.add_argument(
        "--config",
        default="Release",
        help="CMake build configuration. Default: Release",
    )
    parser.add_argument(
        "--build-dir",
        default=str(REPULSIVE_ROOT / "build"),
        help="CMake build directory for external/repulsive-curves.",
    )
    parser.add_argument(
        "--generator",
        default=None,
        help="Optional CMake generator, e.g. Ninja or MinGW Makefiles.",
    )
    parser.add_argument(
        "--skip-python-install",
        action="store_true",
        help="Skip pip installation of Python dependencies.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the CMake configure/build step.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel build jobs passed to CMake. Default: CPU count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not REPO_ROOT.joinpath("pyproject.toml").exists():
        raise SystemExit(f"Expected repo root at {REPO_ROOT}, but pyproject.toml was not found.")
    if not REPULSIVE_ROOT.joinpath("CMakeLists.txt").exists():
        raise SystemExit(
            "Vendored repulsive-curves source not found. Expected "
            f"{REPULSIVE_ROOT / 'CMakeLists.txt'}."
        )

    python = sys.executable
    build_dir = Path(args.build_dir).resolve()

    if not args.skip_python_install:
        if not in_virtualenv():
            print(
                "Warning: no virtual environment detected. "
                "A dedicated venv or conda environment is recommended."
            )
        run([python, "-m", "pip", "install", "-e", ".[repulsion]"], cwd=REPO_ROOT)

    if args.skip_build:
        print("Skipped C++ build step.")
        return

    require_tool("cmake")

    configure_cmd = [
        "cmake",
        "-S",
        str(REPULSIVE_ROOT),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={args.config}",
    ]
    if args.generator:
        configure_cmd.extend(["-G", args.generator])
    run(configure_cmd, cwd=REPO_ROOT)

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        args.config,
        "--parallel",
        str(args.parallel),
        "--target",
        "rcurves_headless",
        "rcurves_shared",
    ]
    run(build_cmd, cwd=REPO_ROOT)

    binaries = find_binaries(build_dir)
    if binaries:
        print("Built repulsive-curves binaries:")
        for path in binaries:
            print(f"  - {path}")
    else:
        print(
            "Build completed, but binaries were not found in the expected locations. "
            "Check the CMake output above."
        )


if __name__ == "__main__":
    main()
