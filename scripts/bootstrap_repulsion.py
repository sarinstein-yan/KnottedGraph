from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPULSOR_ROOT = REPO_ROOT / "external" / "Repulsor"
DEFAULT_REPULSOR_URL = "https://github.com/HenrikSchumacher/Repulsor.git"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required tool: {name}. Install it first, then rerun this script."
        )


def in_virtualenv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the optional Repulsor dependency for "
            "knotted_graph.repulsive_layout without vendoring it into the package."
        )
    )
    parser.add_argument(
        "--repulsor-root",
        type=Path,
        default=DEFAULT_REPULSOR_ROOT,
        help="External Repulsor checkout path. Default: external/Repulsor.",
    )
    parser.add_argument(
        "--repulsor-url",
        default=DEFAULT_REPULSOR_URL,
        help=f"Git URL to clone when --repulsor-root is absent. Default: {DEFAULT_REPULSOR_URL}",
    )
    parser.add_argument(
        "--skip-python-install",
        action="store_true",
        help="Skip installation of Python extras used by repulsive-layout examples.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Do not clone Repulsor; only validate the requested checkout path.",
    )
    return parser.parse_args()


def ensure_repulsor_checkout(root: Path, url: str, *, skip_clone: bool) -> Path:
    root = root.resolve()
    if root.joinpath("Repulsor.hpp").exists():
        return root

    if skip_clone:
        raise SystemExit(
            f"Repulsor headers were not found at {root}. "
            "Clone Repulsor there or set REPULSOR_ROOT to an existing checkout."
        )

    require_tool("git")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise SystemExit(
            f"{root} already exists but does not look like a Repulsor checkout."
        )

    run(["git", "clone", "--recursive", url, str(root)], cwd=REPO_ROOT)
    if not root.joinpath("Repulsor.hpp").exists():
        raise SystemExit(f"Clone completed, but Repulsor.hpp was not found at {root}.")
    return root


def main() -> None:
    args = parse_args()

    if not REPO_ROOT.joinpath("pyproject.toml").exists():
        raise SystemExit(f"Expected repo root at {REPO_ROOT}, but pyproject.toml was not found.")

    if not args.skip_python_install:
        if not in_virtualenv():
            print(
                "Warning: no virtual environment detected. "
                "A dedicated venv or conda environment is recommended."
            )
        run([sys.executable, "-m", "pip", "install", "-e", ".[repulsion]"], cwd=REPO_ROOT)

    repulsor_root = ensure_repulsor_checkout(
        args.repulsor_root,
        args.repulsor_url,
        skip_clone=args.skip_clone,
    )

    print("Repulsor checkout ready:")
    print(f"  {repulsor_root}")
    print("Use it with:")
    print(f"  export REPULSOR_ROOT={repulsor_root}")
    print("The C++ driver is compiled lazily by knotted_graph.repulsive_layout.driver.build_driver().")


if __name__ == "__main__":
    main()
