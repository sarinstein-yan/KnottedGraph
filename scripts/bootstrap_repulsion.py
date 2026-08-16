from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPULSOR_ROOT = REPO_ROOT / "external" / "Repulsor"
DEFAULT_REPULSOR_URL = "https://github.com/HenrikSchumacher/Repulsor.git"

# Exact upstream revision used for the KnottedGraph paper/reproducibility setup.
# Users may override this explicitly with --repulsor-ref.
DEFAULT_REPULSOR_REF = "adc56b61f65f5958b59cbd7e1539f44ed0c5e993"


def run(
    cmd: list[str],
    cwd: Path,
) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
    )


def output(
    cmd: list[str],
    cwd: Path,
) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def require_tool(
    name: str,
) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required tool: {name}. "
            "Install it first, then rerun this script."
        )


def in_virtualenv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or sys.prefix
        != getattr(
            sys,
            "base_prefix",
            sys.prefix,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the optional Repulsor dependency for "
            "knotted_graph.layout.repulsive without vendoring it "
            "into the Python package."
        )
    )
    parser.add_argument(
        "--repulsor-root",
        type=Path,
        default=DEFAULT_REPULSOR_ROOT,
        help=(
            "External Repulsor checkout path. "
            "Default: external/Repulsor."
        ),
    )
    parser.add_argument(
        "--repulsor-url",
        default=DEFAULT_REPULSOR_URL,
        help=(
            "Git URL to clone when --repulsor-root is absent. "
            f"Default: {DEFAULT_REPULSOR_URL}"
        ),
    )
    parser.add_argument(
        "--repulsor-ref",
        default=DEFAULT_REPULSOR_REF,
        help=(
            "Exact Repulsor commit/tag to use. "
            "The default is the revision pinned for this "
            "KnottedGraph release."
        ),
    )
    parser.add_argument(
        "--skip-python-install",
        action="store_true",
        help=(
            "Skip installation of Python extras used by "
            "repulsive-layout examples."
        ),
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help=(
            "Do not clone Repulsor; only validate the "
            "requested checkout path and revision."
        ),
    )
    return parser.parse_args()


def _resolve_ref(
    root: Path,
    ref: str,
) -> str:
    try:
        return output(
            [
                "git",
                "rev-parse",
                f"{ref}^{{commit}}",
            ],
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not resolve Repulsor revision {ref!r} "
            f"inside {root}. Fetch that revision or use "
            "--repulsor-ref with a revision present in the checkout."
        ) from exc


def _validate_existing_checkout(
    root: Path,
    ref: str,
) -> Path:
    if not root.joinpath("Repulsor.hpp").exists():
        raise SystemExit(
            f"{root} exists but does not look like a "
            "Repulsor checkout: Repulsor.hpp is missing."
        )

    if not root.joinpath(".git").exists():
        raise SystemExit(
            f"{root} contains Repulsor.hpp but is not a Git checkout. "
            "For a reproducible paper run, use a Git checkout at "
            f"revision {ref}."
        )

    require_tool("git")

    expected = _resolve_ref(
        root,
        ref,
    )
    current = output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=root,
    )

    if current != expected:
        raise SystemExit(
            "Existing Repulsor checkout is at a different revision.\n"
            f"  checkout: {root}\n"
            f"  current:  {current}\n"
            f"  expected: {expected}\n"
            "KnottedGraph will not silently modify an existing checkout. "
            "Switch that checkout yourself, use a separate checkout, or "
            "override --repulsor-ref intentionally."
        )

    run(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        cwd=root,
    )
    return root


def ensure_repulsor_checkout(
    root: Path,
    url: str,
    ref: str,
    *,
    skip_clone: bool,
) -> Path:
    root = root.resolve()

    if root.exists():
        return _validate_existing_checkout(
            root,
            ref,
        )

    if skip_clone:
        raise SystemExit(
            f"Repulsor checkout was not found at {root}. "
            "Clone it there, set REPULSOR_ROOT to another "
            "checkout, or rerun without --skip-clone."
        )

    require_tool("git")
    root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        [
            "git",
            "clone",
            "--recursive",
            url,
            str(root),
        ],
        cwd=REPO_ROOT,
    )

    run(
        [
            "git",
            "checkout",
            "--detach",
            ref,
        ],
        cwd=root,
    )

    run(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        cwd=root,
    )

    if not root.joinpath(
        "Repulsor.hpp"
    ).exists():
        raise SystemExit(
            "Repulsor clone completed, but Repulsor.hpp "
            f"was not found at {root}."
        )

    return _validate_existing_checkout(
        root,
        ref,
    )


def main() -> None:
    args = parse_args()

    if not REPO_ROOT.joinpath(
        "pyproject.toml"
    ).exists():
        raise SystemExit(
            f"Expected repository root at {REPO_ROOT}, "
            "but pyproject.toml was not found."
        )

    if not args.skip_python_install:
        if not in_virtualenv():
            print(
                "Warning: no virtual environment detected. "
                "A dedicated venv or conda environment is recommended."
            )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".[repulsion]",
            ],
            cwd=REPO_ROOT,
        )

    repulsor_root = ensure_repulsor_checkout(
        args.repulsor_root,
        args.repulsor_url,
        args.repulsor_ref,
        skip_clone=args.skip_clone,
    )

    revision = output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repulsor_root,
    )

    print("Repulsor checkout ready:")
    print(f"  path:     {repulsor_root}")
    print(f"  revision: {revision}")
    print("Use it with:")
    print(
        f"  export REPULSOR_ROOT={repulsor_root}"
    )
    print(
        "The C++ driver is compiled lazily by "
        "knotted_graph.layout.repulsive.driver.build_driver()."
    )


if __name__ == "__main__":
    main()
