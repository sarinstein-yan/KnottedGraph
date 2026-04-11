import argparse
from pathlib import Path

from examples import run_builtin_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run built-in repulsive graph drawing examples.")
    parser.add_argument("--solver", type=Path, default=Path("build/bin/rcurves_headless.exe"))
    parser.add_argument("--workspace", type=Path, default=Path("build/examples"))
    parser.add_argument("--engine", choices=["subprocess", "ctypes"], default="subprocess")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    run_builtin_examples(args.solver, args.workspace, engine=args.engine, steps=args.steps)
    print("Wrote examples under:", args.workspace)


if __name__ == "__main__":
    main()
