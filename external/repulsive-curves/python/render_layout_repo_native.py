import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render layout.json with the repo-native Polyscope renderer")
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--renderer", type=Path, default=Path("build/bin/rcurves_polyscope_render.exe"))
    parser.add_argument("--palette", choices=["default", "k5", "k33"], default="default")
    parser.add_argument("--curve-radius", type=float, default=0.015)
    parser.add_argument("--node-radius", type=float, default=0.038)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--yaw", type=float, default=35.0)
    parser.add_argument("--pitch", type=float, default=18.0)
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()

    output = args.output or args.layout_json.with_name("repo_native.png")
    command = [
        str(args.renderer),
        str(args.layout_json),
        "--output",
        str(output),
        "--palette",
        args.palette,
        "--curve-radius",
        str(args.curve_radius),
        "--node-radius",
        str(args.node_radius),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--yaw",
        str(args.yaw),
        "--pitch",
        str(args.pitch),
    ]
    if args.transparent:
        command.append("--transparent")

    subprocess.run(command, check=True)
    print("Wrote:", output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Renderer failed with exit code {exc.returncode}", file=sys.stderr)
        raise
