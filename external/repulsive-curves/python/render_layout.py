import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTES = {
    "default": None,
    "k5": ["#b59a18", "#275fb2", "#8b2ca8", "#b1222e", "#22a84b"],
    "k33": ["#b1222e", "#b1222e", "#b1222e", "#275fb2", "#275fb2", "#275fb2"],
}


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max([x_range, y_range, z_range, 1e-9])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static 3D figure from layout.json")
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=38.0)
    parser.add_argument("--node-size", type=float, default=90.0)
    parser.add_argument("--edge-width", type=float, default=2.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--palette", choices=list(PALETTES.keys()), default="default")
    args = parser.parse_args()

    data = json.loads(args.layout_json.read_text(encoding="utf-8"))
    edge_polylines = data["edge_polylines_final"]
    node_positions = data["node_positions_final"]
    node_order = data["node_order"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for poly in edge_polylines.values():
        arr = np.asarray(poly, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="#b9b9b9", linewidth=args.edge_width, alpha=0.95)

    coords = np.asarray([node_positions[node] for node in node_order], dtype=float)
    palette = PALETTES[args.palette]
    if palette is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(node_order), endpoint=False))
    else:
        colors = palette[: len(node_order)]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=args.node_size, c=colors, depthshade=True)

    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    set_axes_equal(ax)
    plt.tight_layout()

    output = args.output or args.layout_json.with_suffix(".png")
    fig.savefig(output, dpi=args.dpi, transparent=args.transparent, bbox_inches="tight", pad_inches=0.02)
    print("Wrote:", output)


if __name__ == "__main__":
    main()
