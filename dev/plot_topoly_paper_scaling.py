from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818
FAMILY = "crossings_graph_ensemble"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "size", "sample", "V", "E", "crossings",
            "knottedgraph_s", "topoly_s", "timeout_s",
        ):
            if row.get(key) not in {"", None}:
                row[key] = float(row[key])
    return rows


def bootstrap(values: list[float], seed: int):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.median(x))
    if x.size == 1:
        return center, center, center
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(BOOTSTRAP_SAMPLES, x.size))
    med = np.median(x[idx], axis=1)
    lo, hi = np.quantile(med, [0.025, 0.975])
    return center, float(lo), float(hi)


def aggregate(rows, xkey, framework):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("family") == FAMILY and row.get(xkey) not in {"", None}:
            grouped[float(row[xkey])].append(row)

    points = []
    for point_index, (x, group) in enumerate(sorted(grouped.items())):
        values = [
            float(row[f"{framework}_s"])
            for row in group
            if row.get(f"{framework}_status") == "ok"
            and row.get(f"{framework}_s") not in {"", None}
        ]
        center, lo, hi = bootstrap(
            values,
            BOOTSTRAP_SEED
            + 1009 * point_index
            + (0 if framework == "knottedgraph" else 1),
        )
        points.append(
            {
                "x": x,
                "median": center,
                "ci_low": lo,
                "ci_high": hi,
                "n_ok": len(values),
                "n_total": len(group),
                "n_timeout": sum(
                    row.get(f"{framework}_status") == "timeout"
                    for row in group
                ),
            }
        )
    return points


def plot_crossing(rows, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    for framework, label, marker in (
        ("knottedgraph", "KnottedGraph", "o"),
        ("topoly", "Topoly", "s"),
    ):
        points = aggregate(rows, "crossings", framework)
        usable = [
            p for p in points
            if p["n_ok"] > 0 and np.isfinite(p["median"])
        ]
        if usable:
            xs = np.asarray([p["x"] for p in usable])
            ys = np.asarray([p["median"] for p in usable])
            lo = np.asarray([p["ci_low"] for p in usable])
            hi = np.asarray([p["ci_high"] for p in usable])
            ax.plot(
                xs, ys, marker=marker, linewidth=1.7, markersize=5,
                label=f"{label}: median across graph sizes",
            )
            ax.fill_between(
                xs, lo, hi, alpha=0.18, linewidth=0,
                label=f"{label}: 95% bootstrap CI",
            )

        censored = [
            p for p in points
            if p["n_ok"] == 0 and p["n_timeout"] > 0
        ]
        if censored:
            xs = np.asarray([p["x"] for p in censored])
            ys = np.asarray([
                max(
                    float(row["timeout_s"])
                    for row in rows
                    if row.get("family") == FAMILY
                    and float(row["crossings"]) == p["x"]
                )
                for p in censored
            ])
            ax.scatter(
                xs, ys, marker=marker, facecolors="none",
                label=f"{label}: fully censored",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Projected crossings, c")
    ax.set_ylabel("Yamada evaluation time (s)")
    ax.set_title(
        "Yamada scaling with crossings across heterogeneous connected trivalent graphs"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "topoly_vs_knottedgraph_crossings.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "topoly_vs_knottedgraph_crossings.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_size_slice(rows, *, size_crossings: int, xkey: str, xlabel: str, stem: str):
    selected = [
        row for row in rows
        if row.get("family") == FAMILY
        and int(float(row["crossings"])) == int(size_crossings)
    ]
    if not selected:
        raise ValueError(
            f"no rows found for size-scaling crossing slice c={size_crossings}"
        )

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    for framework, label, marker in (
        ("knottedgraph", "KnottedGraph", "o"),
        ("topoly", "Topoly", "s"),
    ):
        usable = [
            row for row in selected
            if row.get(f"{framework}_status") == "ok"
            and row.get(f"{framework}_s") not in {"", None}
        ]
        usable.sort(key=lambda row: float(row[xkey]))
        if usable:
            xs = np.asarray([float(row[xkey]) for row in usable])
            ys = np.asarray([float(row[f"{framework}_s"]) for row in usable])
            ax.plot(
                xs,
                ys,
                marker=marker,
                linewidth=1.7,
                markersize=5,
                label=label,
            )

        timed_out = [
            row for row in selected
            if row.get(f"{framework}_status") == "timeout"
        ]
        if timed_out:
            xs = np.asarray([float(row[xkey]) for row in timed_out])
            ys = np.asarray([float(row["timeout_s"]) for row in timed_out])
            ax.scatter(
                xs,
                ys,
                marker=marker,
                facecolors="none",
                label=f"{label}: timeout",
            )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Yamada evaluation time (s)")
    ax.set_title(
        f"Yamada input-size scaling at fixed projected crossings c={size_crossings}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_dir = Path(stem).parent
    filename = Path(stem).name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_aggregate(rows, output: Path, size_crossings: int) -> None:
    records = []
    for framework in ("knottedgraph", "topoly"):
        for point in aggregate(rows, "crossings", framework):
            records.append(
                {
                    "view": "crossings",
                    "fixed_crossings": "",
                    "xkey": "crossings",
                    "framework": framework,
                    **point,
                }
            )

    selected = [
        row for row in rows
        if row.get("family") == FAMILY
        and int(float(row["crossings"])) == int(size_crossings)
    ]
    for xkey in ("V", "E"):
        for framework in ("knottedgraph", "topoly"):
            for row in sorted(selected, key=lambda item: float(item[xkey])):
                records.append(
                    {
                        "view": xkey,
                        "fixed_crossings": size_crossings,
                        "xkey": xkey,
                        "framework": framework,
                        "x": float(row[xkey]),
                        "median": row.get(f"{framework}_s", ""),
                        "ci_low": "",
                        "ci_high": "",
                        "n_ok": int(row.get(f"{framework}_status") == "ok"),
                        "n_total": 1,
                        "n_timeout": int(row.get(f"{framework}_status") == "timeout"),
                    }
                )

    if not records:
        raise ValueError("no paper benchmark records available for aggregation")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(
    results_csv: Path,
    figure_dir: Path,
    aggregate_csv: Path,
    size_crossings: int,
):
    rows = read_rows(results_csv)
    families = {row.get("family") for row in rows}
    unexpected = families - {FAMILY}
    if unexpected:
        raise ValueError(
            f"unexpected benchmark families in paper CSV: {sorted(unexpected)}"
        )

    write_aggregate(rows, aggregate_csv, size_crossings)
    plot_crossing(rows, figure_dir)
    plot_size_slice(
        rows,
        size_crossings=size_crossings,
        xkey="V",
        xlabel="Graph vertices, V",
        stem=str(figure_dir / "topoly_vs_knottedgraph_vertices"),
    )
    plot_size_slice(
        rows,
        size_crossings=size_crossings,
        xkey="E",
        xlabel="Graph edges, E",
        stem=str(figure_dir / "topoly_vs_knottedgraph_edges"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--size-crossings", type=int, required=True)
    args = parser.parse_args()
    main(
        args.results_csv,
        args.figure_dir,
        args.aggregate_csv,
        args.size_crossings,
    )
