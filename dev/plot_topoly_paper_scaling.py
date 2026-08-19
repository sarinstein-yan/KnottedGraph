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
FRAMEWORKS = (
    ("knottedgraph", "KnottedGraph", "o", "-"),
    ("topoly", "Topoly", "s", "--"),
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "size",
            "sample",
            "V",
            "E",
            "crossings",
            "knottedgraph_s",
            "topoly_s",
            "timeout_s",
        ):
            if row.get(key) not in {"", None}:
                row[key] = float(row[key])
    return rows


def bootstrap_interval(
    values: list[float],
    *,
    statistic: str,
    seed: int,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan

    if statistic == "median":
        stat = np.median
    elif statistic == "mean":
        stat = np.mean
    else:
        raise ValueError(f"unsupported statistic: {statistic}")

    center = float(stat(x))
    if x.size == 1:
        return center, center, center

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(BOOTSTRAP_SAMPLES, x.size))
    boot = stat(x[idx], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return center, float(lo), float(hi)


def aggregate_crossings(rows: list[dict], framework: str) -> list[dict]:
    """Median runtime at each c across the heterogeneous graph-size panel."""
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("family") == FAMILY and row.get("crossings") not in {"", None}:
            grouped[float(row["crossings"])].append(row)

    points = []
    for point_index, (x, group) in enumerate(sorted(grouped.items())):
        values = [
            float(row[f"{framework}_s"])
            for row in group
            if row.get(f"{framework}_status") == "ok"
            and row.get(f"{framework}_s") not in {"", None}
        ]
        center, lo, hi = bootstrap_interval(
            values,
            statistic="median",
            seed=BOOTSTRAP_SEED
            + 1009 * point_index
            + (0 if framework == "knottedgraph" else 1),
        )
        points.append(
            {
                "x": x,
                "estimate": center,
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


def _size_value(row: dict, framework: str) -> tuple[float | None, bool]:
    """Return a runtime value and whether it is timeout-derived.

    Successful timings enter exactly. A timeout enters at the timeout threshold,
    so the arithmetic mean over all repeated rows is a lower bound on the true
    mean whenever at least one timeout occurs. Errors/skips remain unavailable
    rather than being assigned fabricated timings.
    """
    status = row.get(f"{framework}_status")
    if status == "ok" and row.get(f"{framework}_s") not in {"", None}:
        return float(row[f"{framework}_s"]), False
    if status == "timeout" and row.get("timeout_s") not in {"", None}:
        return float(row["timeout_s"]), True
    return None, False


def aggregate_size(rows: list[dict], xkey: str, framework: str) -> list[dict]:
    """Average every repeated row having the same V or E.

    The same crossing grid is reused at every graph size, so grouping by V or E
    averages over the same crossing-complexity distribution at every size.
    Timeout observations are inserted at the timeout threshold; therefore any
    point with n_timeout > 0 is a conservative lower-bound mean.
    """
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("family") == FAMILY and row.get(xkey) not in {"", None}:
            grouped[float(row[xkey])].append(row)

    points = []
    for point_index, (x, group) in enumerate(sorted(grouped.items())):
        values: list[float] = []
        timeout_flags: list[bool] = []
        for row in group:
            value, is_timeout = _size_value(row, framework)
            if value is not None:
                values.append(value)
                timeout_flags.append(is_timeout)

        center, lo, hi = bootstrap_interval(
            values,
            statistic="mean",
            seed=BOOTSTRAP_SEED
            + 2003 * point_index
            + (0 if framework == "knottedgraph" else 1),
        )
        points.append(
            {
                "x": x,
                "estimate": center,
                "ci_low": lo,
                "ci_high": hi,
                "n_ok": sum(
                    row.get(f"{framework}_status") == "ok"
                    for row in group
                ),
                "n_total": len(group),
                "n_timeout": sum(timeout_flags),
                "n_used": len(values),
                "n_missing": len(group) - len(values),
            }
        )
    return points


def _plot_points(ax, points: list[dict], *, label: str, marker: str, linestyle: str):
    usable = [
        p for p in points
        if np.isfinite(p["estimate"]) and p.get("n_used", p["n_ok"]) > 0
    ]
    if not usable:
        return

    xs = np.asarray([p["x"] for p in usable], dtype=float)
    ys = np.asarray([p["estimate"] for p in usable], dtype=float)
    lo = np.asarray([p["ci_low"] for p in usable], dtype=float)
    hi = np.asarray([p["ci_high"] for p in usable], dtype=float)

    (line,) = ax.plot(
        xs,
        ys,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        markersize=5.5,
        label=label,
    )
    ax.fill_between(
        xs,
        lo,
        hi,
        alpha=0.16,
        linewidth=0,
        color=line.get_color(),
    )

    censored = [p for p in usable if p.get("n_timeout", 0) > 0]
    if censored:
        ax.scatter(
            [p["x"] for p in censored],
            [p["estimate"] for p in censored],
            marker=marker,
            facecolors="none",
            edgecolors=line.get_color(),
            s=52,
            linewidths=1.25,
            zorder=4,
        )


def _style_axis(ax):
    ax.set_yscale("log")
    ax.grid(alpha=0.18, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.tick_params(direction="in", width=0.8, length=4)


def plot_crossing(rows: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for framework, label, marker, linestyle in FRAMEWORKS:
        points = aggregate_crossings(rows, framework)
        _plot_points(
            ax,
            points,
            label=f"{label}: median across graph sizes",
            marker=marker,
            linestyle=linestyle,
        )

        fully_censored = [
            p for p in points if p["n_ok"] == 0 and p["n_timeout"] > 0
        ]
        if fully_censored:
            timeout_s = max(
                float(row["timeout_s"])
                for row in rows
                if row.get("family") == FAMILY
                and row.get(f"{framework}_status") == "timeout"
            )
            ax.scatter(
                [p["x"] for p in fully_censored],
                [timeout_s for _ in fully_censored],
                marker=marker,
                facecolors="none",
                s=55,
                linewidths=1.25,
                label=f"{label}: timeout lower bound",
            )

    _style_axis(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Projected crossings, $c$")
    ax.set_ylabel("Yamada evaluation time (s)")
    ax.set_title("Crossing scaling across heterogeneous connected trivalent graphs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "topoly_vs_knottedgraph_crossings.png",
        dpi=500,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "topoly_vs_knottedgraph_crossings.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_size_average(
    rows: list[dict],
    *,
    xkey: str,
    xlabel: str,
    output_dir: Path,
    stem: str,
):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for framework, label, marker, linestyle in FRAMEWORKS:
        points = aggregate_size(rows, xkey, framework)
        _plot_points(
            ax,
            points,
            label=f"{label}: mean over crossing levels",
            marker=marker,
            linestyle=linestyle,
        )

    _style_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Yamada evaluation time (s)")
    ax.set_title(
        f"Size scaling averaged across all tested crossing levels ({xkey})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=500, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_aggregate(rows: list[dict], output: Path) -> None:
    records: list[dict] = []

    for framework in ("knottedgraph", "topoly"):
        for point in aggregate_crossings(rows, framework):
            records.append(
                {
                    "view": "crossings",
                    "xkey": "crossings",
                    "framework": framework,
                    "x": point["x"],
                    "statistic": "median",
                    "estimate": point["estimate"],
                    "ci_low": point["ci_low"],
                    "ci_high": point["ci_high"],
                    "n_ok": point["n_ok"],
                    "n_total": point["n_total"],
                    "n_timeout": point["n_timeout"],
                    "censoring": (
                        "timeouts excluded; fully censored points shown at "
                        "timeout lower bound"
                    ),
                }
            )

    for xkey in ("V", "E"):
        for framework in ("knottedgraph", "topoly"):
            for point in aggregate_size(rows, xkey, framework):
                records.append(
                    {
                        "view": xkey,
                        "xkey": xkey,
                        "framework": framework,
                        "x": point["x"],
                        "statistic": "mean",
                        "estimate": point["estimate"],
                        "ci_low": point["ci_low"],
                        "ci_high": point["ci_high"],
                        "n_ok": point["n_ok"],
                        "n_total": point["n_total"],
                        "n_timeout": point["n_timeout"],
                        "censoring": (
                            "timeouts inserted at timeout_s, so estimate is a "
                            "lower bound on the true mean"
                        ),
                    }
                )

    if not records:
        raise ValueError("no paper benchmark records available for aggregation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(results_csv: Path, figure_dir: Path, aggregate_csv: Path):
    rows = read_rows(results_csv)
    families = {row.get("family") for row in rows}
    unexpected = families - {FAMILY}
    if unexpected:
        raise ValueError(
            f"unexpected benchmark families in paper CSV: {sorted(unexpected)}"
        )

    write_aggregate(rows, aggregate_csv)
    plot_crossing(rows, figure_dir)
    plot_size_average(
        rows,
        xkey="V",
        xlabel="Graph vertices, $V$",
        output_dir=figure_dir,
        stem="topoly_vs_knottedgraph_vertices",
    )
    plot_size_average(
        rows,
        xkey="E",
        xlabel="Graph edges, $E$",
        output_dir=figure_dir,
        stem="topoly_vs_knottedgraph_edges",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    # Backward-compatible no-op so older notebook copies do not fail.
    parser.add_argument(
        "--size-crossings",
        type=int,
        required=False,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    main(args.results_csv, args.figure_dir, args.aggregate_csv)
