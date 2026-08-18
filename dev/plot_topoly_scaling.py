from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "size", "embedding", "sample", "V", "E", "crossings", "pd_length",
            "knottedgraph_s", "topoly_s", "timeout_s", "topoly_over_kg",
        ):
            if key in row and row[key] not in {"", None}:
                row[key] = float(row[key])
    return rows


def _bootstrap_ci(values: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.median(values))
    if values.size == 1:
        return center, center, center
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    boot = np.median(values[indices], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return center, float(lo), float(hi)


def aggregate(rows: list[dict], family: str, xkey: str, framework: str) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        if row["family"] == family and xkey in row and row[xkey] not in {"", None}:
            grouped[float(row[xkey])].append(row)

    out = []
    for point_index, (x, group) in enumerate(sorted(grouped.items())):
        ok_values = np.asarray(
            [
                float(row[f"{framework}_s"])
                for row in group
                if row.get(f"{framework}_status") == "ok"
                and row.get(f"{framework}_s") not in {"", None}
            ],
            dtype=float,
        )
        median, lo, hi = _bootstrap_ci(
            ok_values,
            seed=BOOTSTRAP_SEED + 1009 * point_index + (0 if framework == "knottedgraph" else 1),
        )
        timeout_n = sum(row.get(f"{framework}_status") == "timeout" for row in group)
        error_n = sum(row.get(f"{framework}_status") == "error" for row in group)
        skipped_n = sum(
            row.get(f"{framework}_status") == "skipped_after_censor_frontier"
            for row in group
        )
        sample_kind = next(
            (row.get("sample_kind") for row in group if row.get("sample_kind")),
            "embedding",
        )
        out.append(
            {
                "x": x,
                "sample_kind": sample_kind,
                "median": median,
                "ci_low": lo,
                "ci_high": hi,
                "n_ok": int(ok_values.size),
                "n_total": len(group),
                "n_timeout": timeout_n,
                "n_error": error_n,
                "n_skipped": skipped_n,
            }
        )
    return out


def _fit(points: list[dict]) -> dict:
    usable = [
        p for p in points
        if p["n_ok"] >= 5
        and np.isfinite(p["median"])
        and p["median"] > 0
        and p["x"] > 0
    ]
    if len(usable) < 3:
        return {}
    x = np.asarray([p["x"] for p in usable], dtype=float)
    y = np.asarray([p["median"] for p in usable], dtype=float)
    lx, ly = np.log(x), np.log(y)
    alpha, log_cp = np.polyfit(lx, ly, 1)
    beta, log_ce = np.polyfit(x, ly, 1)

    def r2(actual, predicted):
        denom = np.sum((actual - actual.mean()) ** 2)
        return 1 - np.sum((actual - predicted) ** 2) / denom if denom else 1.0

    return {
        "alpha": float(alpha),
        "Cp": float(np.exp(log_cp)),
        "power_r2": float(r2(ly, log_cp + alpha * lx)),
        "beta": float(beta),
        "Ce": float(np.exp(log_ce)),
        "exp_r2": float(r2(ly, log_ce + beta * x)),
        "xmin": float(x.min()),
        "xmax": float(x.max()),
    }


def plot_family(
    rows: list[dict],
    family: str,
    xkey: str,
    xlabel: str,
    title: str,
    stem: str,
    output_dir: Path,
    *,
    logx: bool = False,
) -> dict[str, dict]:
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    reports = {}

    for framework, label, marker in [
        ("knottedgraph", "KnottedGraph", "o"),
        ("topoly", "Topoly", "s"),
    ]:
        points = aggregate(rows, family, xkey, framework)
        usable = [p for p in points if p["n_ok"] > 0 and np.isfinite(p["median"])]
        if usable:
            xs = np.asarray([p["x"] for p in usable])
            ys = np.asarray([p["median"] for p in usable])
            lo = np.asarray([p["ci_low"] for p in usable])
            hi = np.asarray([p["ci_high"] for p in usable])
            sample_kind = usable[0].get("sample_kind", "embedding")
            center_label = (
                f"{label}: median across graph instances"
                if sample_kind == "topology"
                else f"{label}: median across embeddings"
            )
            ax.plot(
                xs,
                ys,
                marker=marker,
                linewidth=1.7,
                markersize=5,
                label=center_label,
            )
            ax.fill_between(
                xs,
                lo,
                hi,
                alpha=0.18,
                linewidth=0,
                label=f"{label}: 95% bootstrap CI",
            )

        censored = [p for p in points if p["n_ok"] == 0 and p["n_timeout"] > 0]
        if censored:
            xs = np.asarray([p["x"] for p in censored])
            ys = np.asarray([
                max(
                    float(row["timeout_s"])
                    for row in rows
                    if row["family"] == family and float(row[xkey]) == p["x"]
                )
                for p in censored
            ])
            ax.scatter(
                xs,
                ys,
                marker=marker,
                facecolors="none",
                label=f"{label}: fully censored",
            )

        report = _fit(points)
        reports[framework] = report
        if report:
            xx = (
                np.geomspace(report["xmin"], report["xmax"], 300)
                if logx
                else np.linspace(report["xmin"], report["xmax"], 300)
            )
            yy = report["Cp"] * xx ** report["alpha"]
            ax.plot(
                xx,
                yy,
                linestyle="--",
                linewidth=1.2,
                label=(
                    f"{label} power fit: α={report['alpha']:.2f}, "
                    f"R²={report['power_r2']:.3f}"
                ),
            )

    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Yamada evaluation time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return reports


def _specs():
    return [
        ("crossings_fixed", "crossings", "Projected crossings, c", "Yamada scaling with crossings at fixed V=2, E=3", "topoly_vs_knottedgraph_crossings_fixed", False),
        ("crossings_throughput", "crossings", "Projected crossings, c", "Large-diagram Yamada throughput versus crossing count", "topoly_vs_knottedgraph_crossings_throughput", False),
        ("edges_theta", "E", "Graph edges, E", "Yamada edge scaling at fixed V=2, c=0", "topoly_vs_knottedgraph_edges", True),
        ("vertices_k4", "V", "Graph vertices, V", "Trivalent Yamada input-size scaling using planar K4 components", "topoly_vs_knottedgraph_vertices_k4", True),
        ("connected_prism", "V", "Graph vertices, V", "Connected trivalent Yamada scaling with V", "topoly_vs_knottedgraph_prism_V", True),
        ("connected_prism", "E", "Graph edges, E", "Connected trivalent Yamada scaling with E", "topoly_vs_knottedgraph_prism_E", True),
        (
            "random_cubic",
            "V",
            "Graph vertices, V",
            "Yamada scaling across non-isomorphic connected cubic graphs",
            "topoly_vs_knottedgraph_random_cubic_V",
            True,
        ),
    ]


def write_aggregate_csv(rows: list[dict], output: Path) -> None:
    records = []
    for family, xkey, *_ in _specs():
        if not any(row.get("family") == family for row in rows):
            continue
        for framework in ("knottedgraph", "topoly"):
            for point in aggregate(rows, family, xkey, framework):
                records.append(
                    {"family": family, "xkey": xkey, "framework": framework, **point}
                )
    if not records:
        raise ValueError("no benchmark records available for aggregation")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(results_csv: Path, figure_dir: Path, aggregate_csv: Path) -> None:
    rows = read_rows(results_csv)
    write_aggregate_csv(rows, aggregate_csv)
    for family, xkey, xlabel, title, stem, logx in _specs():
        if not any(row.get("family") == family for row in rows):
            continue
        reports = plot_family(
            rows,
            family,
            xkey,
            xlabel,
            title,
            stem,
            figure_dir,
            logx=logx,
        )
        print(f"[{family}/{xkey}]")
        for framework, report in reports.items():
            if report:
                print(
                    f"{framework}: power alpha={report['alpha']:.4g}, "
                    f"R2={report['power_r2']:.4f}; exp beta={report['beta']:.4g}, "
                    f"R2={report['exp_r2']:.4f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    args = parser.parse_args()
    main(args.results_csv, args.figure_dir, args.aggregate_csv)
