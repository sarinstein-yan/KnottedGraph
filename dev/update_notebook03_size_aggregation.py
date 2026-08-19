from __future__ import annotations

import json
from pathlib import Path

PATH = Path("User_guide/benchmarks/03_knottedgraph_vs_topoly_scaling.ipynb")


def main():
    nb = json.loads(PATH.read_text(encoding="utf-8"))
    by_id = {cell.get("id"): cell for cell in nb["cells"]}

    by_id["title"]["source"] = (
        "# 03 — KnottedGraph vs Topoly: focused paper scaling\n\n"
        "This notebook performs **one benchmark experiment** and reuses its results for three paper views:\n\n"
        "1. **Crossing scaling:** runtime versus projected crossing count $c$, aggregated across a fixed panel of heterogeneous connected trivalent graphs.\n"
        "2. **Vertex scaling:** runtime versus $V$, where every benchmark row having the same $V$ is pooled across the tested crossing grid.\n"
        "3. **Edge scaling:** runtime versus $E$, analogously pooling every row having the same $E$.\n\n"
        "No separate $K_4$, prism, edge-theta, throughput, or random-cubic benchmark is executed. The size views reuse the same sample-level timings, so no additional Yamada calculation is required.\n\n"
        "At each projected crossing count, the same panel of connected trivalent graphs is used. Graph construction and graph-to-PD conversion happen **outside** the timed region, and KnottedGraph and Topoly receive the same PD code.\n"
    )

    by_id["config-md"]["source"] = (
        "## 1. Configuration\n\n"
        "The key controls are directly selectable here.\n\n"
        "- `MAX_PROJECTED_CROSSINGS` is a **target threshold**. Crossing counts follow a pure doubling grid $1,2,4,8,\\ldots$ and stop at the first power of two greater than or equal to your target. Thus `500` gives a terminal crossing count of `512`.\n"
        "- `CROSSING_GRAPHS` controls how many different connected trivalent graph sizes are evaluated at each crossing count.\n"
        "- `SIZE_SCALING_CROSSINGS` is retained only as a benchmark-driver consistency point and must lie on the doubling grid; the publication $V$- and $E$-plots now average over **all** repeated rows with the same size and do not use a fixed-$c$ slice.\n\n"
        "For a normal local paper run the default is 21 graph sizes per $c$. GitHub Actions automatically uses a smaller smoke configuration.\n\n"
        "During execution, a persistent progress bar remains visible while each completed sample also prints KnottedGraph time, Topoly time, and speedup.\n"
    )

    by_id["accept-md"]["source"] = (
        "## 3. Acceptance checks and raw-data export\n\n"
        "The notebook verifies that only `crossings_graph_ensemble` was evaluated and that the same heterogeneous $(V,E)$ panel is reused across crossing counts. This repeated panel is what makes the later grouping by $V$ or $E$ comparable across the same crossing-complexity distribution.\n"
    )

    by_id["plot-md"]["source"] = (
        "## 4. Generate the three paper views\n\n"
        "The crossing plot aggregates across graph sizes at fixed $c$. The vertex and edge plots instead group the full sample-level dataset by $V$ or $E$ and compute an arithmetic mean over all available repeated rows at that size. Thus they use the complete tested crossing grid rather than one selected fixed-crossing slice. Timeout rows enter the size mean at the timeout threshold, so any size point containing a timeout is a conservative lower bound.\n"
    )

    plot_src = by_id["plot"]["source"]
    plot_src = plot_src.replace(
        '    "--size-crossings", str(SIZE_SCALING_CROSSINGS),\n',
        "",
    )
    by_id["plot"]["source"] = plot_src

    if "interpret" in by_id:
        by_id["interpret"]["source"] = (
            "## 5. Interpretation\n\n"
            "**Crossing view.** Each point summarizes the heterogeneous graph-size panel at a fixed projected crossing count.\n\n"
            "**Vertex/edge views.** Each point pools all repeated benchmark rows sharing the same $V$ or $E$, so the size dependence is averaged across the common tested crossing grid. Because every graph in this ensemble is trivalent, $E=3V/2$; therefore the vertex and edge panels are reparameterizations of the same size dependence and should not be presented as independent evidence.\n\n"
            "For size averages containing timeouts, the timeout threshold is inserted for the censored observation. The resulting arithmetic mean is therefore a lower bound on the true mean runtime.\n"
        )

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and cell.get("id") in {"plot"}:
            cell["execution_count"] = None
            cell["outputs"] = []

    PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated notebook 03 size-aggregation narrative and plot call")


if __name__ == "__main__":
    main()
