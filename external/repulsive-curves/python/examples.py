from pathlib import Path

import networkx as nx

from repulsive_layout import run_layout
from ctypes_layout import run_layout_ctypes_workspace


def make_k5() -> nx.Graph:
    return nx.complete_graph(5)


def make_k33() -> nx.Graph:
    return nx.complete_bipartite_graph(3, 3)


def run_builtin_examples(solver: Path, root: Path, engine: str = "subprocess", steps: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)

    runner = run_layout if engine == "subprocess" else run_layout_ctypes_workspace

    if engine == "subprocess":
        runner(
            make_k5(),
            workspace=root / "k5",
            solver=solver,
            samples_per_edge=16,
            steps=steps,
            seed=0,
            jitter=0.03,
            layout_mode="circular",
            bend_strength=0.18,
        )
        runner(
            make_k33(),
            workspace=root / "k33",
            solver=solver,
            samples_per_edge=16,
            steps=steps,
            seed=1,
            jitter=0.03,
            layout_mode="bipartite",
            bend_strength=0.18,
        )
    else:
        runner(
            make_k5(),
            workspace=root / "k5",
            library_path=solver,
            samples_per_edge=16,
            steps=steps,
            seed=0,
            jitter=0.03,
            layout_mode="circular",
            bend_strength=0.18,
        )
        runner(
            make_k33(),
            workspace=root / "k33",
            library_path=solver,
            samples_per_edge=16,
            steps=steps,
            seed=1,
            jitter=0.03,
            layout_mode="bipartite",
            bend_strength=0.18,
        )
