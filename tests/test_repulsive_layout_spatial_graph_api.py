from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np


repo_root = Path(__file__).resolve().parents[1]
kg_pkg = types.ModuleType("knotted_graph")
kg_pkg.__path__ = [str(repo_root / "src" / "knotted_graph")]
sys.modules.setdefault("knotted_graph", kg_pkg)
layout_pkg = types.ModuleType("knotted_graph.layout")
layout_pkg.__path__ = [str(repo_root / "src" / "knotted_graph" / "layout")]
sys.modules.setdefault("knotted_graph.layout", layout_pkg)
repulsive_pkg = types.ModuleType("knotted_graph.layout.repulsive")
repulsive_pkg.__path__ = [str(repo_root / "src" / "knotted_graph" / "layout" / "repulsive")]
sys.modules.setdefault("knotted_graph.layout.repulsive", repulsive_pkg)

from knotted_graph.layout.repulsive import pipeline


def _read_curve(path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = int(lines[0].split()[1])
    vertices = np.asarray(
        [[float(x) for x in line.split()] for line in lines[1 : 1 + vertex_count]],
        dtype=float,
    )
    edge_header = 1 + vertex_count
    edge_count = int(lines[edge_header].split()[1])
    edges = [
        tuple(int(x) for x in line.split())
        for line in lines[edge_header + 1 : edge_header + 1 + edge_count]
    ]
    return vertices, edges


def _write_obj(path: Path, vertices: np.ndarray, edges: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b in edges:
            f.write(f"l {a + 1} {b + 1}\n")


def _write_history(path: Path) -> None:
    path.write_text(
        "accepted,margin,step_size,safe_t,topology_enabled,topology_safe,"
        "topology_min_distance,topology_rejections,energy_before,energy_after\n"
        "1,0.5,0.5,1.0,1,1,0.25,0,10.0,9.0\n",
        encoding="utf-8",
    )


def test_relax_spatial_graph_preserves_skeleton_graph_api(monkeypatch, tmp_path):
    graph = nx.MultiGraph()
    graph.graph["is_trivalent"] = True
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]), residue="A")
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]), residue="B")
    graph.add_edge(
        "u",
        "v",
        key="upper",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]),
        color="red",
    )
    graph.add_edge(
        "u",
        "v",
        key="lower",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [2.0, 0.0, 0.0]]),
        color="blue",
    )

    def fake_build_driver(config, force=False):
        return tmp_path / "fake_driver"

    def fake_run_driver(
        input_curve,
        output_obj,
        history_csv,
        options,
        config,
        save_steps_dir=None,
        pinned_vertices=None,
    ):
        vertices, edges = _read_curve(input_curve)
        pinned = set()
        if pinned_vertices is not None:
            pinned = {
                int(line)
                for line in Path(pinned_vertices).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        for index in range(len(vertices)):
            if index not in pinned:
                vertices[index, 2] += 0.25
        _write_obj(output_obj, vertices, edges)
        _write_history(history_csv)
        return subprocess.CompletedProcess(["fake"], 0, "fake driver\n", "")

    monkeypatch.setattr(pipeline, "build_driver", fake_build_driver)
    monkeypatch.setattr(pipeline, "run_driver", fake_run_driver)

    result = pipeline.relax_spatial_graph(
        graph,
        tmp_path / "layout",
        solver_options=pipeline.SolverOptions(steps=1),
        driver_config=pipeline.DriverConfig(driver_binary=tmp_path / "fake_driver", verbose=False),
        save_steps=False,
        simplify_after_layout=False,
        verify_topology=False,
    )

    relaxed = result.graph

    assert isinstance(relaxed, nx.MultiGraph)
    assert set(relaxed.nodes) == {"u", "v"}
    assert set(relaxed.edges(keys=True)) == {
        ("u", "v", "upper"),
        ("u", "v", "lower"),
    }
    assert relaxed.graph["is_trivalent"] is True
    assert relaxed.nodes["u"]["residue"] == "A"
    assert relaxed.edges["u", "v", "upper"]["color"] == "red"
    assert relaxed.edges["u", "v", "lower"]["color"] == "blue"

    np.testing.assert_allclose(relaxed.nodes["u"]["pos"], graph.nodes["u"]["pos"])
    np.testing.assert_allclose(relaxed.nodes["v"]["pos"], graph.nodes["v"]["pos"])
    assert relaxed.edges["u", "v", "upper"]["pts"][1, 2] == 0.25
    assert relaxed.edges["u", "v", "lower"]["pts"][1, 2] == 0.25

    np.testing.assert_allclose(graph.edges["u", "v", "upper"]["pts"][1], [1.0, 1.0, 0.0])
    assert result.metadata["parameters"]["pin_graph_nodes"] is True
    assert result.metadata["certificate"]["valid"] is True


def test_relax_spatial_graph_fast_defaults_keep_driver_topology_check(monkeypatch, tmp_path):
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([2.0, 0.0, 0.0]))
    graph.add_edge(
        "u",
        "v",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    captured = {}

    def fake_build_driver(config, force=False):
        return tmp_path / "fake_driver"

    def fake_run_driver(
        input_curve,
        output_obj,
        history_csv,
        options,
        config,
        save_steps_dir=None,
        pinned_vertices=None,
    ):
        captured["topology_check"] = options.topology_check
        captured["save_steps_dir"] = save_steps_dir
        vertices, edges = _read_curve(input_curve)
        _write_obj(output_obj, vertices, edges)
        _write_history(history_csv)
        return subprocess.CompletedProcess(["fake"], 0, "fake driver\n", "")

    monkeypatch.setattr(pipeline, "build_driver", fake_build_driver)
    monkeypatch.setattr(pipeline, "run_driver", fake_run_driver)

    result = pipeline.relax_spatial_graph(
        graph,
        tmp_path / "layout",
        solver_options=pipeline.SolverOptions(steps=1),
        driver_config=pipeline.DriverConfig(driver_binary=tmp_path / "fake_driver", verbose=False),
        simplify_after_layout=False,
    )

    assert captured["topology_check"] is True
    assert captured["save_steps_dir"] is None
    assert result.metadata["parameters"]["verify_topology"] is False
    assert result.metadata["steps_dir"] is None
    assert result.metadata["topology_verification"] is None
