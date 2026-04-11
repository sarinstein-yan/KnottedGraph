import ctypes
import os
import json
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

from repulsive_layout import (
    LayoutArtifacts,
    LayoutResult,
    curve_network_from_graph,
    initial_layout,
    reconstruct_layout,
    write_curve_obj,
    write_metadata,
)


def _load_library(path: Path) -> ctypes.CDLL:
    dll_dir = path.resolve().parent
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))
        strawberry_bin = Path(r"C:\Strawberry\c\bin")
        if strawberry_bin.exists():
            os.add_dll_directory(str(strawberry_bin))
    lib = ctypes.CDLL(str(path.resolve()))
    lib.runRepulsiveGraphLayout.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.runRepulsiveGraphLayout.restype = ctypes.c_int
    return lib


def run_layout_ctypes(
    graph: nx.Graph,
    library_path: Path,
    *,
    steps: int = 50,
    seed: int = 0,
    jitter: float = 0.03,
    scale: float = 1.0,
    layout_mode: str = "auto",
    samples_per_edge: int = 16,
    bend_strength: float = 0.15,
    alpha: float = 3.0,
    beta: float = 6.0,
    use_sobolev: bool = True,
    use_multigrid: bool = False,
    use_barnes_hut: bool = False,
    use_backprojection: bool = True,
) -> LayoutResult:
    node_order = [str(node) for node in graph.nodes()]
    initial_positions = initial_layout(graph, seed=seed, jitter=jitter, scale=scale, mode=layout_mode)
    vertices, curve_edges, _node_index, edge_order, edge_vertex_indices = curve_network_from_graph(
        graph,
        initial_positions,
        samples_per_edge=samples_per_edge,
        bend_strength=bend_strength,
        bend_seed=seed,
    )

    positions = np.ascontiguousarray(vertices, dtype=np.float64)
    edges = np.ascontiguousarray(np.asarray(curve_edges, dtype=np.uintp), dtype=np.uintp)

    lib = _load_library(library_path)
    rc = lib.runRepulsiveGraphLayout(
        positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        positions.shape[0],
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        edges.shape[0],
        int(steps),
        float(alpha),
        float(beta),
        int(use_sobolev),
        int(use_multigrid),
        int(use_barnes_hut),
        int(use_backprojection),
    )
    if rc != 0:
        raise RuntimeError(f"runRepulsiveGraphLayout failed with code {rc}")

    return reconstruct_layout(graph, initial_positions, positions, node_order, edge_order, edge_vertex_indices)


def run_layout_ctypes_workspace(
    graph: nx.Graph,
    workspace: Path,
    library_path: Path,
    *,
    steps: int = 50,
    seed: int = 0,
    jitter: float = 0.03,
    scale: float = 1.0,
    layout_mode: str = "auto",
    samples_per_edge: int = 16,
    bend_strength: float = 0.15,
    alpha: float = 3.0,
    beta: float = 6.0,
    use_sobolev: bool = True,
    use_multigrid: bool = False,
    use_barnes_hut: bool = False,
    use_backprojection: bool = True,
) -> LayoutResult:
    workspace.mkdir(parents=True, exist_ok=True)
    result = run_layout_ctypes(
        graph,
        library_path=library_path,
        steps=steps,
        seed=seed,
        jitter=jitter,
        scale=scale,
        layout_mode=layout_mode,
        samples_per_edge=samples_per_edge,
        bend_strength=bend_strength,
        alpha=alpha,
        beta=beta,
        use_sobolev=use_sobolev,
        use_multigrid=use_multigrid,
        use_barnes_hut=use_barnes_hut,
        use_backprojection=use_backprojection,
    )

    node_order = result.node_order
    node_index = {node: i for i, node in enumerate(node_order)}
    edge_vertex_indices = {}
    vertices = []
    for node in node_order:
        node_index[node] = len(vertices)
        vertices.append(result.node_positions_final[node])
    for edge in result.edge_order:
        poly = result.edge_polylines_final[edge]
        start = node_index[edge[0]]
        idxs = [start]
        for pt in poly[1:-1]:
            idxs.append(len(vertices))
            vertices.append(pt)
        idxs.append(node_index[edge[1]])
        edge_vertex_indices[edge] = idxs

    final_curve_edges = []
    for edge in result.edge_order:
        idxs = edge_vertex_indices[edge]
        final_curve_edges.extend(list(zip(idxs[:-1], idxs[1:])))

    curve_path = workspace / "final.obj"
    metadata_path = workspace / "metadata.json"
    layout_json_path = workspace / "layout.json"
    write_curve_obj(curve_path, np.asarray(vertices, dtype=float), final_curve_edges)
    write_metadata(metadata_path, node_order, node_index, result.edge_order, edge_vertex_indices, samples_per_edge)

    result.artifacts = LayoutArtifacts(
        workspace=workspace,
        scene_path=workspace / "native_ctypes",
        curve_path=curve_path,
        final_curve_path=curve_path,
        metadata_path=metadata_path,
        layout_json_path=layout_json_path,
    )
    layout_json_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="utf-8")
    return result
