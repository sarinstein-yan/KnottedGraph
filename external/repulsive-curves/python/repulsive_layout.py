import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np


ArrayLike3 = Sequence[float]
EdgeKey = Tuple[str, str]


@dataclass
class LayoutArtifacts:
    workspace: Path
    scene_path: Path
    curve_path: Path
    final_curve_path: Path
    metadata_path: Path
    layout_json_path: Path


@dataclass
class LayoutResult:
    graph: nx.Graph
    node_order: List[str]
    edge_order: List[EdgeKey]
    node_positions_initial: Dict[str, np.ndarray]
    node_positions_final: Dict[str, np.ndarray]
    edge_polylines_final: Dict[EdgeKey, np.ndarray]
    artifacts: LayoutArtifacts

    def to_json_dict(self) -> dict:
        return {
            "node_order": self.node_order,
            "edge_order": [[u, v] for u, v in self.edge_order],
            "node_positions_initial": {k: self.node_positions_initial[k].tolist() for k in self.node_order},
            "node_positions_final": {k: self.node_positions_final[k].tolist() for k in self.node_order},
            "edge_polylines_final": {
                f"{u}::{v}": self.edge_polylines_final[(u, v)].tolist() for (u, v) in self.edge_order
            },
            "artifacts": {
                "workspace": str(self.artifacts.workspace),
                "scene_path": str(self.artifacts.scene_path),
                "curve_path": str(self.artifacts.curve_path),
                "final_curve_path": str(self.artifacts.final_curve_path),
                "metadata_path": str(self.artifacts.metadata_path),
                "layout_json_path": str(self.artifacts.layout_json_path),
            },
        }


def load_layout_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_graph(path: Path) -> nx.Graph:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "edges" in data:
            graph = nx.Graph()
            graph.add_nodes_from([str(node) for node in data.get("nodes", [])])
            graph.add_edges_from([(str(u), str(v)) for u, v in data["edges"]])
            return graph
        raise ValueError("JSON input must contain an 'edges' field")
    if path.suffix.lower() in {".edgelist", ".txt"}:
        return nx.read_edgelist(path, nodetype=str)
    if path.suffix.lower() == ".graphml":
        graph = nx.read_graphml(path)
        return nx.relabel_nodes(graph, lambda x: str(x))
    raise ValueError(f"Unsupported graph format: {path.suffix}")


def _circular_layout_2d(graph: nx.Graph) -> Dict[str, np.ndarray]:
    raw = nx.circular_layout(graph, dim=2)
    return {str(k): np.array([float(v[0]), float(v[1])], dtype=float) for k, v in raw.items()}


def _spring_layout_2d(graph: nx.Graph, seed: int) -> Dict[str, np.ndarray]:
    raw = nx.spring_layout(graph, dim=2, seed=seed)
    return {str(k): np.array([float(v[0]), float(v[1])], dtype=float) for k, v in raw.items()}


def _bipartite_layout_2d(graph: nx.Graph) -> Dict[str, np.ndarray]:
    left, right = nx.algorithms.bipartite.sets(graph)
    raw = nx.bipartite_layout(graph, nodes=left, align="vertical")
    return {str(k): np.array([float(v[0]), float(v[1])], dtype=float) for k, v in raw.items()}


def initial_layout(
    graph: nx.Graph, seed: int, jitter: float, scale: float = 1.0, mode: str = "auto"
) -> Dict[str, np.ndarray]:
    if mode == "auto":
        layout2d = _bipartite_layout_2d(graph) if nx.is_bipartite(graph) else _spring_layout_2d(graph, seed)
    elif mode == "spring":
        layout2d = _spring_layout_2d(graph, seed)
    elif mode == "circular":
        layout2d = _circular_layout_2d(graph)
    elif mode == "bipartite":
        layout2d = _bipartite_layout_2d(graph)
    else:
        raise ValueError(f"Unsupported layout mode: {mode}")

    rng = random.Random(seed)
    pos3d: Dict[str, np.ndarray] = {}
    for node, xy in layout2d.items():
        x, y = float(xy[0]) * scale, float(xy[1]) * scale
        z = rng.uniform(-jitter, jitter) * scale
        pos3d[str(node)] = np.array([x, y, z], dtype=float)
    return pos3d


def _edge_bend_direction(p0: np.ndarray, p1: np.ndarray, seed_value: int) -> np.ndarray:
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    direction = direction / norm

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(direction, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    n1 = np.cross(direction, reference)
    n1_norm = np.linalg.norm(n1)
    if n1_norm < 1e-12:
        n1 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n1 = n1 / n1_norm

    n2 = np.cross(direction, n1)
    n2 = n2 / max(np.linalg.norm(n2), 1e-12)

    rng = random.Random(seed_value)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    bend = math.cos(theta) * n1 + math.sin(theta) * n2
    return bend / max(np.linalg.norm(bend), 1e-12)


def _normalize_edge(u: str, v: str) -> EdgeKey:
    return (u, v) if u <= v else (v, u)


def curve_network_from_graph(
    graph: nx.Graph,
    pos3d: Dict[str, np.ndarray],
    samples_per_edge: int,
    bend_strength: float = 0.15,
    bend_seed: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[str, int], List[EdgeKey], Dict[EdgeKey, List[int]]]:
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least 2")

    vertices: List[np.ndarray] = []
    edges: List[Tuple[int, int]] = []
    node_index: Dict[str, int] = {}
    edge_order: List[EdgeKey] = []
    edge_vertex_indices: Dict[EdgeKey, List[int]] = {}

    for node in graph.nodes():
        node_str = str(node)
        node_index[node_str] = len(vertices)
        vertices.append(pos3d[node_str])

    for u, v in graph.edges():
        u_str, v_str = str(u), str(v)
        edge_key = _normalize_edge(u_str, v_str)
        edge_order.append(edge_key)

        start_idx = node_index[u_str]
        end_idx = node_index[v_str]
        prev_idx = start_idx
        p0 = pos3d[u_str]
        p1 = pos3d[v_str]
        bend_dir = _edge_bend_direction(p0, p1, hash((edge_key, bend_seed)))
        bend_amp = bend_strength * float(np.linalg.norm(p1 - p0))
        polyline_indices = [start_idx]

        for i in range(1, samples_per_edge):
            t = i / samples_per_edge
            p = (1.0 - t) * p0 + t * p1
            p = p + bend_amp * math.sin(math.pi * t) * bend_dir
            cur_idx = len(vertices)
            vertices.append(p)
            edges.append((prev_idx, cur_idx))
            polyline_indices.append(cur_idx)
            prev_idx = cur_idx

        edges.append((prev_idx, end_idx))
        polyline_indices.append(end_idx)
        edge_vertex_indices[edge_key] = polyline_indices

    return np.asarray(vertices, dtype=float), edges, node_index, edge_order, edge_vertex_indices


def write_curve_obj(path: Path, vertices: np.ndarray, edges: Iterable[Tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for i, j in edges:
            f.write(f"l {i + 1} {j + 1}\n")


def write_scene(path: Path, curve_filename: str, steps: int) -> None:
    scene = [
        f"curve {curve_filename}",
        "repel_curve 3 6",
        "fix_barycenter",
        "fix_edgelengths",
        f"iteration_limit {steps}",
    ]
    path.write_text("\n".join(scene) + "\n", encoding="utf-8")


def write_metadata(
    path: Path,
    node_order: List[str],
    node_index: Dict[str, int],
    edge_order: List[EdgeKey],
    edge_vertex_indices: Dict[EdgeKey, List[int]],
    samples_per_edge: int,
) -> None:
    payload = {
        "node_order": node_order,
        "node_index": node_index,
        "edge_order": [[u, v] for u, v in edge_order],
        "edge_vertex_indices": {f"{u}::{v}": idxs for (u, v), idxs in edge_vertex_indices.items()},
        "samples_per_edge": samples_per_edge,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_curve_obj(path: Path) -> np.ndarray:
    vertices: List[List[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append([float(x), float(y), float(z)])
    return np.asarray(vertices, dtype=float)


def reconstruct_layout(
    graph: nx.Graph,
    initial_positions: Dict[str, np.ndarray],
    final_vertices: np.ndarray,
    node_order: List[str],
    edge_order: List[EdgeKey],
    edge_vertex_indices: Dict[EdgeKey, List[int]],
) -> LayoutResult:
    node_positions_final = {node: final_vertices[idx] for idx, node in enumerate(node_order)}
    edge_polylines_final = {edge: final_vertices[idxs] for edge, idxs in edge_vertex_indices.items()}

    workspace = Path(".")
    dummy = LayoutArtifacts(workspace, workspace, workspace, workspace, workspace, workspace)
    return LayoutResult(
        graph=graph,
        node_order=node_order,
        edge_order=edge_order,
        node_positions_initial=initial_positions,
        node_positions_final=node_positions_final,
        edge_polylines_final=edge_polylines_final,
        artifacts=dummy,
    )


def run_layout(
    graph: nx.Graph,
    workspace: Path,
    solver: Path,
    *,
    samples_per_edge: int = 16,
    steps: int = 300,
    seed: int = 0,
    jitter: float = 0.03,
    scale: float = 1.0,
    layout_mode: str = "auto",
    bend_strength: float = 0.15,
    extra_solver_args: Sequence[str] = (),
) -> LayoutResult:
    workspace.mkdir(parents=True, exist_ok=True)

    node_order = [str(node) for node in graph.nodes()]
    initial_positions = initial_layout(graph, seed=seed, jitter=jitter, scale=scale, mode=layout_mode)
    vertices, curve_edges, node_index, edge_order, edge_vertex_indices = curve_network_from_graph(
        graph, initial_positions, samples_per_edge, bend_strength=bend_strength, bend_seed=seed
    )

    curve_path = workspace / "curve.obj"
    scene_path = workspace / "scene.txt"
    final_curve_path = workspace / "final.obj"
    metadata_path = workspace / "metadata.json"
    layout_json_path = workspace / "layout.json"

    write_curve_obj(curve_path, vertices, curve_edges)
    write_scene(scene_path, curve_path.name, steps)
    write_metadata(metadata_path, node_order, node_index, edge_order, edge_vertex_indices, samples_per_edge)

    cmd = [str(solver), str(scene_path), "--output", str(final_curve_path), "--steps", str(steps), *extra_solver_args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    final_vertices = read_curve_obj(final_curve_path)
    result = reconstruct_layout(graph, initial_positions, final_vertices, node_order, edge_order, edge_vertex_indices)
    result.artifacts = LayoutArtifacts(
        workspace=workspace,
        scene_path=scene_path,
        curve_path=curve_path,
        final_curve_path=final_curve_path,
        metadata_path=metadata_path,
        layout_json_path=layout_json_path,
    )
    layout_json_path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="utf-8")
    return result
