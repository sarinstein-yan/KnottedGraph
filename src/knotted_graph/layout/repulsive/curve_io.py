from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import CurveNetwork


def network_to_vertices(network: CurveNetwork) -> tuple[np.ndarray, dict[str, list[int]]]:
    if len(network.node_order) != 2:
        raise ValueError("The current Repulsor protein examples expect exactly two graph nodes.")

    vertices: list[np.ndarray] = [
        np.asarray(network.node_positions[network.node_order[0]], dtype=float),
        np.asarray(network.node_positions[network.node_order[1]], dtype=float),
    ]
    arc_indices: dict[str, list[int]] = {}

    for arc_name in network.arc_order:
        polyline = np.asarray(network.arc_polylines[arc_name], dtype=float)
        indices = [0]
        for point in polyline[1:-1]:
            vertices.append(point)
            indices.append(len(vertices) - 1)
        indices.append(1)
        arc_indices[arc_name] = indices

    return np.asarray(vertices, dtype=float), arc_indices


def write_curve_obj(network: CurveNetwork, path: Path) -> dict[str, list[int]]:
    vertices, arc_indices = network_to_vertices(network)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for point in vertices:
            f.write(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
        for arc_name in network.arc_order:
            indices = arc_indices[arc_name]
            for a, b in zip(indices, indices[1:]):
                f.write(f"l {a + 1} {b + 1}\n")

    return arc_indices


def write_repulsor_curve(
    vertices: np.ndarray,
    arc_indices: dict[str, list[int]],
    arc_order: tuple[str, ...],
    output: Path,
) -> None:
    edges: list[tuple[int, int]] = []
    for arc_name in arc_order:
        indices = arc_indices[arc_name]
        edges.extend((int(a), int(b)) for a, b in zip(indices, indices[1:]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(f"vertices {len(vertices)}\n")
        for point in vertices:
            f.write(f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
        f.write(f"edges {len(edges)}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
    return np.asarray(vertices, dtype=float)


def network_from_vertices(
    template: CurveNetwork,
    vertices: np.ndarray,
    arc_indices: dict[str, list[int]],
) -> CurveNetwork:
    node_positions = {
        template.node_order[0]: vertices[0],
        template.node_order[1]: vertices[1],
    }
    arc_polylines = {
        arc_name: vertices[np.asarray(indices, dtype=int)]
        for arc_name, indices in arc_indices.items()
    }
    return CurveNetwork(
        name=template.name,
        node_order=template.node_order,
        node_positions=node_positions,
        arc_order=template.arc_order,
        arc_polylines=arc_polylines,
        arc_specs=template.arc_specs,
        node_colors=template.node_colors,
        arc_colors=template.arc_colors,
        metadata=dict(template.metadata),
    )


def layout_payload(network: CurveNetwork, params: dict[str, object]) -> dict[str, object]:
    return {
        "example": network.name,
        "node_order": list(network.node_order),
        "node_positions": {name: network.node_positions[name].tolist() for name in network.node_order},
        "edge_order": list(network.arc_order),
        "edge_polylines": {name: network.arc_polylines[name].tolist() for name in network.arc_order},
        "arc_specs": network.arc_specs,
        "arc_colors": network.arc_colors,
        "node_colors": network.node_colors,
        "metadata": network.metadata,
        "parameters": params,
    }


def write_layout_json(network: CurveNetwork, params: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(layout_payload(network, params), indent=2), encoding="utf-8")
