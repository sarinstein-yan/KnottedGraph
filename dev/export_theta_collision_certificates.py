from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from knotted_graph.projection import PDCode, compute_yamada_polynomial

import discover_yamada_theta_collisions as core

A = sp.Symbol("A")

TARGETS = [
    ("clean_left", 7, 57, 0.12),
    ("clean_right", 13, 67, 0.08),
    ("same_constituent_6_left", 32, 58, 0.12),
    ("same_constituent_6_right", 39, 153, 0.05),
    ("same_constituent_9_left", 32, 197, 0.12),
    ("same_constituent_9_right", 39, 102, 0.05),
]


def _json_point(point) -> list[float]:
    return [float(value) for value in point]


def projection_certificate(graph) -> dict:
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arcs = {}
    for arc_id, arc in sorted(pd.arcs.items()):
        arcs[str(arc_id)] = {
            "start_type": arc.start_type,
            "start_id": arc.start_id,
            "end_type": arc.end_type,
            "end_id": arc.end_id,
            "coords": [_json_point(point) for point in arc.line.coords],
        }
    crossings = {}
    for crossing_id, crossing in sorted(pd.crossings.items()):
        crossings[str(crossing_id)] = {
            "raw_ccw_ordered_arcs": list(crossing._raw_ccw_ordered_arcs),
        }
    vertices = {}
    for vertex_id, vertex in sorted(pd.vertices.items()):
        vertices[str(vertex_id)] = {
            "ccw_ordered_arcs": list(vertex.ccw_ordered_arcs),
        }
    return {
        "num_crossings": len(pd.crossings),
        "arcs": arcs,
        "crossings": crossings,
        "vertices": vertices,
    }


def render_projection(edge_points: list[np.ndarray], output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for points in edge_points:
        ax.plot(points[:, 0], points[:, 1], linewidth=2.0)
    all_points = np.vstack(edge_points)
    ax.scatter([all_points[0, 0], all_points[-1, 0]], [all_points[0, 1], all_points[-1, 1]], s=28)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def run(plantri: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    shadows = core.generate_shadows(plantri, 8)
    by_index = {shadow.index: shadow for shadow in shadows}
    records = []

    for label, shadow_index, bits, fraction in TARGETS:
        shadow = by_index[shadow_index]
        traces = core.trace_theta_edges(shadow)
        graph, edge_points = core.spatial_theta(
            shadow,
            bits,
            approach_fraction=fraction,
        )
        result = compute_yamada_polynomial(
            graph,
            A,
            rotation_angles=(0.0, 0.0, 0.0),
            normalize=True,
            n_jobs=1,
            crossing_warning_threshold=None,
            return_result=True,
        )
        if int(result.projection.num_crossings) != 8:
            raise AssertionError(f"{label}: reconstructed projection is not eight-crossing")

        crossing_vertices = shadow.crossing_vertices
        assignment = {
            str(crossing): int((bits >> i) & 1)
            for i, crossing in enumerate(crossing_vertices)
        }
        record = {
            "label": label,
            "shadow": shadow_index,
            "bits": bits,
            "bitstring": format(bits, "08b"),
            "approach_fraction": fraction,
            "rotation_system": {
                str(vertex): list(neighbours)
                for vertex, neighbours in sorted(shadow.rotation.items())
            },
            "theta_traces": traces,
            "crossing_vertices_in_bit_order": list(crossing_vertices),
            "over_under_assignment": assignment,
            "edge_polylines_xyz": [
                [_json_point(point) for point in points]
                for points in edge_points
            ],
            "projection_code": projection_certificate(graph),
            "normalized_yamada": str(sp.expand(result.polynomial)),
        }
        records.append(record)
        render_projection(
            edge_points,
            output_dir / f"{label}.svg",
            f"{label}: shadow {shadow_index}, bits {format(bits, '08b')}",
        )

    payload = {"targets": records}
    (output_dir / "theta8_exact_diagram_certificates.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output_dir)


if __name__ == "__main__":
    main()
