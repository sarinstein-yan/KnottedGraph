from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import certify_theta_associated_link_homfly as base
import certify_theta_associated_link_homfly_v3 as v3
import certify_theta_associated_link_homfly_v5 as v5


def fixed_build_boundary(graph, m, eps: float, trim: float):
    edge_points = [None] * 3
    for _, _, data in graph.edges(data=True):
        edge_points[int(data["role"])] = np.asarray(data["pts"], dtype=float)
    side_paths = {
        role: base.ribbon_sides(edge_points[role], m[role], eps, trim)
        for role in range(3)
    }
    centers = {
        name: np.asarray(graph.nodes[name]["pos"], dtype=float)
        for name in ("u", "v")
    }
    pair = {
        name: v3.twist_aware_vertex_pairings(side_paths, centers[name], name)
        for name in ("u", "v")
    }

    visited = set()
    components = []
    for start_role in range(3):
        for start_side in (1, -1):
            state = (start_role, start_side, 1)
            if state in visited or (start_role, start_side, -1) in visited:
                continue
            points = []
            current = state
            first = state
            for _ in range(20):
                role, side, direction = current
                visited.add(current)
                path = side_paths[role][side]
                if direction == 1:
                    seg = path
                    vertex = "v"
                else:
                    seg = path[::-1]
                    vertex = "u"
                if points:
                    points.extend(seg[1:].tolist())
                else:
                    points.extend(seg.tolist())
                nxt_role, nxt_side = pair[vertex][(role, side)]
                a = np.asarray(points[-1])
                b = side_paths[nxt_role][nxt_side][-1 if vertex == "v" else 0]
                conn = base.connector(a, b, centers[vertex])
                points.extend(conn[1:].tolist())
                current = (nxt_role, nxt_side, -direction)
                if current == first:
                    break
            else:
                raise AssertionError("boundary trace did not close")
            comp = np.asarray(points, dtype=float)
            if not np.allclose(comp[0], comp[-1]):
                comp = np.vstack([comp, comp[0]])
            components.append(comp)

    # Directed tracing can produce the same geometric component with different
    # sampling density/orientation.  Never compare arrays pointwise: that is not
    # invariant under reparameterisation and can fail on unequal shapes.  Use a
    # coarse geometric fingerprint only to remove exact reverse duplicates.
    unique = []
    fingerprints = set()
    for comp in components:
        cloud = comp[:-1]
        centroid = np.mean(cloud, axis=0)
        radii = np.linalg.norm(cloud - centroid, axis=1)
        # Length, centroid, and sorted radial quantiles are invariant under
        # reversing the parametrisation and robust to the connector sampling.
        seglen = np.sum(np.linalg.norm(np.diff(comp, axis=0), axis=1))
        quantiles = np.quantile(radii, [0.0, 0.25, 0.5, 0.75, 1.0])
        key = tuple(np.round(np.r_[seglen, centroid, quantiles], 7))
        if key not in fingerprints:
            fingerprints.add(key)
            unique.append(comp)

    if len(unique) != 3:
        raise AssertionError(
            f"expected 3 boundary components, got {len(unique)} from {len(components)} traces"
        )
    return unique


base.build_boundary = fixed_build_boundary
base.vertex_pairings = v3.twist_aware_vertex_pairings
base.crossing_data = v5.zero_linking_crossing_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.plantri, args.output, args.xyz_dir)


if __name__ == "__main__":
    main()
