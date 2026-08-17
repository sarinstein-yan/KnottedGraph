from __future__ import annotations

import json

import sympy as sp

from benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.projection import compute_yamada_polynomial

A = sp.Symbol("A")

ROTATIONS = (
    (0.0, 0.0, 0.0),
    (0.13, -0.21, 0.08),
    (-0.17, 0.11, 0.19),
)


def main():
    rows = []
    for components in range(1, 5):
        graph = multi_crossing_theta(components)
        for rotation in ROTATIONS:
            for method in ("negami", "recursive"):
                for normalize in (False, True):
                    answer = compute_yamada_polynomial(
                        graph,
                        A,
                        rotation_angles=rotation,
                        normalize=normalize,
                        n_jobs=1,
                        method=method,
                        return_result=True,
                    )
                    rows.append({
                        "components": components,
                        "rotation": list(rotation),
                        "method": method,
                        "normalize": normalize,
                        "crossings": answer.projection.num_crossings,
                        "pd_code": answer.projection.pd_code,
                        "polynomial": str(sp.expand(answer.polynomial)),
                    })

    # Explicitly include the public default scheduling argument. This row must
    # remain numerically identical even if the optimized branch changes how the
    # work is scheduled internally.
    graph = multi_crossing_theta(4)
    for method in ("negami", "recursive"):
        answer = compute_yamada_polynomial(
            graph,
            A,
            rotation_angles=(0.0, 0.0, 0.0),
            normalize=True,
            n_jobs=-1,
            method=method,
            return_result=True,
        )
        rows.append({
            "components": 4,
            "rotation": [0.0, 0.0, 0.0],
            "method": method,
            "normalize": True,
            "n_jobs": -1,
            "crossings": answer.projection.num_crossings,
            "pd_code": answer.projection.pd_code,
            "polynomial": str(sp.expand(answer.polynomial)),
        })

    print(json.dumps(rows, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
