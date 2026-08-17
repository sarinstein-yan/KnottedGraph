from __future__ import annotations

import inspect
import json

import sympy as sp

from benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def main():
    import topoly.polynomial as polynomial
    from topoly import yamada
    from topoly.invariants import YamadaGraph
    from topoly.params import Translate

    for name in ("print_short", "__str__", "__init__", "__mul__", "__add__"):
        obj = getattr(polynomial.Poly, name, None)
        if obj is not None:
            try:
                print(f"POLY_SOURCE_BEGIN {name}")
                print(inspect.getsource(obj))
                print(f"POLY_SOURCE_END {name}")
            except Exception as exc:
                print("POLY_SOURCE_ERROR", name, type(exc).__name__, str(exc))

    for crossings in range(0, 5):
        graph = multi_crossing_theta(max(1, crossings))
        processor = PDCode(graph)
        pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        if crossings == 0:
            # Use a crossing-free theta from the same fixture by rotating the
            # third path is unnecessary here; Topoly's known-case parser can be
            # tested directly on the resolved all-vertex graph from one state.
            calculator = processor
        kg_raw = processor.compute_yamada(A, normalize=False, n_jobs=1, method="negami")
        kg_norm = processor.compute_yamada(A, normalize=True, n_jobs=1, method="negami")

        topoly_raw = yamada(
            pdcode,
            max_cross=200,
            poly_reduce=True,
            translate=Translate.NO,
            hide_trivial=False,
            hide_rare=False,
            minimal=False,
            cuda=False,
            run_parallel=False,
            parallel_workers=1,
        )
        graph_obj = YamadaGraph(pdcode)
        full = graph_obj.point(max_cross=200)
        try:
            full_str = str(full)
        except Exception as exc:
            full_str = f"ERROR:{type(exc).__name__}:{exc}"
        try:
            short_str = full.print_short()
        except Exception as exc:
            short_str = f"ERROR:{type(exc).__name__}:{exc}"

        print("IDENTICAL_PD", json.dumps({
            "crossings": processor.crossings.__len__(),
            "pdcode": pdcode,
            "kg_raw": str(sp.expand(kg_raw)),
            "kg_normalized": str(sp.expand(kg_norm)),
            "topoly_yamada": str(topoly_raw),
            "topoly_point_str": full_str,
            "topoly_point_short": str(short_str),
            "topoly_point_type": type(full).__name__,
            "topoly_point_dict": repr(getattr(full, "__dict__", None)),
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
