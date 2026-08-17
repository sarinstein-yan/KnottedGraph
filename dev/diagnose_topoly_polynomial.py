from __future__ import annotations

import inspect
import json

import sympy as sp

from benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.projection import PDCode

A = sp.Symbol("A")


def main():
    from topoly import yamada
    from topoly.invariants import YamadaGraph
    from topoly.params import Translate

    inspected_type = False
    for requested_crossings in range(1, 5):
        graph = multi_crossing_theta(requested_crossings)
        processor = PDCode(graph)
        pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))
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

        if not inspected_type:
            poly_type = type(full)
            print("POLY_TYPE", poly_type, inspect.getmodule(poly_type))
            for name in (
                "print_short",
                "__str__",
                "__init__",
                "__mul__",
                "__add__",
                "__neg__",
            ):
                obj = getattr(poly_type, name, None)
                if obj is not None:
                    try:
                        print(f"POLY_SOURCE_BEGIN {name}")
                        print(inspect.getsource(obj))
                        print(f"POLY_SOURCE_END {name}")
                    except Exception as exc:
                        print("POLY_SOURCE_ERROR", name, type(exc).__name__, str(exc))
            inspected_type = True

        try:
            full_str = str(full)
        except Exception as exc:
            full_str = f"ERROR:{type(exc).__name__}:{exc}"
        try:
            short_str = full.print_short()
        except Exception as exc:
            short_str = f"ERROR:{type(exc).__name__}:{exc}"

        attrs = {}
        for name in dir(full):
            if name.startswith("_"):
                continue
            try:
                value = getattr(full, name)
            except Exception:
                continue
            if isinstance(value, (int, float, str, list, tuple, dict)):
                attrs[name] = repr(value)

        print("IDENTICAL_PD", json.dumps({
            "crossings": len(processor.crossings),
            "pdcode": pdcode,
            "kg_raw": str(sp.expand(kg_raw)),
            "kg_normalized": str(sp.expand(kg_norm)),
            "topoly_yamada": str(topoly_raw),
            "topoly_point_str": full_str,
            "topoly_point_short": str(short_str),
            "topoly_point_type": type(full).__name__,
            "topoly_point_dict": repr(getattr(full, "__dict__", None)),
            "topoly_public_attrs": attrs,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
