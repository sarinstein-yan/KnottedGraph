from __future__ import annotations

import inspect
import json

from benchmark_topoly_common import (
    calibration_cases,
    coords,
    matching_near,
    run_kg,
    run_topoly,
)


def _print_source(label, obj, limit=16000):
    print("OBJECT", label, obj)
    if obj is None:
        return
    try:
        source = inspect.getsource(obj)
        print("SOURCE_BEGIN", label)
        print(source[:limit])
        print("SOURCE_END", label)
    except Exception as exc:
        print("SOURCE_ERROR", label, type(exc).__name__, str(exc))


def main():
    import topoly

    print("TOPOLY_FILE", topoly.__file__)
    print("YAMADA_SIGNATURE", inspect.signature(topoly.yamada))

    for offset in (0, 1):
        print("OFFSET", offset)
        for n, target in calibration_cases():
            xyz = coords(n)
            bridges = matching_near(n, target, 9000 + n)
            kg = run_kg(xyz, bridges, repeats=1)
            try:
                tp = run_topoly(xyz, bridges, offset, repeats=1)
                payload = {
                    "n": n,
                    "target": target,
                    "bridges_input": bridges,
                    "bridges_topoly": [(a + offset, b + offset) for a, b in bridges],
                    "kg_crossings": kg["crossings"],
                    "kg_polynomial": kg["polynomial"],
                    "kg_signature": kg["signature"],
                    "topoly_raw": tp["raw"],
                    "topoly_signature": tp["signature"],
                }
                print("OBS", json.dumps(payload, separators=(",", ":")))
            except Exception as exc:
                print("ERROR", n, type(exc).__name__, str(exc))

    try:
        import topoly.invariants as inv
        print("INVARIANTS_FILE", inv.__file__)
        for name in ("Graph", "Invariant", "YamadaGraph"):
            _print_source(name, getattr(inv, name, None))

        graph_cls = getattr(inv, "Graph", None)
        if graph_cls is not None:
            for method_name in (
                "__init__",
                "parse_input",
                "parse_bridges",
                "parse_coordinates",
                "parse_closed",
                "close",
            ):
                _print_source(
                    f"Graph.{method_name}",
                    getattr(graph_cls, method_name, None),
                    limit=12000,
                )

        invariant_cls = getattr(inv, "Invariant", None)
        if invariant_cls is not None:
            for method_name in (
                "__init__",
                "calculate_spatial",
                "analyze_points",
                "analyze_single_point",
            ):
                _print_source(
                    f"Invariant.{method_name}",
                    getattr(invariant_cls, method_name, None),
                    limit=12000,
                )

        # Directly construct Invariant on one calibration graph and report the
        # internal representation created from the bridges argument.
        if invariant_cls is not None:
            n, target = calibration_cases()[0]
            xyz = coords(n)
            bridges = matching_near(n, target, 9000 + n)
            for offset in (0, 1):
                shifted = [(a + offset, b + offset) for a, b in bridges]
                try:
                    obj = invariant_cls(xyz.tolist(), bridges=shifted, breaks=[])
                    snapshot = {}
                    for attr in (
                        "arcs",
                        "bridges",
                        "breaks",
                        "coordinates",
                        "init_data",
                        "pdcode",
                        "emcode",
                        "run_from_code",
                    ):
                        if hasattr(obj, attr):
                            value = getattr(obj, attr)
                            try:
                                snapshot[attr] = repr(value)[:6000]
                            except Exception:
                                snapshot[attr] = f"<{type(value).__name__}>"
                    print(
                        "INVARIANT_SNAPSHOT",
                        offset,
                        json.dumps(snapshot, separators=(",", ":")),
                    )
                except Exception as exc:
                    print(
                        "INVARIANT_CONSTRUCT_ERROR",
                        offset,
                        type(exc).__name__,
                        str(exc),
                    )
    except Exception as exc:
        print("INVARIANT_IMPORT_ERROR", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
