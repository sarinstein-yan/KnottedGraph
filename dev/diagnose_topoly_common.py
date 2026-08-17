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

    # Inspect the object used internally by yamada if its class is publicly
    # importable in this Topoly build. This helps determine whether bridges are
    # graph edges, closure instructions, or filtered polymer annotations.
    try:
        import topoly.invariants as inv
        print("INVARIANTS_FILE", inv.__file__)
        for name in ("Invariant", "YamadaGraph"):
            obj = getattr(inv, name, None)
            print("OBJECT", name, obj)
            if obj is not None:
                try:
                    source = inspect.getsource(obj)
                    print("SOURCE_BEGIN", name)
                    print(source[:12000])
                    print("SOURCE_END", name)
                except Exception as exc:
                    print("SOURCE_ERROR", name, type(exc).__name__, str(exc))
    except Exception as exc:
        print("INVARIANT_IMPORT_ERROR", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
