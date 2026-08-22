from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UG = ROOT / "User_guide"
CANONICAL_BRANCH = "YAMADA_Optimization_LATEST"

YAMADA_MARKERS = (
    "compute_yamada_polynomial(",
    "compute_graph_yamada_polynomial(",
    "Yamada(",
)

PLOTTING_ONLY = {
    UG / "benchmarks" / "06_paper_scaling_publication_figure.ipynb",
    UG / "benchmarks" / "06_paper_scaling_publication_figure_final_push.ipynb",
}

SOURCE_OVERRIDE_FRAGMENTS = (
    "sys.path.insert(0, str(SRC_ROOT))",
    "sys.path.insert(0,str(SRC_ROOT))",
    "sys.path.insert(0, str(SRC))",
    "sys.path.insert(0,str(SRC))",
    'env["PYTHONPATH"] = str(SRC)',
    "env['PYTHONPATH'] = str(SRC)",
)

STALE_BRANCH_FRAGMENTS = (
    "Latest_Workplace",
    "perf/yamada-max-optimization",
    "cleanup/yamada-single-production-path-20260822",
    "fix/yamada-projection-skeleton-robustness-20260822",
    "diagnostic/hard-exact-matrix-20260822",
    "diagnostic/hard-source-inspect-20260822",
)


def text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def has_yamada(nb: dict) -> bool:
    return any(
        any(marker in text(cell) for marker in YAMADA_MARKERS)
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _parseable_python(source: str) -> str:
    """Remove IPython magic/shell command lines before Python syntax checking."""
    return "\n".join(
        line
        for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )


def validate(path: Path, nb: dict) -> None:
    all_text: list[str] = []
    for cell in nb.get("cells", []):
        source = text(cell)
        all_text.append(source)
        if cell.get("cell_type") != "code":
            continue
        ast.parse(_parseable_python(source) or "pass")
        for fragment in SOURCE_OVERRIDE_FRAGMENTS:
            if fragment in source:
                raise AssertionError(f"{path}: stale source override: {fragment}")

    joined = "\n".join(all_text)
    for fragment in STALE_BRANCH_FRAGMENTS:
        if fragment in joined:
            raise AssertionError(f"{path}: stale optimization branch reference: {fragment}")

    if has_yamada(nb) and path not in PLOTTING_ONLY:
        code = "\n".join(
            text(cell)
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        if "native_available" not in code or "native_import_error" not in code:
            raise AssertionError(
                f"{path}: Yamada evaluation lacks native-backend provenance"
            )


def process(path: Path, *, check: bool) -> None:
    nb = json.loads(path.read_text())
    before = json.dumps(nb, sort_keys=True)

    metadata = nb.setdefault("metadata", {})
    metadata["based_on_branch"] = CANONICAL_BRANCH

    validate(path, nb)
    after = json.dumps(nb, sort_keys=True)
    changed = before != after

    if check and changed:
        raise AssertionError(
            f"{path} is not normalized; run this script without --check"
        )
    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("updated", path.relative_to(ROOT))
    else:
        print("ok     ", path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    paths = sorted(UG.rglob("*.ipynb"))
    for path in paths:
        process(path, check=args.check)
    print(f"validated {len(paths)} notebooks")


if __name__ == "__main__":
    main()
