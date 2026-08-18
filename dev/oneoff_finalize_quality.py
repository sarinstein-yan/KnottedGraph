from __future__ import annotations

import json
from pathlib import Path
import subprocess

from promote_native_prepared import main as promote_native_prepared

ROOT = Path(__file__).resolve().parents[1]


def update_tests_workflow() -> int:
    coverage = json.loads((ROOT / "coverage.json").read_text())
    percent = float(coverage["totals"]["percent_covered"])
    threshold = max(50, int(percent) - 2)

    path = ROOT / ".github/workflows/tests.yml"
    text = path.read_text()
    marker = "\n  oneoff-quality-upgrade:\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"

    old_test = """      - name: Run complete Python test suite\n        run: uv run --no-sync pytest -q\n"""
    new_test = f"""      - name: Run complete Python test suite with coverage\n        run: >\n          uv run --no-sync pytest -q\n          --cov=knotted_graph\n          --cov-report=term-missing\n          --cov-report=xml\n          --cov-fail-under={threshold}\n"""
    assert text.count(old_test) == 1
    text = text.replace(old_test, new_test)

    start = text.index("\n  wheel:\n")
    end = text.index("\n  lint:\n", start)
    wheel = r'''
  wheel:
    name: Native wheel (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            venv-python: wheel-test/bin/python
          - os: macos-latest
            venv-python: wheel-test/bin/python
          - os: windows-latest
            venv-python: wheel-test/Scripts/python.exe

    steps:
      - name: Check out repository
        uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build distribution
        run: uv build
      - name: Create clean wheel environment
        shell: bash
        run: python -m venv wheel-test
      - name: Install built wheel
        shell: bash
        run: |
          WHEEL=$(python -c "from pathlib import Path; print(next(Path('dist').glob('*.whl')))")
          "${{ matrix.venv-python }}" -m pip install "$WHEEL"
      - name: Smoke-test installed wheel and native backend
        shell: bash
        run: |
          "${{ matrix.venv-python }}" - <<'PY'
          import sympy as sp
          import knotted_graph
          from knotted_graph.core import ThetaGraph
          from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive
          from knotted_graph.invariants.yamada.native import native_available

          A = sp.Symbol("A")
          result = compute_yamada_polynomial_recursive(ThetaGraph(3), A)
          print("version:", knotted_graph.__version__)
          print("native Yamada backend:", native_available())
          print("Theta_3:", result)
          assert native_available(), "built wheel does not contain native Yamada extension"
          assert result != 0
          PY
'''
    text = text[:start] + "\n" + wheel.rstrip() + text[end:]

    ruff = """      - name: Run Ruff\n        run: uv run --no-sync ruff check src tests dev/check_repository_consistency.py\n"""
    assert text.count(ruff) == 1
    text = text.replace(
        ruff,
        ruff
        + "\n"
          "      - name: Run incremental Pyright checks\n"
          "        run: uv run --no-sync pyright\n",
    )
    path.write_text(text)
    return threshold


def update_benchmark_workflow() -> None:
    path = ROOT / ".github/workflows/yamada-benchmark.yml"
    text = path.read_text()
    text = text.replace(
        '      - "dev/benchmark_topoly_random_cubic_ensemble.py"\n',
        '      - "dev/benchmark_topoly_random_cubic_ensemble.py"\n'
        '      - "dev/generate_topoly_random_cubic_corpus.py"\n'
        '      - "dev/benchmark_data/topoly_random_cubic_v1.jsonl"\n',
    )
    old = """          uv sync --group dev\n          uv pip install topoly\n"""
    if old in text:
        text = text.replace(old, "          uv sync --group dev --group benchmark\n")

    # Remove research-only candidates whose exact experiments showed no runtime
    # benefit. Production CI retains the permanent structural/native regressions.
    for line in (
        '      - "dev/benchmark_skein_hybrid_candidate.py"\n',
        '      - "dev/benchmark_frontier_flow_candidate.py"\n',
        '      - "dev/benchmark_native_discrete_canonical_candidate.py"\n',
        '      - "dev/benchmark_native_prepared_candidate.py"\n',
    ):
        text = text.replace(line, "")
    for block in (
        """      - name: Benchmark guarded hybrid skein candidate on irreducible diagrams\n        run: uv run --no-sync python dev/benchmark_skein_hybrid_candidate.py\n""",
    ):
        text = text.replace(block, "")
    path.write_text(text)


def update_notebook_workflow() -> None:
    path = ROOT / ".github/workflows/notebooks.yml"
    text = path.read_text()
    old = """          uv sync --extra all --group dev --group docs\n          uv pip install topoly\n"""
    if old in text:
        text = text.replace(
            old,
            "          uv sync --extra all --group dev --group docs --group benchmark\n",
        )
    path.write_text(text)


def remove_rejected_experiments() -> None:
    """Remove candidate-only files after their exact benchmark decisions."""
    candidates = (
        ".github/workflows/yamada-flow-candidate.yml",
        "dev/benchmark_frontier_flow_candidate.py",
        "dev/benchmark_native_discrete_canonical_candidate.py",
        "dev/benchmark_native_prepared_candidate.py",
        "dev/benchmark_skein_hybrid_candidate.py",
        "dev/benchmark_two_vertex_candidate.py",
        "dev/benchmark_two_vertex_decomposition_candidate.py",
        "dev/benchmark_exact_isomorphism_candidate.py",
        "dev/benchmark_isomorphism_candidate.py",
        "dev/benchmark_state_streaming_candidate.py",
        "dev/benchmark_native_streaming_candidate.py",
        "dev/benchmark_streaming_candidate.py",
    )
    for relative in candidates:
        path = ROOT / relative
        if path.exists():
            path.unlink()


def remove_temporary_migration_files() -> None:
    for relative in (
        ".github/workflows/oneoff-integrate-nodal-memory.yml",
        ".github/workflows/oneoff-fix-frontier-union.yml",
        "dev/oneoff_quality_upgrade.py",
        "dev/oneoff_finalize_quality.py",
        "dev/promote_native_prepared.py",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def validate_promoted_native_path() -> None:
    """Apply the pre-benchmarked native path and gate it before any commit."""
    promote_native_prepared()
    subprocess.run(
        [
            "uv",
            "sync",
            "--reinstall-package",
            "knotted-graph",
            "--all-extras",
            "--group",
            "dev",
            "--group",
            "docs",
            "--group",
            "benchmark",
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "pytest", "-q"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "python", "dev/run_yamada_sanity_checks.py"],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    validate_promoted_native_path()
    threshold = update_tests_workflow()
    update_benchmark_workflow()
    update_notebook_workflow()
    remove_rejected_experiments()
    remove_temporary_migration_files()
    print(f"Permanent coverage floor set to {threshold}% from measured baseline.")


if __name__ == "__main__":
    main()
