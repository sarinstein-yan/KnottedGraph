"""Persistent cache for the long benchmark drivers used by notebook 03.

This module is intentionally private and activates only when Python is executing
an approved benchmark driver used by
``User_guide/benchmarks/03_knottedgraph_vs_topoly_scaling.ipynb``.

A successful driver run ends by printing ``SUMMARY=<json>``. The summary is
stored locally under the repository's Git metadata. On the next run with the
same command-line configuration, benchmark source, relevant KnottedGraph source,
Python/dependency versions, and machine architecture, the cached rows are
replayed to stdout and the driver exits before any expensive timings start.

The notebook therefore receives exactly the same streamed-row/SUMMARY protocol
on a cache hit and continues directly to its acceptance, aggregation, and
plotting cells.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import sys
from typing import Any

_TARGETS = {
    "benchmark_topoly_paper_scaling.py",
    "benchmark_topoly_extended_scaling.py",
    "benchmark_topoly_random_cubic_ensemble.py",
}
_CACHE_SCHEMA = 1
_DISABLE_VALUES = {"1", "true", "yes", "on"}


def _repo_root(script: Path) -> Path | None:
    root = script.resolve().parent.parent
    if (root / "pyproject.toml").exists() and (root / "src" / "knotted_graph").exists():
        return root
    return None


def _environment_fingerprint() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }
    for package in ("knotted_graph", "topoly", "networkx", "numpy", "sympy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _relevant_source_files(root: Path, script: Path) -> list[Path]:
    files = [script.resolve(), Path(__file__).resolve()]

    # The focused paper driver deliberately reuses the graph constructors and
    # polynomial-equivalence helpers from the broader diagnostic driver, so that
    # imported source must participate in cache invalidation as well.
    if script.name == "benchmark_topoly_paper_scaling.py":
        helper = root / "dev" / "benchmark_topoly_extended_scaling.py"
        if helper.exists():
            files.append(helper.resolve())

    for relative in (
        Path("src/knotted_graph/core"),
        Path("src/knotted_graph/projection"),
        Path("src/knotted_graph/invariants/yamada"),
    ):
        base = root / relative
        if not base.exists():
            continue
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix in {".py", ".cpp", ".hpp", ".h"}
        )
    return sorted(set(files), key=lambda path: str(path))


def _code_fingerprint(root: Path, script: Path) -> str:
    digest = hashlib.sha256()
    for path in _relevant_source_files(root, script):
        try:
            label = path.relative_to(root)
        except ValueError:
            label = path
        digest.update(str(label).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(
        json.dumps(_environment_fingerprint(), sort_keys=True).encode()
    )
    return digest.hexdigest()


def _resolve_git_dir(root: Path) -> Path | None:
    dotgit = root / ".git"
    if dotgit.is_dir():
        return dotgit
    if not dotgit.is_file():
        return None

    try:
        line = dotgit.read_text().strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not line.lower().startswith(prefix):
        return None
    target = line[len(prefix) :].strip()
    path = Path(target)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path if path.exists() else None


def _cache_directory(root: Path) -> Path:
    override = os.environ.get("KG_BENCHMARK_CACHE_DIR")
    if override:
        directory = Path(override).expanduser().resolve()
    else:
        git_dir = _resolve_git_dir(root)
        if git_dir is not None:
            directory = git_dir / "knotted_graph_benchmark_cache"
        else:
            root_key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
            directory = (
                Path.home()
                / ".cache"
                / "knotted_graph"
                / "benchmark_03"
                / root_key
            )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cache_path(root: Path, script: Path, fingerprint: str) -> Path:
    key_payload = {
        "schema": _CACHE_SCHEMA,
        "script": script.name,
        "argv": sys.argv[1:],
        "code_fingerprint": fingerprint,
    }
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode()
    ).hexdigest()[:24]
    return _cache_directory(root) / f"{script.stem}-{key}.json"


def _load_cache(path: Path, fingerprint: str) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if data.get("schema") != _CACHE_SCHEMA:
        return None
    if data.get("code_fingerprint") != fingerprint:
        return None
    if data.get("argv") != sys.argv[1:]:
        return None

    rows = data.get("summary")
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    return rows


def _save_cache(
    path: Path,
    fingerprint: str,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": _CACHE_SCHEMA,
        "script": Path(sys.argv[0]).name,
        "argv": list(sys.argv[1:]),
        "code_fingerprint": fingerprint,
        "environment": _environment_fingerprint(),
        "summary": rows,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


class _SummaryCachingStdout:
    """Tee stdout and atomically cache a completed ``SUMMARY=`` record."""

    def __init__(self, wrapped, path: Path, fingerprint: str):
        self._wrapped = wrapped
        self._path = path
        self._fingerprint = fingerprint
        self._buffer = ""

    def _consume_line(self, line: str) -> None:
        if not line.startswith("SUMMARY="):
            return
        try:
            rows = json.loads(line[len("SUMMARY=") :])
        except json.JSONDecodeError:
            return
        if (
            isinstance(rows, list)
            and rows
            and all(isinstance(row, dict) for row in rows)
        ):
            _save_cache(self._path, self._fingerprint, rows)

    def write(self, text: str):
        result = self._wrapped.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line)
        return result

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


def install_benchmark_cache() -> None:
    """Activate persistent caching for notebook-03 benchmark driver processes."""
    if os.environ.get("KG_DISABLE_BENCHMARK_CACHE", "").lower() in _DISABLE_VALUES:
        return
    if mp.current_process().name != "MainProcess" or not sys.argv:
        return

    script = Path(sys.argv[0]).resolve()
    if script.name not in _TARGETS:
        return

    root = _repo_root(script)
    if root is None:
        return

    fingerprint = _code_fingerprint(root, script)
    path = _cache_path(root, script, fingerprint)
    rows = _load_cache(path, fingerprint)

    if rows is not None:
        print(f"CACHE_HIT={path}")
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        print("SUMMARY=" + json.dumps(rows, sort_keys=True))
        sys.stdout.flush()
        raise SystemExit(0)

    print(f"CACHE_MISS={path}")
    sys.stdout = _SummaryCachingStdout(sys.stdout, path, fingerprint)
