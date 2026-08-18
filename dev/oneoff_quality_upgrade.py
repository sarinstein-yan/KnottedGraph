from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def integrate_nodal_memory() -> None:
    path = ROOT / "src/knotted_graph/applications/nodal/skeleton.py"
    text = path.read_text()
    if "class _LazyCoordinateGrid:" not in text:
        marker = "\n\nclass NodalSkeleton:\n"
        assert text.count(marker) == 1
        descriptor = r'''

_MISSING_GRID = object()


class _LazyCoordinateGrid:
    """Legacy dense coordinate grid materialized only on explicit access.

    The returned object is a normal writable C-contiguous ``ndarray`` matching
    ``np.meshgrid(..., indexing="ij")``. Explicit assignment and deletion keep
    the ordinary instance-attribute semantics of the historical API.
    """

    def __init__(self, axis: int, name: str):
        self.axis = axis
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.name, _MISSING_GRID)
        if value is _MISSING_GRID:
            shape = (instance.dimension,) * 3
            axis_values = (
                instance.kx_vals,
                instance.ky_vals,
                instance.kz_vals,
            )[self.axis]
            reshape = [1, 1, 1]
            reshape[self.axis] = instance.dimension
            value = np.broadcast_to(axis_values.reshape(reshape), shape).copy()
            instance.__dict__[self.name] = value
        return value

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        instance.__dict__.pop(self.name, None)
'''
        text = text.replace(marker, descriptor + marker)

        pauli = "    pauli_vec = (pauli_x, pauli_y, pauli_z)\n"
        assert text.count(pauli) == 1
        text = text.replace(
            pauli,
            pauli
            + "\n"
            + "    # Public compatibility arrays, now materialized lazily.\n"
            + "    kx_grid = _LazyCoordinateGrid(0, \"kx_grid\")\n"
            + "    ky_grid = _LazyCoordinateGrid(1, \"ky_grid\")\n"
            + "    kz_grid = _LazyCoordinateGrid(2, \"kz_grid\")\n",
        )

        eager = """        self.kx_grid, self.ky_grid, self.kz_grid = np.meshgrid(\n            self.kx_vals, self.ky_vals, self.kz_vals,\n            indexing='ij'\n        )\n\n"""
        assert text.count(eager) == 1
        text = text.replace(
            eager,
            "        # Dense k-space coordinate grids are compatibility attributes and\n"
            "        # are intentionally not materialized until explicitly accessed.\n\n",
        )

        old_spectrum = "        return np.sqrt(np.sum(self._bloch_vec_grid**2, axis=0))\n"
        assert text.count(old_spectrum) == 1
        new_spectrum = """        shape = (self.dimension,) * 3\n        grids = (\n            self.kx_vals[:, None, None],\n            self.ky_vals[None, :, None],\n            self.kz_vals[None, None, :],\n        )\n        total = np.empty(shape, dtype=np.complex128)\n\n        for index, (expr, func) in enumerate(\n            zip(self.bloch_vec, self.bloch_vec_funcs)\n        ):\n            if expr.free_symbols:\n                component = np.asarray(func(*grids), dtype=np.complex128)\n            else:\n                component = np.asarray(complex(expr), dtype=np.complex128)\n\n            if component.flags.writeable:\n                np.multiply(component, component, out=component)\n                squared = component\n            else:\n                squared = np.multiply(component, component)\n\n            if index == 0:\n                np.copyto(total, squared)\n            else:\n                np.add(total, squared, out=total)\n\n        np.sqrt(total, out=total)\n        return total\n"""
        text = text.replace(old_spectrum, new_spectrum)

        old_coords = """        point_mask = np.where(self._skeleton_image)\n        return np.asarray([self.kx_grid[point_mask],\n                           self.ky_grid[point_mask],\n                           self.kz_grid[point_mask]]).T\n"""
        assert text.count(old_coords) == 1
        new_coords = """        point_mask = np.where(self._skeleton_image)\n        return np.asarray(\n            [\n                self.kx_vals[point_mask[0]],\n                self.ky_vals[point_mask[1]],\n                self.kz_vals[point_mask[2]],\n            ]\n        ).T\n"""
        text = text.replace(old_coords, new_coords)
        path.write_text(text)

    init_path = ROOT / "src/knotted_graph/applications/nodal/__init__.py"
    init_text = init_path.read_text()
    patch = """from ._memory import install_memory_optimizations as _install_memory_optimizations\n\n_install_memory_optimizations(NodalSkeleton)\ndel _install_memory_optimizations\n\n"""
    if patch in init_text:
        init_path.write_text(init_text.replace(patch, ""))

    memory_path = ROOT / "src/knotted_graph/applications/nodal/_memory.py"
    if memory_path.exists():
        memory_path.unlink()

    test_path = ROOT / "tests/applications/test_nodal_regressions.py"
    test_text = test_path.read_text()
    if "test_memory_optimizations_are_native_class_implementation" not in test_text:
        test_text = test_text.rstrip() + r'''


def test_memory_optimizations_are_native_class_implementation():
    """Optimized behavior must not be installed by import-time monkeypatching."""
    assert NodalSkeleton.__init__.__module__ == skeleton_module.__name__
    assert NodalSkeleton.spectrum.func.__module__ == skeleton_module.__name__
    assert NodalSkeleton.skeleton_coords.func.__module__ == skeleton_module.__name__
    assert NodalSkeleton.kx_grid.__class__.__module__ == skeleton_module.__name__
''' + "\n"
        test_path.write_text(test_text)


def update_pyproject() -> None:
    path = ROOT / "pyproject.toml"
    text = path.read_text()
    old = '''dev = [\n    "pytest",\n    "ruff",\n]\ndocs = ['''
    new = '''dev = [\n    "pytest",\n    "pytest-cov",\n    "pyright",\n    "ruff",\n]\nbenchmark = [\n    "topoly==1.1.0",\n]\ndocs = ['''
    if old in text:
        text = text.replace(old, new)
    assert '"topoly==1.1.0"' in text
    assert '"pytest-cov"' in text and '"pyright"' in text

    if "[tool.pyright]" not in text:
        text += '''\n[tool.pyright]\ninclude = [\n    "src/knotted_graph/invariants/yamada/fast.py",\n]\npythonVersion = "3.11"\ntypeCheckingMode = "basic"\nreportMissingTypeStubs = false\n'''
    path.write_text(text)


def update_committed_corpus_benchmark() -> None:
    path = ROOT / "dev/benchmark_topoly_random_cubic_ensemble.py"
    text = path.read_text()
    if "DEFAULT_CORPUS" not in text:
        anchor = "DEFAULT_SEED = 20260818\n"
        assert text.count(anchor) == 1
        text = text.replace(
            anchor,
            anchor
            + "DEFAULT_CORPUS = (\n"
              "    Path(__file__).resolve().parent\n"
              "    / \"benchmark_data\"\n"
              "    / \"topoly_random_cubic_v1.jsonl\"\n"
              ")\n",
        )
        text = text.replace(
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass\nfrom pathlib import Path\n",
        )

        marker = "\n\ndef _seed(*parts: object) -> int:\n"
        assert text.count(marker) == 1
        loader = r'''


def load_committed_ensemble(
    vertex_count: int,
    n_samples: int,
    corpus_path: Path = DEFAULT_CORPUS,
) -> list[tuple[Sample, nx.Graph]]:
    """Load the versioned paper corpus instead of regenerating graph topology."""
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Committed benchmark corpus is missing: {corpus_path}. "
            "Regenerate it only with dev/generate_topoly_random_cubic_corpus.py."
        )
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = sorted(
        (row for row in rows if int(row["V"]) == vertex_count),
        key=lambda row: int(row["sample"]),
    )[:n_samples]
    if len(rows) != n_samples:
        raise AssertionError(
            f"corpus has {len(rows)} samples at V={vertex_count}, expected {n_samples}"
        )

    ensemble = []
    for row in rows:
        abstract = nx.from_graph6_bytes(row["graph6"].encode("ascii"))
        expected_edges = sorted(tuple(map(int, edge)) for edge in row["edge_list"])
        actual_edges = sorted((min(u, v), max(u, v)) for u, v in abstract.edges())
        if actual_edges != expected_edges:
            raise AssertionError(f"graph6/edge-list mismatch at V={vertex_count}")
        abstract.graph["_committed_benchmark"] = row
        sample = Sample(
            vertex_count=vertex_count,
            sample_index=int(row["sample"]),
            topology_seed=int(row["topology_seed"]),
            topology_attempt=int(row["topology_attempt"]),
        )
        ensemble.append((sample, abstract))
    return ensemble
'''
        text = text.replace(marker, loader + marker)

        prep_anchor = '''    if abstract.number_of_edges() != 3 * sample.vertex_count // 2:\n        raise AssertionError("cubic topology must satisfy E=3V/2")\n\n    last_error = None\n'''
        assert text.count(prep_anchor) == 1
        prep_new = '''    if abstract.number_of_edges() != 3 * sample.vertex_count // 2:\n        raise AssertionError("cubic topology must satisfy E=3V/2")\n\n    committed = abstract.graph.get("_committed_benchmark")\n    if committed is not None:\n        embedded = nx.MultiGraph()\n        positions = committed["node_positions"]\n        for node in sorted(abstract.nodes()):\n            embedded.add_node(\n                node,\n                pos=np.asarray(positions[str(int(node))], dtype=float),\n            )\n        for u, v in abstract.edges():\n            embedded.add_edge(\n                u,\n                v,\n                pts=np.vstack([embedded.nodes[u]["pos"], embedded.nodes[v]["pos"]]),\n            )\n        processor = PDCode(embedded)\n        pdcode = processor.compute(rotation_angles=(0.0, 0.0, 0.0))\n        if pdcode != committed["pdcode"]:\n            raise AssertionError(\n                f"committed PD drift at V={sample.vertex_count}, sample={sample.sample_index}"\n            )\n        if len(processor.crossings) != int(committed["crossings"]):\n            raise AssertionError("committed crossing-count drift")\n        if _abstract_hash(abstract) != committed["topology_instance_hash"]:\n            raise AssertionError("committed topology hash drift")\n        if _embedding_hash(embedded) != committed["embedding_hash"]:\n            raise AssertionError("committed embedding hash drift")\n        return embedded, processor, pdcode, int(committed["embedding_attempt"])\n\n    last_error = None\n'''
        text = text.replace(prep_anchor, prep_new)

        old = "        ensemble = topology_ensemble(vertex_count, samples_per_v, base_seed)\n"
        assert text.count(old) == 1
        text = text.replace(
            old,
            "        ensemble = load_committed_ensemble(vertex_count, samples_per_v)\n",
        )
        path.write_text(text)


def add_corpus_tests() -> None:
    path = ROOT / "tests/benchmarks/test_topoly_paper_setup.py"
    text = path.read_text()
    if "test_committed_random_cubic_corpus_is_complete" not in text:
        text = text.rstrip() + r'''


def test_committed_random_cubic_corpus_is_complete_and_reconstructs_pd():
    bench = _random_cubic()
    corpus = bench.DEFAULT_CORPUS
    assert corpus.exists()
    rows = [
        __import__("json").loads(line)
        for line in corpus.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == len(bench.vertex_grid("paper")) * bench.DEFAULT_SAMPLES

    for vertex_count in bench.vertex_grid("paper"):
        group = [row for row in rows if int(row["V"]) == vertex_count]
        assert len(group) == bench.DEFAULT_SAMPLES
        assert len({row["graph6"] for row in group}) == bench.DEFAULT_SAMPLES
        assert len({row["pdcode"] for row in group}) == bench.DEFAULT_SAMPLES

    # Reconstruct representative small, middle and large committed instances.
    for vertex_count in (10, 64, 200):
        sample, abstract = bench.load_committed_ensemble(vertex_count, 1)[0]
        embedded, processor, pdcode, _ = bench.prepare_sample(
            sample, abstract, bench.DEFAULT_SEED
        )
        row = abstract.graph["_committed_benchmark"]
        assert pdcode == row["pdcode"]
        assert len(processor.crossings) == int(row["crossings"])
        assert embedded.number_of_nodes() == vertex_count
        assert embedded.number_of_edges() == 3 * vertex_count // 2
''' + "\n"
        path.write_text(text)


def normalize_notebook_cell_ids() -> None:
    import hashlib

    for path in (ROOT / "User_guide").rglob("*.ipynb"):
        data = json.loads(path.read_text())
        changed = False
        used = {cell.get("id") for cell in data.get("cells", []) if cell.get("id")}
        for index, cell in enumerate(data.get("cells", [])):
            if not cell.get("id"):
                payload = f"{path.relative_to(ROOT)}:{index}:{''.join(cell.get('source', []))}"
                candidate = hashlib.sha1(payload.encode()).hexdigest()[:8]
                suffix = 0
                while candidate in used:
                    suffix += 1
                    candidate = hashlib.sha1(f"{payload}:{suffix}".encode()).hexdigest()[:8]
                cell["id"] = candidate
                used.add(candidate)
                changed = True
        if changed:
            path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    integrate_nodal_memory()
    update_pyproject()
    update_committed_corpus_benchmark()
    add_corpus_tests()
    normalize_notebook_cell_ids()


if __name__ == "__main__":
    main()
