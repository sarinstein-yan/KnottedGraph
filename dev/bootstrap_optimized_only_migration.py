from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = "".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "dev/.migration_payload").glob("part*.txt")))


def write_payload() -> None:
    files = json.loads(zlib.decompress(base64.b85decode(PAYLOAD.encode())).decode())
    for relative, content in files.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def replace_text(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def patch_nodal() -> None:
    path = ROOT / "src/knotted_graph/applications/nodal/skeleton.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("import skimage.morphology as morph\n", "")
    text = text.replace(
        "from knotted_graph.extraction import skeleton_image_to_graph\n",
        "from knotted_graph.extraction import skeleton_image_to_graph, skeletonize_volume\n",
    )
    pattern = re.compile(
        r"    @cached_property\n    def _skeleton_image\(self\) -> NDArray:\n.*?(?=    @cached_property\n    def skeleton_coords)",
        re.S,
    )
    replacement = '''    @cached_property
    def _skeleton_image(self) -> NDArray:
        """Optimized Lee skeleton in the original global voxel frame."""
        try:
            return skeletonize_volume(self._interior_mask)
        except ValueError as exc:
            if "does not contain any True voxels" in str(exc):
                raise ValueError(
                    "The skeleton image is empty. "
                    "Ensure the Hamiltonian has a non-empty exceptional surface."
                ) from exc
            raise

'''
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("could not replace NodalSkeleton._skeleton_image")
    path.write_text(text, encoding="utf-8")


def patch_material_base() -> None:
    path = ROOT / "src/knotted_graph/applications/_material_surface_base.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("import skimage.morphology as morph\n", "")
    text = text.replace(
        "from knotted_graph.extraction import skeleton_image_to_graph\n",
        "from knotted_graph.extraction import skeleton_image_to_graph, skeletonize_volume\n",
    )
    pattern = re.compile(
        r"    @cached_property\n    def _skeleton_image\(self\) -> NDArray:\n.*?(?=    @property\n    def berry_curvature)",
        re.S,
    )
    replacement = '''    @cached_property
    def _skeleton_image(self) -> NDArray:
        """Optimized Lee skeleton of the thickened nodal region."""
        try:
            return skeletonize_volume(self._interior_mask)
        except ValueError as exc:
            if "does not contain any True voxels" in str(exc):
                raise ValueError(
                    "The skeleton image is empty. "
                    "Try increasing gap_tol, checking band_pair, or enlarging the k-span."
                ) from exc
            raise

'''
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("could not replace MaterialFermiSurface._skeleton_image")
    path.write_text(text, encoding="utf-8")


def patch_material_public() -> None:
    path = ROOT / "src/knotted_graph/applications/material_surface.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("from skimage import morphology\n", "")
    pattern = re.compile(
        r"\n    @cached_property\n    def _skeleton_image\(self\).*?(?=\n    @cached_property|\n    def |\Z)",
        re.S,
    )
    text = pattern.sub("", text, count=1)
    path.write_text(text, encoding="utf-8")


def patch_notebooks_and_docs() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "dev/migrate_optimized_paths.py")],
        check=True,
        cwd=ROOT,
    )
    replace_text(
        "doc/developer/architecture.md",
        '`knotted_graph.extraction.skeleton_image_to_graph` is the canonical\nskeleton-to-graph entry point. For every 3-D image, `backend="auto"` uses the\nsecond-generation sparse extractor: empty image margins are cropped before\nforeground indexing, 26-neighbour adjacency is generated in exact historical\nlexicographic order, and returned coordinates remain in the original global\nvoxel frame. `knotted_graph.extraction.skeleton` exports the same function\nobjects, so the package and submodule import paths cannot diverge.\n\nThe historical `poly2graph.skeleton2graph` parser is retained only behind the\nexplicit `backend="poly2graph"` compatibility route used by regression and\nbenchmark code. It is not a normal 3-D production path.',
        '`knotted_graph.extraction.skeleton_image_to_graph` is the canonical\nskeleton-to-graph entry point and always uses the current sparse extractor.\nEmpty image margins are cropped before foreground indexing, 26-neighbour\nadjacency is generated deterministically, and returned coordinates remain in\nthe original global voxel frame. The obsolete selectable skeleton backend has\nbeen removed; historical behavior is preserved only by Git history and by the\nisolated `02_application_output_regression.ipynb` worktree comparison.',
    )
    replace_text(
        "dev/Architecture.md",
        'Every normal 3-D image uses the second-generation cropped exact-order sparse extractor. Explicit `backend="poly2graph"` is retained only for compatibility/regression. Optional bounded-valence persistence repair is enabled only when a real `max_junction_degree` is supplied.',
        'Every normal 3-D image uses the current cropped exact-order sparse extractor. The obsolete selectable skeleton backend has been removed; optional bounded-valence persistence repair is enabled only when a real `max_junction_degree` is supplied.',
    )
    replace_text(
        "dev/Architecture.md",
        '`src/knotted_graph/invariants/yamada/polynomial.py`, `src/knotted_graph/invariants/yamada/recursive.py`, `src/knotted_graph/invariants/yamada/compact.py`, `src/knotted_graph/invariants/yamada/native.py`, `src/knotted_graph/projection/pd_code.py::compute_yamada_polynomial`',
        '`src/knotted_graph/invariants/yamada/polynomial.py`, `src/knotted_graph/invariants/yamada/compact.py`, `src/knotted_graph/invariants/yamada/native.py`, `src/knotted_graph/projection/pd_code.py::compute_yamada_polynomial`',
    )


def delete_obsolete() -> None:
    for relative in (
        "src/knotted_graph/extraction/_legacy_skeleton.py",
        "src/knotted_graph/extraction/_sparse_compat.py",
        "tests/invariants/yamada/test_recursive.py",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def main() -> None:
    write_payload()
    patch_nodal()
    patch_material_base()
    patch_material_public()
    delete_obsolete()
    patch_notebooks_and_docs()
    print("PASS: optimized-only migration applied deterministically")


if __name__ == "__main__":
    main()
