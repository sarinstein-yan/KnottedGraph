from __future__ import annotations

import sys
import types
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
kg_pkg = types.ModuleType("knotted_graph")
kg_pkg.__path__ = [str(repo_root / "src" / "knotted_graph")]
sys.modules.setdefault("knotted_graph", kg_pkg)
repulsive_pkg = types.ModuleType("knotted_graph.repulsive_layout")
repulsive_pkg.__path__ = [str(repo_root / "src" / "knotted_graph" / "repulsive_layout")]
sys.modules.setdefault("knotted_graph.repulsive_layout", repulsive_pkg)

from knotted_graph.repulsive_layout.topology import verify_obj_step_sequence


def write_obj(path: Path, vertices: list[tuple[float, float, float]]) -> None:
    lines = [f"v {x} {y} {z}\n" for x, y, z in vertices]
    lines.extend(["l 1 2\n", "l 3 4\n"])
    path.write_text("".join(lines), encoding="utf-8")


def test_verify_obj_step_sequence_detects_swept_crossing(tmp_path):
    write_obj(
        tmp_path / "step_0000.obj",
        [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, -1.0),
            (0.0, -1.0, 1.0),
        ],
    )
    write_obj(
        tmp_path / "step_0001.obj",
        [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, -1.0),
            (0.0, 1.0, 1.0),
        ],
    )

    result = verify_obj_step_sequence(tmp_path, epsilon=1e-7)

    assert not result["verified"]
    assert result["violation_count"] == 1
    assert result["violations"][0]["tau"] == 0.5


def test_verify_obj_step_sequence_accepts_non_crossing_motion(tmp_path):
    write_obj(
        tmp_path / "step_0000.obj",
        [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, -1.0),
            (0.0, 2.0, 1.0),
        ],
    )
    write_obj(
        tmp_path / "step_0001.obj",
        [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 3.0, -1.0),
            (0.0, 3.0, 1.0),
        ],
    )

    result = verify_obj_step_sequence(tmp_path, epsilon=1e-7)

    assert result["verified"]
    assert result["violation_count"] == 0
