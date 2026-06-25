import sys
import types
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
kg_pkg = types.ModuleType("knotted_graph")
kg_pkg.__path__ = [str(repo_root / "src" / "knotted_graph")]
sys.modules.setdefault("knotted_graph", kg_pkg)
repulsive_pkg = types.ModuleType("knotted_graph.repulsive_layout")
repulsive_pkg.__path__ = [str(repo_root / "src" / "knotted_graph" / "repulsive_layout")]
sys.modules.setdefault("knotted_graph.repulsive_layout", repulsive_pkg)

from knotted_graph.repulsive_layout.decimation import DecimationOptions, decimate_curve_network


def test_safe_downsampling_leaves_edges_without_targets_unchanged():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    edge_indices = {
        "targeted": [0, 2, 3, 4, 1],
        "untargeted": [0, 5, 6, 7, 1],
    }

    result = decimate_curve_network(
        vertices,
        edge_indices,
        ("targeted", "untargeted"),
        pinned_indices={0, 1},
        options=DecimationOptions(
            min_points_per_edge=2,
            max_points_per_edge={"targeted": 2},
            min_clearance=0.01,
            preserve_pinned_neighbors=False,
        ),
    )

    counts = result.report["edge_point_counts_after"]
    assert counts["targeted"] == 2
    assert counts["untargeted"] == 5
