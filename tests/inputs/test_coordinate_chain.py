import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from knotted_graph.inputs import from_coordinate_chain


def edge_points(graph):
    return next(iter(graph.edges(data=True)))[2]["pts"]


class CoordinateChainInputTests(unittest.TestCase):
    def test_coordinate_chain_from_array_open_curve(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)

        result = from_coordinate_chain(coords, input_id="open_chain")

        self.assertEqual(result.input_id, "open_chain")
        self.assertEqual(result.source_format, "array")
        self.assertFalse(result.closed)
        self.assertEqual(result.graph.number_of_nodes(), 2)
        self.assertEqual(result.graph.number_of_edges(), 1)
        self.assertEqual({"start", "end"}, set(result.graph.nodes))
        np.testing.assert_allclose(edge_points(result.graph), coords)
        self.assertEqual(result.issues, [])

    def test_coordinate_chain_direct_closure_appends_first_point(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

        result = from_coordinate_chain(coords, input_id="triangle", closure="direct")
        pts = edge_points(result.graph)

        self.assertTrue(result.closed)
        self.assertEqual(result.graph.number_of_nodes(), 1)
        self.assertTrue(result.graph.has_node("loop_anchor"))
        self.assertEqual(pts.shape, (4, 3))
        np.testing.assert_allclose(pts[0], pts[-1])
        np.testing.assert_allclose(result.coords, coords)

    def test_coordinate_chain_marked_closed_requires_explicit_closure(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "closure='direct'"):
            from_coordinate_chain(coords, closed=True)

    def test_coordinate_chain_csv_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("x,y\n0,0\n1,1\n")

            with self.assertRaisesRegex(ValueError, "missing coordinate columns"):
                from_coordinate_chain(path)

    def test_coordinate_chain_csv_non_numeric_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("x,y,z\n0,0,0\n1,nope,0\n")

            with self.assertRaisesRegex(ValueError, "invalid coordinate value"):
                from_coordinate_chain(path)

    def test_coordinate_chain_json_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.json"
            coords = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
            path.write_text(json.dumps({"points": coords}))

            result = from_coordinate_chain(path)

            self.assertEqual(result.source_format, "json")
            np.testing.assert_allclose(result.coords, np.asarray(coords, dtype=float))

    def test_coordinate_chain_xyz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.xyz"
            path.write_text("3\nsmall chain\nC 0 0 0\nC 1 0 0\nC 1 1 0\n")

            result = from_coordinate_chain(path)

            self.assertEqual(result.source_format, "xyz")
            self.assertEqual(result.coords.shape, (3, 3))

    def test_coordinate_chain_xyz_allows_empty_comment_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.xyz"
            path.write_text("3\n\nC 0 0 0\nC 1 0 0\nC 1 1 0\n")

            result = from_coordinate_chain(path)

            np.testing.assert_allclose(
                result.coords,
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            )

    def test_coordinate_chain_xyz_reports_physical_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.xyz"
            path.write_text("3\n\nC 0 0 0\nC bad 0 0\nC 1 1 0\n")

            with self.assertRaisesRegex(ValueError, "line 4: invalid XYZ coordinate value"):
                from_coordinate_chain(path)

    def test_coordinate_chain_text_accepts_only_xyz_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = Path(tmp) / "valid.txt"
            valid_path.write_text("X Y Z\n0 0 0\n1 0 0\n")
            invalid_path = Path(tmp) / "invalid.txt"
            invalid_path.write_text("sample coordinates follow\n0 0 0\n1 0 0\n")

            result = from_coordinate_chain(valid_path)

            np.testing.assert_allclose(result.coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            with self.assertRaisesRegex(ValueError, "line 1: invalid coordinate value"):
                from_coordinate_chain(invalid_path)

    def test_coordinate_chain_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            from_coordinate_chain(np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
