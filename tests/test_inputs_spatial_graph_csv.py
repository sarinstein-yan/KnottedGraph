import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from knotted_graph.inputs import from_spatial_graph_csv


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_valid_graph(tmp_path):
    nodes = tmp_path / "nodes.csv"
    edges = tmp_path / "edges.csv"
    write_csv(
        nodes,
        [
            ["node_id", "x", "y", "z", "label", "type"],
            ["1", "0", "0", "0", "Component 1", "component"],
            ["2", "1", "0", "0", "Component 2", "component"],
            ["3", "1", "1", "0", "Component 3", "component"],
        ],
    )
    write_csv(
        edges,
        [
            ["edge_id", "source", "target", "label", "type"],
            ["e1", "1", "2", "Wire 1", "wire"],
            ["e2", "2", "3", "Pipe 1", "pipe"],
        ],
    )
    return nodes, edges


class SpatialGraphCsvInputTests(unittest.TestCase):
    def test_spatial_graph_csv_valid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes, edges = write_valid_graph(Path(tmp))

            result = from_spatial_graph_csv(nodes, edges, graph_id="demo", metadata={"domain": "circuit"})

            self.assertEqual(result.graph_id, "demo")
            self.assertEqual(result.graph.number_of_nodes(), 3)
            self.assertEqual(result.graph.number_of_edges(), 2)
            self.assertEqual(result.graph.graph["domain"], "circuit")
            np.testing.assert_allclose(result.graph.nodes["1"]["pos"], [0, 0, 0])
            self.assertEqual(result.graph.nodes["1"]["label"], "Component 1")
            self.assertEqual(result.graph.edges["1", "2", "e1"]["label"], "Wire 1")
            self.assertEqual(result.graph.edges["1", "2", "e1"]["type"], "wire")
            self.assertEqual(result.issues, [])

    def test_spatial_graph_csv_accepts_legacy_id_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes = tmp_path / "nodes.csv"
            edges = tmp_path / "edges.csv"
            write_csv(nodes, [["id", "x", "y", "z"], ["a", "0", "0", "0"], ["b", "1", "0", "0"]])
            write_csv(edges, [["id", "source", "target"], ["ab", "a", "b"]])

            result = from_spatial_graph_csv(nodes, edges)

            self.assertTrue(result.graph.has_edge("a", "b", "ab"))

    def test_spatial_graph_csv_missing_required_node_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes = tmp_path / "nodes.csv"
            edges = tmp_path / "edges.csv"
            write_csv(nodes, [["node_id", "x", "y"], ["1", "0", "0"]])
            write_csv(edges, [["edge_id", "source", "target"], ["e1", "1", "1"]])

            with self.assertRaisesRegex(ValueError, "missing coordinate columns"):
                from_spatial_graph_csv(nodes, edges)

    def test_spatial_graph_csv_non_numeric_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes = tmp_path / "nodes.csv"
            edges = tmp_path / "edges.csv"
            write_csv(nodes, [["node_id", "x", "y", "z"], ["1", "0", "bad", "0"]])
            write_csv(edges, [["edge_id", "source", "target"], ["e1", "1", "1"]])

            with self.assertRaisesRegex(ValueError, "must be numeric"):
                from_spatial_graph_csv(nodes, edges)

    def test_spatial_graph_csv_duplicate_node_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes = tmp_path / "nodes.csv"
            edges = tmp_path / "edges.csv"
            write_csv(
                nodes,
                [
                    ["node_id", "x", "y", "z"],
                    ["1", "0", "0", "0"],
                    ["1", "1", "0", "0"],
                ],
            )
            write_csv(edges, [["edge_id", "source", "target"], ["e1", "1", "1"]])

            with self.assertRaisesRegex(ValueError, "duplicate node ID"):
                from_spatial_graph_csv(nodes, edges)

    def test_spatial_graph_csv_duplicate_edge_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes, edges = write_valid_graph(tmp_path)
            write_csv(
                edges,
                [
                    ["edge_id", "source", "target"],
                    ["e1", "1", "2"],
                    ["e1", "2", "3"],
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate edge ID"):
                from_spatial_graph_csv(nodes, edges)

    def test_spatial_graph_csv_invalid_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes, edges = write_valid_graph(tmp_path)
            write_csv(edges, [["edge_id", "source", "target"], ["e1", "1", "missing"]])

            with self.assertRaisesRegex(ValueError, "unknown target"):
                from_spatial_graph_csv(nodes, edges)

    def test_spatial_graph_csv_points_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes, edges = write_valid_graph(tmp_path)
            write_csv(
                edges,
                [
                    ["edge_id", "source", "target", "points_json"],
                    ["curve", "1", "2", "[[0, 0, 0], [0.5, 0.2, 0.3], [1, 0, 0]]"],
                ],
            )

            result = from_spatial_graph_csv(nodes, edges)
            pts = result.graph.edges["1", "2", "curve"]["pts"]

            self.assertEqual(pts.shape, (3, 3))
            np.testing.assert_allclose(pts[1], [0.5, 0.2, 0.3])

    def test_spatial_graph_csv_points_json_must_match_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nodes, edges = write_valid_graph(tmp_path)
            write_csv(
                edges,
                [
                    ["edge_id", "source", "target", "points_json"],
                    ["curve", "1", "2", "[[0, 0, 0], [0.5, 0.2, 0.3], [2, 0, 0]]"],
                ],
            )

            with self.assertRaisesRegex(ValueError, "last point does not match target"):
                from_spatial_graph_csv(nodes, edges)


if __name__ == "__main__":
    unittest.main()
