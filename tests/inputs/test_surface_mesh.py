import tempfile
import unittest
from pathlib import Path

from knotted_graph.inputs import from_surface_mesh


def write_closed_tetrahedron_off(path):
    path.write_text(
        "\n".join(
            [
                "OFF",
                "4 4 0",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "3 0 2 1",
                "3 0 1 3",
                "3 1 2 3",
                "3 2 0 3",
            ]
        )
        + "\n"
    )


class SurfaceMeshInputTests(unittest.TestCase):
    def test_surface_mesh_from_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tetra.off"
            write_closed_tetrahedron_off(path)

            result = from_surface_mesh(path, mesh_id="tetra")

            self.assertEqual(result.mesh_id, "tetra")
            self.assertEqual(result.source_format, "off")
            self.assertEqual(result.mesh.n_points, 4)
            self.assertEqual(result.mesh.n_cells, 4)
            self.assertEqual(result.issues, [])

    def test_surface_mesh_unsupported_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.foo"
            path.write_text("not a mesh\n")

            with self.assertRaisesRegex(ValueError, "Unsupported mesh suffix"):
                from_surface_mesh(path)

    def test_surface_mesh_reports_open_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triangle.off"
            path.write_text(
                "\n".join(
                    [
                        "OFF",
                        "3 1 0",
                        "0 0 0",
                        "1 0 0",
                        "0 1 0",
                        "3 0 1 2",
                    ]
                )
                + "\n"
            )

            result = from_surface_mesh(path)

            self.assertTrue(any("open boundary edges" in issue for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
