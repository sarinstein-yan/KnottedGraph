import tempfile
import unittest
from pathlib import Path

import numpy as np

from knotted_graph.inputs import (
    from_gromacs_gro,
    from_lammps_dump,
    write_gro_coords,
    write_lammps_dump,
)


def edge_points(graph):
    return next(iter(graph.edges(data=True)))[2]["pts"]


class PolymerInputTests(unittest.TestCase):
    def test_lammps_dump_polymer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.dump"
            coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
            write_lammps_dump(coords, path, molecule_id=7)

            result = from_lammps_dump(path, molecule_id=7, polymer_id="chain")

            self.assertEqual(result.polymer_id, "chain")
            self.assertEqual(result.source_format, "lammps_dump")
            self.assertEqual(result.graph.graph["input_kind"], "polymer_snapshot")
            np.testing.assert_allclose(result.coords, coords)
            np.testing.assert_allclose(edge_points(result.graph), coords)
            self.assertEqual(result.issues, [])

    def test_lammps_dump_filters_molecule_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.dump"
            path.write_text(
                "\n".join(
                    [
                        "ITEM: TIMESTEP",
                        "0",
                        "ITEM: NUMBER OF ATOMS",
                        "4",
                        "ITEM: BOX BOUNDS pp pp pp",
                        "-1 2",
                        "-1 2",
                        "-1 2",
                        "ITEM: ATOMS id mol type x y z",
                        "1 7 1 0 0 0",
                        "2 8 1 9 9 9",
                        "3 7 1 1 0 0",
                        "4 7 1 1 1 0",
                    ]
                )
                + "\n"
            )

            result = from_lammps_dump(path, molecule_id=7)

            self.assertEqual(result.coords.shape, (3, 3))
            np.testing.assert_allclose(result.coords[:, 0], [0, 1, 1])

    def test_gromacs_gro_direct_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ring.gro"
            coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            write_gro_coords(coords, path, residue_name="RNG", atom_name="BB")

            result = from_gromacs_gro(
                path,
                atom_name="BB",
                residue_name="RNG",
                closed=True,
                closure="direct",
                polymer_id="ring",
            )
            pts = edge_points(result.graph)

            self.assertTrue(result.closed)
            self.assertEqual(pts.shape, (4, 3))
            np.testing.assert_allclose(pts[0], pts[-1])

    def test_gromacs_gro_closed_requires_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ring.gro"
            coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            write_gro_coords(coords, path, residue_name="RNG", atom_name="BB")

            with self.assertRaisesRegex(ValueError, "closure='direct'"):
                from_gromacs_gro(
                    path,
                    atom_name="BB",
                    residue_name="RNG",
                    closed=True,
                    polymer_id="ring",
                )


if __name__ == "__main__":
    unittest.main()
