import tempfile
import unittest
from pathlib import Path

import numpy as np

from knotted_graph.inputs import (
    from_mmcif_backbone,
    from_nucleic_acid_backbone,
    from_protein_ca_backbone,
)


def pdb_atom(serial, atom_name, residue_name, chain_id, resseq, x, y, z):
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id:1s}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        "  1.00 20.00           C\n"
    )


def edge_points(graph):
    return next(iter(graph.edges(data=True)))[2]["pts"]


class BiomolecularInputTests(unittest.TestCase):
    def test_protein_ca_backbone_from_local_pdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.pdb"
            path.write_text(
                "".join(
                    [
                        pdb_atom(1, "CA", "ALA", "A", 1, 0, 0, 0),
                        pdb_atom(2, "CA", "GLY", "A", 2, 1, 0, 0),
                        pdb_atom(3, "CA", "SER", "A", 3, 1, 1, 0),
                    ]
                )
            )

            result = from_protein_ca_backbone(path, pdb_id="TEST", chain_id="A")

            self.assertEqual(result.pdb_id, "TEST")
            self.assertEqual(result.chain_id, "A")
            self.assertEqual(result.atom_name, "CA")
            self.assertFalse(result.downloaded)
            self.assertEqual(result.graph.graph["input_kind"], "protein_ca_backbone")
            self.assertEqual(result.coords.shape, (3, 3))
            np.testing.assert_allclose(edge_points(result.graph), result.coords)
            self.assertEqual(result.issues, [])

    def test_protein_ca_backbone_requires_chain_when_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.pdb"
            path.write_text(
                "".join(
                    [
                        pdb_atom(1, "CA", "ALA", "A", 1, 0, 0, 0),
                        pdb_atom(2, "CA", "GLY", "A", 2, 1, 0, 0),
                        pdb_atom(3, "CA", "ALA", "B", 1, 0, 1, 0),
                        pdb_atom(4, "CA", "GLY", "B", 2, 1, 1, 0),
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "Multiple chains"):
                from_protein_ca_backbone(path, pdb_id="TEST")

    def test_nucleic_acid_backbone_filters_residue_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dna.pdb"
            path.write_text(
                "".join(
                    [
                        pdb_atom(1, "P", "DA", "A", 1, 0, 0, 0),
                        pdb_atom(2, "P", "DT", "A", 2, 1, 0, 0),
                        pdb_atom(3, "P", "ALA", "A", 3, 2, 0, 0),
                    ]
                )
            )

            result = from_nucleic_acid_backbone(path, pdb_id="DNA1", chain_id="A")

            self.assertEqual(result.graph.graph["input_kind"], "nucleic_acid_backbone")
            self.assertEqual(result.coords.shape, (2, 3))
            self.assertEqual([record["residue_name"] for record in result.records], ["DA", "DT"])

    def test_mmcif_backbone_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.cif"
            path.write_text(
                "\n".join(
                    [
                        "data_TEST",
                        "loop_",
                        "_atom_site.group_PDB",
                        "_atom_site.auth_atom_id",
                        "_atom_site.label_atom_id",
                        "_atom_site.auth_asym_id",
                        "_atom_site.label_asym_id",
                        "_atom_site.pdbx_PDB_model_num",
                        "_atom_site.label_alt_id",
                        "_atom_site.Cartn_x",
                        "_atom_site.Cartn_y",
                        "_atom_site.Cartn_z",
                        "_atom_site.auth_comp_id",
                        "_atom_site.label_comp_id",
                        "_atom_site.auth_seq_id",
                        "_atom_site.label_seq_id",
                        "ATOM P P A A 1 . 0.0 0.0 0.0 A A 1 1",
                        "ATOM P P A A 1 . 1.0 0.0 0.0 C C 2 2",
                        "ATOM P P A A 1 . 1.0 1.0 0.0 G G 3 3",
                        "#",
                    ]
                )
                + "\n"
            )

            result = from_mmcif_backbone(path, pdb_id="CIF1", chain_id="A", atom_name="P")

            self.assertEqual(result.pdb_id, "CIF1")
            self.assertEqual(result.source_url, None)
            self.assertEqual(result.graph.graph["input_kind"], "mmcif_backbone")
            self.assertEqual(result.coords.shape, (3, 3))
            self.assertEqual(result.issues, [])


if __name__ == "__main__":
    unittest.main()
