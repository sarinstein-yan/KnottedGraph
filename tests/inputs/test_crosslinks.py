from pathlib import Path

import numpy as np
import pytest

from knotted_graph.core.embedding import validate_embedding
from knotted_graph.inputs import (
    ResidueKey,
    load_crosslinked_protein,
    parse_mmcif_crosslinks,
    parse_pdb_crosslinks,
)


def _pdb_atom(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    sequence_id: int,
    xyz: tuple[float, float, float],
    *,
    group: str = "ATOM",
) -> str:
    x, y, z = xyz
    element = atom_name.strip()[0]
    return (
        f"{group:<6}{serial:>5} {atom_name:^4} {residue_name:>3} {chain_id:1}"
        f"{sequence_id:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"  1.00 20.00          {element:>2}"
    )


def _pdb_ssbond(
    serial: int,
    chain_a: str,
    sequence_a: int,
    chain_b: str,
    sequence_b: int,
    *,
    distance: float = 2.03,
) -> str:
    chars = [" "] * 80
    chars[0:6] = list("SSBOND")
    chars[7:10] = list(f"{serial:>3}")
    chars[11:14] = list("CYS")
    chars[15] = chain_a
    chars[17:21] = list(f"{sequence_a:>4}")
    chars[25:28] = list("CYS")
    chars[29] = chain_b
    chars[31:35] = list(f"{sequence_b:>4}")
    chars[73:78] = list(f"{distance:>5.2f}")
    return "".join(chars).rstrip()


def _pdb_link(
    atom_a: str,
    residue_a: str,
    chain_a: str,
    sequence_a: int,
    atom_b: str,
    residue_b: str,
    chain_b: str,
    sequence_b: int,
) -> str:
    chars = [" "] * 80
    chars[0:6] = list("LINK  ")
    chars[12:16] = list(f"{atom_a:>4}")
    chars[17:20] = list(f"{residue_a:>3}")
    chars[21] = chain_a
    chars[22:26] = list(f"{sequence_a:>4}")
    chars[42:46] = list(f"{atom_b:>4}")
    chars[47:50] = list(f"{residue_b:>3}")
    chars[51] = chain_b
    chars[52:56] = list(f"{sequence_b:>4}")
    chars[73:78] = list(" 1.33")
    return "".join(chars).rstrip()


def _write_small_pdb(path: Path) -> None:
    lines = [
        _pdb_ssbond(1, "A", 2, "A", 4),
        _pdb_link("SG", "CYS", "A", 2, "SG", "CYS", "A", 4),
        _pdb_link("C", "GLY", "A", 1, "N", "CYS", "A", 2),
    ]
    serial = 1
    for sequence_id, residue_name in enumerate(["GLY", "CYS", "ALA", "CYS"], 1):
        lines.append(
            _pdb_atom(
                serial,
                "CA",
                residue_name,
                "A",
                sequence_id,
                (float(sequence_id - 1), 0.0, 0.0),
            )
        )
        serial += 1
        if residue_name == "CYS":
            lines.append(
                _pdb_atom(
                    serial,
                    "SG",
                    residue_name,
                    "A",
                    sequence_id,
                    (float(sequence_id - 1), 1.0, 0.0),
                )
            )
            serial += 1
    path.write_text("\n".join([*lines, "END", ""]))


def test_pdb_crosslinks_are_normalized_and_deduplicated(tmp_path):
    path = tmp_path / "small.pdb"
    _write_small_pdb(path)

    records = parse_pdb_crosslinks(path)

    assert [record.kind for record in records] == ["backbone_link", "disulfide"]
    disulfide = records[1]
    assert disulfide.source_record == "SSBOND"
    assert disulfide.endpoint_a.residue == ResidueKey("A", "2")
    assert disulfide.endpoint_b.residue == ResidueKey("A", "4")
    assert disulfide.distance == 2.03


def test_load_crosslinked_pdb_builds_valid_multigraph(tmp_path):
    path = tmp_path / "small.pdb"
    _write_small_pdb(path)

    result = load_crosslinked_protein(path, pdb_id="TEST", chain_ids=["A"])

    assert result.allowed_crosslink_types == ("covalent", "disulfide")
    assert [record.kind for record in result.crosslinks] == ["disulfide"]
    assert [record.kind for record in result.excluded_crosslinks] == ["backbone_link"]
    assert result.issues == []
    assert validate_embedding(result.graph) == []
    assert result.graph.number_of_nodes() == 3
    assert result.graph.number_of_edges() == 3
    edge_kinds = [data["edge_kind"] for *_, data in result.graph.edges(data=True)]
    assert edge_kinds.count("backbone") == 2
    assert edge_kinds.count("crosslink") == 1
    crosslink_data = next(
        data
        for *_, data in result.graph.edges(data=True)
        if data["edge_kind"] == "crosslink"
    )
    np.testing.assert_allclose(
        crosslink_data["pts"],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
    )


def test_loader_can_select_stable_crosslink_ids(tmp_path):
    path = tmp_path / "small.pdb"
    _write_small_pdb(path)

    result = load_crosslinked_protein(
        path,
        pdb_id="TEST",
        chain_ids=["A"],
        crosslink_ids={"ssbond:1"},
    )

    assert [record.crosslink_id for record in result.crosslinks] == ["ssbond:1"]
    assert result.metadata["requested_crosslink_ids"] == ("ssbond:1",)
    with pytest.raises(ValueError, match="Unknown crosslink IDs"):
        load_crosslinked_protein(
            path,
            pdb_id="TEST",
            chain_ids=["A"],
            crosslink_ids={"missing"},
        )


def test_mmcif_crosslinks_and_atoms_build_same_contract(tmp_path):
    path = tmp_path / "small.cif"
    path.write_text(
        """data_small
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_atom_id
_struct_conn.pdbx_dist_value
disulf1 disulf A 2 CYS SG
A 4 CYS SG 2.04
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.label_alt_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 CA GLY A 1 . 0 0 0 1
ATOM 2 CA CYS A 2 . 1 0 0 1
ATOM 3 SG CYS A 2 . 1 1 0 1
ATOM 4 CA ALA A 3 . 2 0 0 1
ATOM 5 CA CYS A 4 . 3 0 0 1
ATOM 6 SG CYS A 4 . 3 1 0 1
#
"""
    )

    parsed = parse_mmcif_crosslinks(path)
    result = load_crosslinked_protein(
        path,
        source_format="mmcif",
        pdb_id="TEST",
        chain_ids=["A"],
    )

    assert len(parsed) == 1
    assert parsed[0].kind == "disulfide"
    assert parsed[0].distance == 2.04
    assert result.crosslinks == parsed
    assert result.issues == []
    assert validate_embedding(result.graph) == []


def test_unrepresentable_crosslink_is_reported_and_excluded(tmp_path):
    path = tmp_path / "missing.pdb"
    _write_small_pdb(path)
    text = (
        path.read_text()
        .replace(
            _pdb_ssbond(1, "A", 2, "A", 4),
            _pdb_ssbond(1, "A", 2, "A", 9),
        )
        .replace(
            _pdb_link("SG", "CYS", "A", 2, "SG", "CYS", "A", 4) + "\n",
            "",
        )
    )
    path.write_text(text)

    result = load_crosslinked_protein(path, pdb_id="TEST", chain_ids=["A"])

    assert result.crosslinks == []
    assert len(result.excluded_crosslinks) == 2
    assert "crosslink endpoint A:9 has no representable atom" in result.issues


def test_metal_coordination_uses_heteroatom_as_graph_node(tmp_path):
    path = tmp_path / "metal.pdb"
    lines = [
        _pdb_link("OD1", "ASP", "A", 2, "MG", "MG", "Z", 9),
        _pdb_link("OE1", "GLU", "A", 4, "MG", "MG", "Z", 9),
    ]
    serial = 1
    for sequence_id, residue_name in enumerate(["GLY", "ASP", "ALA", "GLU"], 1):
        lines.append(
            _pdb_atom(
                serial,
                "CA",
                residue_name,
                "A",
                sequence_id,
                (float(sequence_id - 1), 0.0, 0.0),
            )
        )
        serial += 1
        if sequence_id in {2, 4}:
            atom_name = "OD1" if sequence_id == 2 else "OE1"
            lines.append(
                _pdb_atom(
                    serial,
                    atom_name,
                    residue_name,
                    "A",
                    sequence_id,
                    (float(sequence_id - 1), 1.0, 0.0),
                )
            )
            serial += 1
    lines.append(_pdb_atom(serial, "MG", "MG", "Z", 9, (2.0, 2.0, 0.0), group="HETATM"))
    path.write_text("\n".join([*lines, "END", ""]))

    result = load_crosslinked_protein(
        path,
        pdb_id="TEST",
        chain_ids=["A"],
        allowed_crosslink_types={"metal_coordination"},
    )

    assert len(result.crosslinks) == 2
    assert result.issues == []
    metal = ResidueKey("Z", "9")
    assert result.graph.nodes[metal]["node_type"] == "metal_center"
    assert result.graph.degree[metal] == 2
    assert validate_embedding(result.graph) == []


def test_reused_ligand_atom_gets_deterministic_general_position_lanes(tmp_path):
    path = tmp_path / "shared_atom.pdb"
    lines = [
        _pdb_link("OD1", "ASP", "A", 2, "MG", "MG", "Z", 9),
        _pdb_link("OD1", "ASP", "A", 2, "MG", "MG", "Z", 10),
        _pdb_atom(1, "CA", "GLY", "A", 1, (0.0, 0.0, 0.0)),
        _pdb_atom(2, "CA", "ASP", "A", 2, (1.0, 0.0, 0.0)),
        _pdb_atom(3, "OD1", "ASP", "A", 2, (1.0, 1.0, 0.0)),
        _pdb_atom(4, "CA", "GLY", "A", 3, (2.0, 0.0, 0.0)),
        _pdb_atom(5, "MG", "MG", "Z", 9, (0.5, 2.0, 0.5), group="HETATM"),
        _pdb_atom(6, "MG", "MG", "Z", 10, (1.5, 2.0, -0.5), group="HETATM"),
    ]
    path.write_text("\n".join([*lines, "END", ""]))

    result = load_crosslinked_protein(
        path,
        pdb_id="TEST",
        chain_ids=["A"],
        allowed_crosslink_types={"metal_coordination"},
    )

    edges = sorted(
        (
            data["crosslink_id"],
            np.asarray(data["pts"]),
            data["general_position_offsets"],
        )
        for *_, data in result.graph.edges(data=True)
        if data["edge_kind"] == "crosslink"
    )
    assert len(edges) == 2
    assert result.graph.graph["general_position_offset_count"] == 2
    assert result.graph.graph["general_position_max_offset"] > 0.0
    assert set(edges[0][2]) == {"a"}
    assert set(edges[1][2]) == {"a"}
    assert not np.allclose(edges[0][1][1], edges[1][1][1])
    assert validate_embedding(result.graph) == []
