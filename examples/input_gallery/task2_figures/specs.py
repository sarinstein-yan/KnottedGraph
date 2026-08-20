"""Immutable specifications for the current Main, S1, and S2 figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """One accepted panel image and its publication metadata."""

    key: str
    title: str
    asset: Path
    sha256: str
    formats: str | None = None
    polynomial: str | None = None


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """Grid and output contract for one publication figure."""

    key: str
    output_stem: str
    rows: int
    cols: int
    figsize: tuple[float, float]
    panels: tuple[PanelSpec, ...]
    show_formats: bool = False
    show_yamada: bool = False
    tight_bbox: bool = False

    def __post_init__(self) -> None:
        if len(self.panels) != self.rows * self.cols:
            raise ValueError(
                f"{self.key} requires {self.rows * self.cols} panels, "
                f"received {len(self.panels)}"
            )


def _panel(
    key: str,
    title: str,
    asset: str,
    sha256: str,
    *,
    formats: str | None = None,
    polynomial: str | None = None,
) -> PanelSpec:
    return PanelSpec(
        key=key,
        title=title,
        asset=Path(asset),
        sha256=sha256,
        formats=formats,
        polynomial=polynomial,
    )


MAIN = FigureSpec(
    key="main",
    output_stem="main_text_input_gallery_framed_v9",
    rows=2,
    cols=4,
    figsize=(7.25, 3.95),
    show_formats=True,
    panels=(
        _panel(
            "a_protein_backbone_pdb",
            "Biomolecular\nbackbones",
            "main_text_input_gallery_grouped_balanced_v8_panels/a_protein_backbone_pdb.png",
            "62f7ad20cd4dda3ced0601fa199efa9aada8d47798448f51971e5264fd9297f5",
            formats="PDB / mmCIF",
        ),
        _panel(
            "h_neuron_morphology_swc",
            "Neural & vascular\nnetworks",
            "main_text_input_gallery_grouped_balanced_v8_panels/h_neuron_morphology_swc.png",
            "9bd63e54cd60a696f0383b23b38d4fec2f23f62b3718b06fbfa8f7fd3aad9ad3",
            formats="SWC / NPY / NPZ",
        ),
        _panel(
            "d_polymer_lammps",
            "Polymer structures",
            "main_text_input_gallery_grouped_balanced_v8_panels/d_polymer_lammps.png",
            "8e9eaf7eb2fe4a4a719df6df13e86c7bfc7319b40bd88ba3ced7bb31af950596",
            formats="GRO / LAMMPS dump",
        ),
        _panel(
            "f_engineering_network_csv",
            "Engineering spatial\nnetworks",
            "main_text_input_gallery_grouped_balanced_v8_panels/f_engineering_network_csv.png",
            "820d3848ff370c85656862dac06a7691651db57e2c2f84fc0083c71f3c9f5f34",
            formats="CSV / JSON / GraphML",
        ),
        _panel(
            "t_fermi_surface_volume",
            "Hamiltonian-derived\nspatial graphs",
            "main_text_input_gallery_grouped_balanced_v8_panels/t_fermi_surface_volume.png",
            "7b21aef736ed64c15659c7d7411fc9861f76e884d528a41a2138183b8d58a4b1",
            formats="Hamiltonian / JSON / NPZ",
        ),
        _panel(
            "r_surface_volume_skeleton",
            "Surface & volume\nskeletons",
            "main_text_input_gallery_grouped_balanced_v8_panels/r_surface_volume_skeleton.png",
            "592303c26f42f7ec466f5da6c1f07705db45d4b0985d4a0649c0428b7a7beaff",
            formats="OBJ / OFF / PLY / STL\nVTK / VTP / NPY / NPZ",
        ),
        _panel(
            "s_oriented_vector_field",
            "Oriented fields\n& flows",
            "main_text_input_gallery_grouped_balanced_v8_panels/s_oriented_vector_field.png",
            "d09ecef40413f98880a5bb644d09c504c41d4bea3af6dcff743ee9f4c9c66c4f",
            formats="NPY / NPZ",
        ),
        _panel(
            "v_abstract_mathematical_graph",
            "Mathematical\nspatial graphs",
            "main_text_input_gallery_grouped_balanced_v8_panels/v_abstract_mathematical_graph.png",
            "a8083757505e16c6de1d383076bbfc2816124b25033129b0591d6e077b330c06",
            formats="CSV / JSON / GraphML",
        ),
    ),
)


S1 = FigureSpec(
    key="s1",
    output_stem="appendix_s1_yamada_nonzero_v9",
    rows=3,
    cols=5,
    figsize=(12.08, 6.82),
    show_yamada=True,
    tight_bbox=True,
    panels=(
        _panel(
            "c_trefoil_polymer_lammps",
            "Trefoil Polymer (LAMMPS)",
            "appendix_polymer_coordinate_panels_v8_yamada/c_trefoil_polymer_lammps_converted.png",
            "47dd9e7afa7bad4234992f837eb6a5c91875efb0cd040af14f29bfb75ed27626",
            polynomial="-A**11 + A**9 + A**8 + A**7 - A**4 - A**3 - A**2 - A - 1",
        ),
        _panel(
            "petersen",
            "Petersen Spatial Network\n(JSON)",
            "appendix_reorganized_v9_panels/petersen_converted.png",
            "e9d448cbcfb88c49d3eac7868889240c0ce7b235fa624e23fba2e1cdc968ced7",
            polynomial="-A**13 - A**12 + 5*A**11 - 15*A**10 + 25*A**9 - 35*A**8 + 40*A**7 - 40*A**6 + 35*A**5 - 25*A**4 + 15*A**3 - 5*A**2 + A + 1",
        ),
        _panel(
            "a_genus2_surface_ply",
            "Genus-2 Generating Spine (PLY)",
            "appendix_surface_volume_fermi_panels_v8_yamada/a_genus2_surface_ply_skeleton.png",
            "ac3b99594dd9ea62ef95d1abc4000e9e1bcf09ad29c83bd744ad2d144ec4fb25",
            polynomial="-A**4 - A**3 - 2*A**2 - A - 1",
        ),
        _panel(
            "mobius_ladder",
            "Moebius Ladder M8\n(GraphML)",
            "appendix_reorganized_v9_panels/mobius_ladder_converted.png",
            "da15d96db2209f772ed0f31191a69620be9213ccbd0601fdd4c2a546729ed577",
            polynomial="A**12 + A**11 - A**10 + 7*A**9 - 7*A**8 + 13*A**7 - 10*A**6 + 13*A**5 - 7*A**4 + 7*A**3 - A**2 + A + 1",
        ),
        _panel(
            "g_figure_eight_knot_xyz",
            "Figure-Eight Knot (XYZ)",
            "appendix_polymer_coordinate_panels_v8_yamada/g_figure_eight_knot_xyz_converted.png",
            "650b547abceafe2218bce07ff35f21954e3a898da2b457254bc8ef21495d109b",
            polynomial="-A**14 + A**12 - A**8 - A**7 - A**6 + A**2 - 1",
        ),
        _panel(
            "engineering_network_csv",
            "Engineering Network (CSV)",
            "appendix_spatial_graph_panels_v8_yamada/engineering_network_csv_converted.png",
            "9e9694a0594a311547b4e1e4829da41ff0c324860aeedc701c6ecee3b4717eca",
            polynomial="-A**8 + A**7 - A**6 + 2*A**5 + 2*A**3 + A**2 + A + 1",
        ),
        _panel(
            "f_cinquefoil_knot_xyz",
            "Cinquefoil Knot (XYZ)",
            "appendix_reorganized_v9_panels/f_cinquefoil_knot_xyz_converted.png",
            "34b91350f99b35d6e4c57063c772dbc9905342bbb8aa3e0a59a796758911007d",
            polynomial="-A**17 + A**13 + A**12 + A**11 - A**4 - A**3 - A**2 - A - 1",
        ),
        _panel(
            "warped_cube",
            "Warped Cubical Cage\n(GraphML)",
            "appendix_reorganized_v9_panels/warped_cube_converted.png",
            "1b824587391a082f94de44b4506fe21118a0bef552c1f0caca931a45fe1a4768",
            polynomial="-A**10 + 2*A**9 - 7*A**8 + 5*A**7 - 14*A**6 + 6*A**5 - 14*A**4 + 5*A**3 - 7*A**2 + 2*A - 1",
        ),
        _panel(
            "i_lissajous_loop_json",
            "Lissajous Loop (JSON)",
            "appendix_polymer_coordinate_panels_v8_yamada/i_lissajous_loop_json_converted.png",
            "eda78cc276dc4be56f443a927287e4186a8338c59631ac96a2fa318911d16c34",
            polynomial="-A**20 - A**17 + A**16 + A**15 - A**14 - A**10 - A**9 - A**7 + A**5 + A**2 - 1",
        ),
        _panel(
            "triple_lens",
            "Triple-Lens Necklace\n(CSV)",
            "appendix_reorganized_v9_panels/triple_lens_converted.png",
            "36b55f1909c79657c8903678110fc33d1b6d1263cceac821b20926e9b541fa29",
            polynomial="-A**8 - A**7 - 4*A**6 - 3*A**5 - 6*A**4 - 3*A**3 - 4*A**2 - A - 1",
        ),
        _panel(
            "spatial_graph_json",
            "Spatial Graph (JSON)",
            "appendix_spatial_graph_panels_v8_yamada/spatial_graph_json_converted.png",
            "1416bcf62eb43e311d8de59295c0582041265e934e82ee0e02bfca9a184d2583",
            polynomial="-A**14 - A**13 - A**12 - A**11 - 2*A**10 + 2*A**9 - 4*A**8 + 4*A**7 - 4*A**6 + 4*A**5 - A**4 + 2*A**3 + 2*A**2 + 1",
        ),
        _panel(
            "twisted_prism",
            "Twisted Triangular Prism\n(CSV)",
            "appendix_reorganized_v9_panels/twisted_prism_converted.png",
            "f48c043e784df59be296ac99fdee523ab3f93711a47655510ba68b26adf6357b",
            polynomial="-A**8 + A**7 - 3*A**6 + 2*A**5 - 4*A**4 + 2*A**3 - 3*A**2 + A - 1",
        ),
        _panel(
            "torus_link_3_3",
            "Three-Component Torus Link\nT(3,3) (JSON)",
            "appendix_reorganized_v9_panels/torus_link_3_3_converted.png",
            "25c007d2cac2a66682d1021a6a5d9e7dea232af7f21ceae6904b150d331de54b",
            polynomial="-A**15 - 3*A**14 - 3*A**13 - 3*A**12 - 2*A**11 - 2*A**10 - 2*A**9 - 2*A**8 - 2*A**7 - A**6 - A**5 - A**4 - A**3 - A**2 - A - 1",
        ),
        _panel(
            "spatial_network_graphml",
            "Spatial Network (GraphML)",
            "appendix_spatial_graph_panels_v8_yamada/spatial_network_graphml_converted.png",
            "8de5303d23728de6fe2635c4fec64a144305669614fdcb8145ceff1ff8bd1cb0",
            polynomial="-A**13 + 2*A**12 - A**11 + 3*A**9 - 4*A**8 + 6*A**7 - 5*A**6 + 5*A**5 - 2*A**4 + A**3 + 2*A**2 - A + 1",
        ),
        _panel(
            "lattice_truss_csv",
            "Lattice Truss (CSV)",
            "appendix_spatial_graph_panels_v8_yamada/lattice_truss_csv_converted.png",
            "dffeebad378afe4c28752b401f7dde92751b2b359ad54723a0d672fa62518764",
            polynomial="-A**6 - 2*A**4 - 2*A**2 - 1",
        ),
    ),
)


S2 = FigureSpec(
    key="s2",
    # Keep the accepted source stem used by the manuscript delivery mapping;
    # the public builder target and manuscript label are both S2.
    output_stem="appendix_s3_skeletonization_beyond_yamada_v9",
    rows=3,
    cols=4,
    figsize=(7.25, 5.45),
    tight_bbox=True,
    panels=(
        _panel(
            "a_crambin_pdb",
            "Crambin (PDB)",
            "appendix_biology_panels_v8_yamada/a_crambin_pdb_converted.png",
            "b284f1f9de86d987d8af4037e89f1a4b605897e9daff124ffc8dba58a9bae512",
        ),
        _panel(
            "b_ubiquitin_pdb",
            "Ubiquitin (PDB)",
            "appendix_biology_panels_v8_yamada/b_ubiquitin_pdb_converted.png",
            "2043b6c49e44f362bcac9a3eae895a67aa351d1b27e74e0a36480b3290564b99",
        ),
        _panel(
            "d_hemoglobin_pdb",
            "Hemoglobin (PDB)",
            "appendix_biology_panels_v8_yamada/d_hemoglobin_pdb_converted.png",
            "fad14fb521e202af3e22789c29131d56d9befb86f9200af06107f2a6cfa8d756",
        ),
        _panel(
            "e_b_dna_duplex_pdb",
            "B-DNA Duplex (PDB)",
            "appendix_biology_panels_v8_yamada/e_b_dna_duplex_pdb_converted.png",
            "7c915d05dc8dd14ee14342e945cbdb2db28e23fa4ffa45aef4967c7fb4b81e20",
        ),
        _panel(
            "f_trna_mmcif",
            "tRNA (mmCIF)",
            "appendix_biology_panels_v8_yamada/f_trna_mmcif_converted.png",
            "4e88ebacc7b342af6d361d0a86d50fcd7007eed51713c3183512e67d3acd24ce",
        ),
        _panel(
            "e_coiled_cable_dat",
            "Coiled Cable (DAT)",
            "appendix_polymer_coordinate_panels_v8_yamada/e_coiled_cable_dat_converted.png",
            "2d47fd7643482a4c39d4d7cb95389124fd99116fd702fa145c09c1dbf1135e40",
        ),
        _panel(
            "l_plain_text_cable_txt",
            "Plain Text Cable (TXT)",
            "appendix_polymer_coordinate_panels_v8_yamada/l_plain_text_cable_txt_converted.png",
            "ca6b748ec2fecfcd0c5aa396bfd341f0a5e85374e78b79e5c6e02ba88553a9c8",
        ),
        _panel(
            "vascular_branch",
            "Vascular Branch (CSV)",
            "appendix_spatial_graph_panels_v8_yamada/vascular_branch_csv_converted.png",
            "da7fa625a2f9ab8f066184bad2ada5bbb40f3e92469a10610f0aa725ed0d153b",
        ),
        _panel(
            "c_vector_flow_npz",
            "Integrated Flow Paths\n(NPZ)",
            "appendix_surface_volume_fermi_panels_v8_yamada/c_vector_flow_npz_oriented_graph.png",
            "c2ec0b98f322f5401c33ef7dff864ab53727581cb30828bd127851957966bd20",
        ),
        _panel(
            "d_gyroid_volume_npz",
            "Gyroid Phase Skeleton\n(NPZ)",
            "appendix_surface_volume_fermi_panels_v8_yamada/d_gyroid_volume_npz_skeleton.png",
            "9e119165735580ced56d78f0a883277cd9034e343c51122a60f59fa3952d0eb9",
        ),
        _panel(
            "e_schwarz_p_volume_npz",
            "Schwarz-P Phase Skeleton\n(NPZ)",
            "appendix_surface_volume_fermi_panels_v8_yamada/e_schwarz_p_volume_npz_skeleton.png",
            "afa3a75a2533850e10a525d23e04c0779981e7d6045e634b28d76c0076835566",
        ),
        _panel(
            "f_chiral_lattice_fermi_npz",
            "Hamiltonian Periodic\nSkeleton (NPZ)",
            "appendix_reorganized_v9_panels/f_chiral_lattice_fermi_npz_clean_skeleton.png",
            "f544c3e007d947d835c18f72d8a4c6a4175e052b679e8171625b53ddf8a29f57",
        ),
    ),
)


FIGURES = {spec.key: spec for spec in (MAIN, S1, S2)}
