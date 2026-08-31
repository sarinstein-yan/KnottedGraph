"""Keep the canonical feature-status matrix aligned with public routes."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
STATUS_PAGE = ROOT / "doc" / "feature_status.md"
HEADERS = [
    "Starting object or goal",
    "Status",
    "Required extra",
    "Public call",
    "Return object",
    "Next step",
    "Runtime / scaling",
]
ALLOWED_STATUSES = {
    "Public · base",
    "Public · optional",
    "Application API · base",
    "Application API · optional",
    "External backend",
}
EXPECTED_ROUTE_FIELDS = {
    "Ordered coordinate array or CSV/DAT/JSON/NPY/TSV/TXT/XYZ": (
        "Public · base",
        "none",
        "`knotted_graph.inputs.from_coordinate_chain`",
        "`CoordinateInputResult`",
    ),
    "PDB ID or local PDB backbone": (
        "Public · base",
        "none",
        "`knotted_graph.inputs.from_pdb_backbone`; protein and nucleic-acid shortcuts are also available",
        "`PDBBackboneInputResult`",
    ),
    "RCSB ID or local CIF/mmCIF atom trace": (
        "Public · base",
        "none",
        "`knotted_graph.inputs.from_mmcif_backbone`",
        "`MMCIFBackboneInputResult`",
    ),
    "GROMACS GRO snapshot": (
        "Public · base",
        "none",
        "`knotted_graph.inputs.from_gromacs_gro`",
        "`PolymerInputResult`",
    ),
    "First frame of a LAMMPS dump": (
        "Public · base",
        "none",
        "`knotted_graph.inputs.from_lammps_dump`",
        "`PolymerInputResult`",
    ),
    "Paired node/edge CSV spatial graph": (
        "Public · base",
        "none",
        "`knotted_graph.inputs.from_spatial_graph_csv`",
        "`SpatialGraphInputResult`",
    ),
    "OBJ/OFF/PLY/STL/VTK/VTP surface mesh": (
        "Public · optional",
        "`surface`",
        "`knotted_graph.inputs.from_surface_mesh`",
        "`SurfaceInputResult`",
    ),
    "Embedded `networkx.MultiGraph` with node `pos` and edge `pts` data": (
        "Public · base",
        "none",
        "`knotted_graph.core.ensure_embedding`",
        "`networkx.MultiGraph`",
    ),
    "Validated embedded graph needing an inspectable projection": (
        "Public · base",
        "none",
        "`knotted_graph.projection.select_projection`",
        "`ProjectionResult`",
    ),
    "Embedded graph needing its Yamada invariant and provenance": (
        "Public · base",
        "none",
        "`knotted_graph.projection.compute_yamada_polynomial(..., return_result=True)`",
        "`YamadaComputationResult`",
    ),
    "Abstract undirected Graph/MultiGraph without crossing data": (
        "Public · base",
        "none",
        "`knotted_graph.invariants.yamada.compute_yamada_polynomial_recursive`",
        "`sympy.Expr`",
    ),
    "Named structured mathematical graph family": (
        "Application API · base",
        "none",
        "`knotted_graph.applications.mathematical.build_graph_case`",
        "`(networkx.MultiGraph, dict)`",
    ),
    "Already skeletonized 3-D boolean image": (
        "Public · optional",
        "`nodal`",
        "`knotted_graph.extraction.skeleton_image_to_graph`",
        "`networkx.MultiGraph`",
    ),
    "In-memory two-band non-Hermitian Hamiltonian or Bloch vector": (
        "Application API · optional",
        "`nodal`",
        "`knotted_graph.applications.nodal.NodalSkeleton`",
        "`NodalSkeleton`",
    ),
    "In-memory Hermitian multiband Hamiltonian": (
        "Application API · optional",
        "`nodal`",
        "`knotted_graph.applications.materials.MaterialFermiSurface`",
        "`MaterialFermiSurface`",
    ),
    "Embedded graph needing repulsive relaxation": (
        "External backend",
        "none for the direct graph call; separately installed native Repulsor solver",
        "`knotted_graph.layout.repulsive.relax_spatial_graph`",
        "`GraphLayoutResult`",
    ),
    "Embedded graph needing an interactive 3-D view": (
        "Public · optional",
        "`viz`",
        "`knotted_graph.visualization.plot_3D_graph_plotly`",
        "`plotly.graph_objects.Figure`",
    ),
}
BASE_CALLS = [
    ("knotted_graph.inputs", "from_coordinate_chain"),
    ("knotted_graph.inputs", "from_pdb_backbone"),
    ("knotted_graph.inputs", "from_mmcif_backbone"),
    ("knotted_graph.inputs", "from_gromacs_gro"),
    ("knotted_graph.inputs", "from_lammps_dump"),
    ("knotted_graph.inputs", "from_spatial_graph_csv"),
    ("knotted_graph.core", "ensure_embedding"),
    ("knotted_graph.projection", "select_projection"),
    ("knotted_graph.projection", "compute_yamada_polynomial"),
    ("knotted_graph.invariants.yamada", "compute_yamada_polynomial_recursive"),
    ("knotted_graph.applications.mathematical", "build_graph_case"),
    ("knotted_graph.layout.repulsive", "relax_spatial_graph"),
]
OPTIONAL_CALLS = [
    ("knotted_graph.inputs", "from_surface_mesh", ("pyvista",)),
    ("knotted_graph.extraction", "skeleton_image_to_graph", ("poly2graph",)),
    (
        "knotted_graph.applications.nodal",
        "NodalSkeleton",
        ("poly2graph", "pyvista", "skimage", "minorminer", "tabulate"),
    ),
    (
        "knotted_graph.applications.materials",
        "MaterialFermiSurface",
        ("poly2graph", "pyvista", "skimage", "minorminer", "tabulate"),
    ),
    ("knotted_graph.visualization", "plot_3D_graph_plotly", ("plotly",)),
]


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _load_rows() -> list[dict[str, str]]:
    lines = STATUS_PAGE.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if _split_table_row(line) == HEADERS
    )
    separator = _split_table_row(lines[header_index + 1])
    assert len(separator) == len(HEADERS)
    assert all(set(cell) <= {"-", ":"} for cell in separator)

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = _split_table_row(line)
        assert len(cells) == len(HEADERS)
        rows.append(dict(zip(HEADERS, cells, strict=True)))
    return rows


def test_feature_status_table_has_one_complete_route_per_row():
    rows = _load_rows()

    assert [row[HEADERS[0]] for row in rows] == list(EXPECTED_ROUTE_FIELDS)
    assert len({row[HEADERS[0]] for row in rows}) == 17
    assert {row["Status"] for row in rows} <= ALLOWED_STATUSES
    for row in rows:
        assert all(row[column] for column in HEADERS)
        expected = EXPECTED_ROUTE_FIELDS[row[HEADERS[0]]]
        assert (
            row["Status"],
            row["Required extra"],
            row["Public call"],
            row["Return object"],
        ) == expected


def test_feature_status_extras_exist_in_project_metadata():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]

    for row in _load_rows():
        documented = set(re.findall(r"`([^`]+)`", row["Required extra"]))
        assert documented <= set(extras)


@pytest.mark.parametrize(("module_name", "attribute"), BASE_CALLS)
def test_documented_base_calls_are_importable(module_name, attribute):
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))


@pytest.mark.parametrize(("module_name", "attribute", "requirements"), OPTIONAL_CALLS)
def test_documented_optional_calls_import_when_dependencies_are_installed(
    module_name, attribute, requirements
):
    missing = [name for name in requirements if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"optional dependencies not installed: {', '.join(missing)}")

    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))


def test_feature_status_is_linked_from_six_entry_points():
    entry_points = {
        "README.md": "doc/feature_status.md",
        "doc/index.md": "feature_status",
        "doc/quickstart.md": "feature_status",
        "doc/user_guide/index.md": "feature_status",
        "doc/installation.md": "feature_status",
        "doc/api/index.md": "feature_status",
    }

    for relative_path, expected in entry_points.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in text


def test_figure_only_formats_are_explicitly_outside_the_support_table():
    rows = _load_rows()
    starting_objects = "\n".join(row["Starting object or goal"] for row in rows).lower()
    rows_text = "\n".join(" ".join(row.values()) for row in rows).lower()
    full_text = STATUS_PAGE.read_text(encoding="utf-8").lower()
    boundaries = (
        "graphml",
        "generic edge lists",
        "swc",
        "spatial-graph json",
        "npz scalar/vector fields or volumes",
        "hamiltonian files",
        "oriented-flow files",
    )

    for boundary in boundaries:
        assert boundary not in starting_objects
        assert boundary in full_text
    assert "does not perform a generic" in full_text
    assert "mesh-to-graph conversion" in full_text
    assert "separately installed native repulsor solver" in rows_text
