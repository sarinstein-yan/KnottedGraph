"""Build the manuscript-oriented KnottedGraph companion notebook.

This notebook is not a website export. It is organized around the paper story:
geometric input -> spatial graph -> planar diagram -> PD code -> Yamada
polynomial -> research interpretation.
"""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
USER_GUIDE_DIR = ROOT / "User_guide"
USER_GUIDE_OUT = USER_GUIDE_DIR / "user_guide.ipynb"
COVERAGE_OUT = ROOT / "notebook_source_coverage.json"
CELL_COUNTER = 0


LEGACY_ROOT_NOTEBOOKS = [
    ROOT / "KnottedGraph_Paper_Companion.ipynb",
    ROOT / "KnottedGraph_Website_Examples.ipynb",
]


SECTION_NOTEBOOKS = {
    0: ("00_setup_preflight.ipynb", "Setup, Preflight, And Shared Plot Style"),
    1: ("01_quick_start.ipynb", "Quick Start: Surface To Yamada"),
    2: ("02_input_adapters.ipynb", "Input Adapter Gallery"),
    3: ("03_inspection_mode.ipynb", "Inspection Mode"),
    4: ("04_physical_fields.ipynb", "Physical Fields From Hamiltonians"),
    5: ("05_projection_pd_yamada.ipynb", "Projection, PD Codes, And Yamada"),
    6: ("06_mathematical_workflows.ipynb", "Mathematical Workflows"),
    7: ("07_repulsive_curves_and_proteins.ipynb", "Repulsive Curves And Proteins"),
    8: ("08_application_gallery.ipynb", "Application Gallery"),
    9: ("09_paper_figure_map.ipynb", "Paper Figure Reproduction Map"),
    10: ("10_appendix_workflows.ipynb", "Appendix-Style Figure Workflows"),
    11: ("11_advanced_diagnostics.ipynb", "Advanced Projection And Yamada Diagnostics"),
    12: ("12_library_improvements.ipynb", "Library Improvements"),
}


SECTION_DEPENDENCIES = {
    2: [1],
    5: [1],
    6: [3],
    11: [1, 2],
}

NODAL_SECTION_IMPORTS = {4, 8, 9, 10}


LOCAL_SOURCE_NOTEBOOKS = [
    (
        "surface skeletonization pipeline",
        Path(
            "/Users/hakanakgun/Desktop/Projects/ProfLeeProjects/Knotted_graph_code_paper/"
            "KnottedGraph_latest_git_for_codex/CodePaperFigureGeneration/Archive/"
            "surface_skeletonization_pipeline_awesome_knotted_graph_minimal.ipynb"
        ),
    ),
    (
        "PD-code emergence",
        Path(
            "/Users/hakanakgun/Desktop/Projects/ProfLeeProjects/Knotted_graph_code_paper/"
            "KnottedGraph_latest_git_for_codex/CodePaperFigureGeneration/Archive/"
            "pd_code_emergence_figure.ipynb"
        ),
    ),
    (
        "Yamada calculation",
        Path(
            "/Users/hakanakgun/Desktop/Projects/ProfLeeProjects/Knotted_graph_code_paper/"
            "KnottedGraph_latest_git_for_codex/CodePaperFigureGeneration/Archive/"
            "YamadaCalculation.ipynb"
        ),
    ),
    (
        "Yamada route details",
        Path(
            "/Users/hakanakgun/Desktop/Projects/ProfLeeProjects/Knotted_graph_code_paper/"
            "KnottedGraph_latest_git_for_codex/CodePaperFigureGeneration/Archive/"
            "YamadaCalculation_Route1Details.ipynb"
        ),
    ),
    (
        "projection and randomized scaling",
        Path(
            "/Users/hakanakgun/Desktop/Projects/ProfLeeProjects/Knotted_graph_code_paper/"
            "KnottedGraph_latest_git_for_codex/CodePaperFigureGeneration/Archive/"
            "YamadaRandomizedScalingAndProjectionFigures.ipynb"
        ),
    ),
    (
        "physics appendix figure 2",
        Path(
            "/Users/hakanakgun/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
            "xwechat_files/wxid_pk71tth5hmu022_6378/msg/file/2026-06/"
            "Figures_HakanPart(1)/Fig2-appendix.ipynb"
        ),
    ),
    (
        "physics appendix gallery",
        Path(
            "/Users/hakanakgun/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
            "xwechat_files/wxid_pk71tth5hmu022_6378/msg/file/2026-06/"
            "Figures_HakanPart(1)/FigsAppendix.ipynb"
        ),
    ),
    (
        "finished appendix figure workflows",
        Path(
            "/Users/hakanakgun/Desktop/Projects/ProfLeeProjects/Finished/Knotted_Graphs/"
            "Results/APPENDIXFIGURES/plot/FigsAppendix.ipynb"
        ),
    ),
    (
        "multiband material notebook",
        Path(
            "/Users/hakanakgun/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
            "xwechat_files/wxid_pk71tth5hmu022_6378/msg/file/2026-06/"
            "Figures_HakanPart(1)/Multiband.ipynb"
        ),
    ),
    ("structured mathematical dataset", Path("/Users/hakanakgun/Downloads/math_dataset.ipynb")),
]


BRANCH_SOURCE_FILES = [
    ("Latest_Workplace docs", "origin/Latest_Workplace", "README.md"),
    ("Latest_Workplace docs", "origin/Latest_Workplace", "KnottedGraph_Website_Examples.ipynb"),
    ("Latest_Workplace docs", "origin/Latest_Workplace", "doc/user_guide/input_adapters.md"),
    ("Latest_Workplace docs", "origin/Latest_Workplace", "doc/user_guide/repulsive_layout.md"),
    ("Latest_Workplace docs", "origin/Latest_Workplace", "doc/user_guide/projection_yamada.md"),
    ("Latest_Workplace applications", "origin/Latest_Workplace", "doc/applications/nodal_skeleton.md"),
    ("Latest_Workplace applications", "origin/Latest_Workplace", "doc/applications/biomolecular_protein_workflow.md"),
    ("Latest_Workplace applications", "origin/Latest_Workplace", "doc/applications/material_fingerprints.md"),
    ("Latest_Workplace applications", "origin/Latest_Workplace", "doc/applications/mathematical_workflows.md"),
    ("legacy nodal workflow", "origin/legacy-main", "getting_started.ipynb"),
    ("legacy nodal workflow", "origin/legacy-main", "README.md"),
    ("legacy nodal workflow", "origin/legacy-main", "src/knotted_graph/examples.py"),
    ("legacy nodal workflow", "origin/legacy-main", "src/knotted_graph/surface_modes.py"),
    ("legacy materials", "origin/legacy-main", "src/knotted_graph/NodalSkeletonMultiBand.py"),
    ("legacy repulsive protein", "origin/legacy-main", "Repulsion_protein.ipynb"),
    ("legacy math dataset", "origin/legacy-main", "math_dataset.ipynb"),
    ("input adapters", "origin/input-adapter", "examples/coordinate_chains/README.md"),
    ("input adapters", "origin/input-adapter", "examples/coordinate_chains/plot_coordinate_chain_examples.py"),
    ("input adapters", "origin/input-adapter", "examples/dna/README.md"),
    ("input adapters", "origin/input-adapter", "examples/mmcif/README.md"),
    ("input adapters", "origin/input-adapter", "examples/polymers/README.md"),
    ("input adapters", "origin/input-adapter", "examples/proteins/README.md"),
    ("input adapters", "origin/input-adapter", "examples/spatial_graphs/README.md"),
    ("input adapters", "origin/input-adapter", "examples/surfaces/README.md"),
    ("input adapters", "origin/input-adapter", "examples/volumetric_fields/README.md"),
    ("input adapters", "origin/input-adapter", "examples/fermi_surfaces/README.md"),
    ("input adapters", "origin/input-adapter", "examples/electric_circuits/README.md"),
    ("input adapters", "origin/input-adapter", "examples/input_gallery/README.md"),
    ("input adapters", "origin/input-adapter", "src/knotted_graph/inputs/coordinate_chain.py"),
    ("input adapters", "origin/input-adapter", "src/knotted_graph/inputs/pdb.py"),
    ("input adapters", "origin/input-adapter", "src/knotted_graph/inputs/mmcif.py"),
    ("input adapters", "origin/input-adapter", "src/knotted_graph/inputs/polymer.py"),
    ("input adapters", "origin/input-adapter", "src/knotted_graph/inputs/spatial_graph_csv.py"),
    ("input adapters", "origin/input-adapter", "src/knotted_graph/inputs/surface_mesh.py"),
    ("repulsive curves", "origin/add-repulsive-curves", "Repulsion_protein.ipynb"),
    ("repulsive curves", "origin/add-repulsive-curves", "external/repulsive-curves/python/examples.py"),
    ("repulsive curves", "origin/add-repulsive-curves", "external/repulsive-curves/python/run_examples.py"),
    ("repulsive curves", "origin/add-repulsive-curves", "external/repulsive-curves/python/render_layout.py"),
    ("repulsive curves", "origin/add-repulsive-curves", "src/knotted_graph/repulsive_layout/protein_examples.py"),
    ("generic dev baseline", "origin/generic-dev", "getting_started.ipynb"),
    ("current main comparison", "origin/main", "doc/applications/nodal_skeleton.md"),
]


KEYWORDS = [
    "NodalSkeleton",
    "NodalSkeletonMultiBand",
    "berry",
    "surface_modes",
    "PDCode",
    "compute_yamada",
    "sample_projections",
    "select_projection",
    "repulsive",
    "protein",
    "mmcif",
    "polymer",
    "spatial_graph_csv",
    "surface_mesh",
    "Hamiltonian",
    "material",
    "planarity",
    "intrinsic",
]


def next_cell_id() -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"kg-paper-{CELL_COUNTER:05d}"


def md(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str, *, tags: list[str] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "id": next_cell_id(),
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.strip() + "\n",
    }


def _source_summary(source: str) -> dict[str, Any]:
    lower = source.lower()
    return {
        "sha12": hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest()[:12],
        "lines": source.count("\n") + 1 if source else 0,
        "first_line": next((line.strip() for line in source.splitlines() if line.strip()), ""),
        "keywords": [key for key in KEYWORDS if key.lower() in lower],
    }


def _git_show(ref: str, path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def _notebook_rows(topic: str, label: str, notebook_text: str) -> list[dict[str, Any]]:
    try:
        nb = json.loads(notebook_text)
    except json.JSONDecodeError:
        return [
            {
                "topic": topic,
                "source": label,
                "kind": "notebook",
                "cell_index": None,
                "cell_type": "parse_error",
                **_source_summary(""),
            }
        ]

    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", ""))
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        summary = _source_summary(source)
        if cell.get("cell_type") == "code" or summary["keywords"]:
            rows.append(
                {
                    "topic": topic,
                    "source": label,
                    "kind": "notebook",
                    "cell_index": index,
                    "cell_type": cell.get("cell_type"),
                    **summary,
                }
            )
    return rows


def collect_source_coverage() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for topic, path in LOCAL_SOURCE_NOTEBOOKS:
        label = str(path)
        if not path.exists():
            rows.append(
                {
                    "topic": topic,
                    "source": label,
                    "kind": "local_notebook",
                    "cell_index": None,
                    "cell_type": "missing",
                    **_source_summary(""),
                }
            )
            continue
        rows.extend(_notebook_rows(topic, label, path.read_text(encoding="utf-8")))

    for topic, ref, path in BRANCH_SOURCE_FILES:
        label = f"{ref}:{path}"
        text = _git_show(ref, path)
        if text is None:
            rows.append(
                {
                    "topic": topic,
                    "source": label,
                    "kind": "git_source",
                    "cell_index": None,
                    "cell_type": "missing",
                    **_source_summary(""),
                }
            )
            continue
        if path.endswith(".ipynb"):
            rows.extend(_notebook_rows(topic, label, text))
        else:
            rows.append(
                {
                    "topic": topic,
                    "source": label,
                    "kind": "git_source",
                    "cell_index": None,
                    "cell_type": Path(path).suffix.lstrip(".") or "text",
                    **_source_summary(text),
                }
            )

    return rows


def notebook() -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cells: list[dict[str, Any]] = []

    cells.extend(
        [
            md(
                r"""
# KnottedGraph Paper Companion Notebook

This notebook is the manuscript-facing companion to `KnottedGraph`.  It is
organized around the computational claim of the paper:

$$
\text{surface / geometry / data}
\longrightarrow
G\subset\mathbb{R}^3
\longrightarrow
D(G)
\longrightarrow
\operatorname{PD}(G)
\longrightarrow
\Upsilon(G;Y)
\longrightarrow
\text{research interpretation}.
$$

The notebook is designed for `Run All`.  Runnable cells use public package
interfaces.  Blocks that require paper-only helpers or an external repulsive
layout driver are shown as reference code and are clearly labeled.

Generated from the local checkout on **%s**.
"""
                % now
            ),
            md(
                """
## 0. Setup, Preflight, And Shared Plot Style

The whole notebook uses the same visual convention:

- blue: surfaces, skeleton points, and graph edges;
- red: graph vertices;
- black axes;
- paper notation: `Upsilon(G; Y)`.

The helper functions in this section remove repeated plotting boilerplate from
the rest of the notebook.  This is also the library-level pattern worth
promoting later into public visualization helpers.
"""
            ),
            code(
                r"""
from pathlib import Path
import sys
import importlib.util
import os
import tempfile

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DOC_ROOT = PROJECT_ROOT / "doc"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "knottedgraph-mpl"))

print("project paths configured")
for package in ["numpy", "networkx", "sympy", "plotly", "matplotlib", "pyvista"]:
    print(f"{package:10s} = {importlib.util.find_spec(package) is not None}")
""",
                tags=["setup"],
            ),
            code(
                r"""
import math
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import sympy as sp
from IPython.display import Math, display

from knotted_graph.projection import (
    compute_yamada_polynomial,
    sample_projections,
    select_projection,
)
from knotted_graph.visualization import plot_3D_graph_plotly

BLUE = "#1f77b4"
RED = "#d62728"
CAMERA = dict(eye=dict(x=1.45, y=1.55, z=1.18))
pio.renderers.default = "notebook_connected"
Y = sp.Symbol("Y")
kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)


def axis_style():
    return dict(
        visible=True,
        title="",
        showticklabels=False,
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=2,
    )


def apply_kg_layout(fig, *, width=760, height=620):
    fig.update_layout(
        title=None,
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=axis_style(),
            yaxis=axis_style(),
            zaxis=axis_style(),
            aspectmode="data",
            camera=CAMERA,
        ),
    )
    return fig


def plot_surface_polydata(surface, *, opacity=0.58):
    mesh = surface.triangulate()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    pts = mesh.points
    fig = go.Figure(
        go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=BLUE,
            opacity=opacity,
        )
    )
    return apply_kg_layout(fig)


def plot_points_3d(points, *, size=3):
    points = np.asarray(points)
    fig = go.Figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=size, color=BLUE),
        )
    )
    return apply_kg_layout(fig)


def plot_graph_kg(graph):
    return apply_kg_layout(plot_3D_graph_plotly(graph))


def print_upsilon(label, expr):
    print(f"Upsilon({label}; Y) = {sp.expand(expr)}")


def display_bloch_vector(label, components):
    display(Math(label + r"=" + sp.latex(sp.Matrix(components))))


print("shared plotting and notation helpers ready")
"""
            ),
            md(
                """
## 1. Minimal Pipeline: Surface To Spatial Graph To `Upsilon(G;Y)`

This section is the first ten-minute demonstration of the whole library.  The
example is a synthetic tube surface whose spine is a trivalent K4-type
spatial graph.  It is intentionally richer than a toy loop: it has four real
branch vertices, six edges, nontrivial projection crossings, and a nonconstant
Yamada polynomial.
"""
            ),
            code(
                r"""
def trivalent_k4_spine(samples=90, amplitude=0.75):
    vertices = {
        "a": np.array([-1.15, -0.78, -0.38]),
        "b": np.array([1.18, -0.64, 0.30]),
        "c": np.array([0.86, 0.95, -0.26]),
        "d": np.array([-0.88, 0.84, 0.52]),
    }
    edge_specs = [
        ("a", "b", "ab", np.array([0.00, 0.90, 0.70]), 0.0),
        ("a", "c", "ac", np.array([0.35, -0.15, 1.00]), 1.1),
        ("a", "d", "ad", np.array([0.95, 0.15, -0.25]), 2.2),
        ("b", "c", "bc", np.array([-0.90, 0.25, 0.35]), 0.7),
        ("b", "d", "bd", np.array([-0.20, 1.00, -0.60]), 1.7),
        ("c", "d", "cd", np.array([0.10, -0.90, -0.85]), 2.8),
    ]

    s = np.linspace(0.0, 1.0, samples)
    graph = nx.MultiGraph()
    for vertex_id, pos in vertices.items():
        graph.add_node(vertex_id, pos=pos)

    for u, v, key, bend, phase in edge_specs:
        start = vertices[u]
        end = vertices[v]
        chord = end - start
        bend = bend / np.linalg.norm(bend)
        side = np.cross(chord, bend)
        side = side / np.linalg.norm(side)
        envelope = np.sin(np.pi * s)
        pts = (1 - s)[:, None] * start + s[:, None] * end
        pts += amplitude * envelope[:, None] * (
            np.cos(phase + np.pi * s)[:, None] * bend
            + 0.6 * np.sin(2 * np.pi * s + phase)[:, None] * side
        )
        pts[0] = start
        pts[-1] = end
        graph.add_edge(u, v, key=key, pts=pts)

    graph.graph.update(
        graph_id="paper_companion_trivalent_k4",
        input_kind="synthetic_surface_spine",
        is_closed=True,
    )
    return graph


def tube_patch(points, radius=0.11, sides=28):
    tangents = np.gradient(points, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    nearly_parallel = np.abs(np.sum(tangents * reference, axis=1)) > 0.92
    reference[nearly_parallel] = np.array([0.0, 1.0, 0.0])
    normals = np.cross(tangents, reference)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    binormals = np.cross(tangents, normals)

    theta = np.linspace(0.0, 2 * np.pi, sides, endpoint=True)
    circle = (
        np.cos(theta)[None, :, None] * normals[:, None, :]
        + np.sin(theta)[None, :, None] * binormals[:, None, :]
    )
    tube = points[:, None, :] + radius * circle
    return {"x": tube[:, :, 0], "y": tube[:, :, 1], "z": tube[:, :, 2]}


def sphere_patch(center, radius=0.18, samples=28):
    phi = np.linspace(0.0, np.pi, samples)
    theta = np.linspace(0.0, 2 * np.pi, samples)
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    return {
        "x": center[0] + radius * np.sin(phi) * np.cos(theta),
        "y": center[1] + radius * np.sin(phi) * np.sin(theta),
        "z": center[2] + radius * np.cos(phi),
    }


def trivalent_k4_surface_graph(tube_radius=0.12):
    graph = trivalent_k4_spine()
    tube_surfaces = [
        tube_patch(data["pts"], radius=tube_radius)
        for _, _, data in graph.edges(data=True)
    ]
    vertex_surfaces = [
        sphere_patch(data["pos"], radius=1.45 * tube_radius)
        for _, data in graph.nodes(data=True)
    ]
    return tube_surfaces, vertex_surfaces, graph


tube_surfaces, vertex_surfaces, graph = trivalent_k4_surface_graph()
print(f"surface tube patches = {len(tube_surfaces)}")
print(f"surface vertex patches = {len(vertex_surfaces)}")
print(f"nodes_edges = {(graph.number_of_nodes(), graph.number_of_edges())}")
print(f"degrees = {dict(graph.degree())}")
"""
            ),
            code(
                r"""
fig = go.Figure()
for patch in tube_surfaces:
    fig.add_trace(
        go.Surface(
            x=patch["x"],
            y=patch["y"],
            z=patch["z"],
            surfacecolor=np.zeros_like(patch["x"]),
            colorscale=[[0, BLUE], [1, BLUE]],
            showscale=False,
            opacity=0.42,
        )
    )
for patch in vertex_surfaces:
    fig.add_trace(
        go.Surface(
            x=patch["x"],
            y=patch["y"],
            z=patch["z"],
            surfacecolor=np.zeros_like(patch["x"]),
            colorscale=[[0, RED], [1, RED]],
            showscale=False,
            opacity=0.92,
        )
    )
apply_kg_layout(fig).show()
"""
            ),
            md(
                """
The surface is the finite-thickness geometric object.  The red balls are real
branch neighborhoods: each is incident to three blue tube pieces, so the spine
has genuine trivalent graph vertices.
"""
            ),
            code(
                r"""
fig = plot_graph_kg(graph)
fig.show()
"""
            ),
            code(
                r"""
projection = select_projection(graph, rotation_angles=(0.0, 0.0, 0.0))
print(f"rotation_angles = {tuple(round(a, 2) for a in projection.rotation_angles)}")
print(f"crossings = {projection.num_crossings}")
print(f"pd_code = {projection.pd_code}")
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.2, 4.4))
for arc in projection.arcs:
    ax.plot(*arc.line.xy, color=BLUE, linewidth=2.4)
for vertex in projection.vertices:
    ax.scatter(*vertex.point.xy, color=RED, s=64, zorder=3)
for crossing in projection.crossings:
    ax.scatter(*crossing.point.xy, marker="x", color="black", s=58, zorder=4)
ax.set_aspect("equal")
ax.axis("off")
plt.show()
"""
            ),
            code(
                r"""
Y = sp.Symbol("Y")
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=projection.rotation_angles,
    return_result=True,
    n_jobs=1,
)
print_upsilon("G", result.polynomial)
print(f"selected crossings = {result.projection.num_crossings}")
"""
            ),
            md(
                r"""
The output is the first complete library demonstration.  Geometry is no longer
needed after the PD code is built; the invariant calculation uses the symbolic
diagrammatic data.
"""
            ),
            md(
                """
## 2. Input Adapter Gallery: Data To KnottedGraph Objects

This section shows each public adapter as a real conversion step.  The notebook
creates small input files where needed, converts them to package objects, and
then plots the resulting graph or surface.
"""
            ),
            md(
                """
# To be filled by Zhaoyun

The executable examples below are temporary working examples.  Zhaoyun should
replace this section with the final public input-adapter tutorial and decide
which input formats need the most detailed presentation.
"""
            ),
            code(
                r"""
import csv
import json

import pyvista as pv

from knotted_graph.inputs import (
    from_coordinate_chain,
    from_lammps_dump,
    from_mmcif_backbone,
    from_nucleic_acid_backbone,
    from_protein_ca_backbone,
    from_spatial_graph_csv,
    from_surface_mesh,
    write_lammps_dump,
)

input_workdir = Path(tempfile.mkdtemp(prefix="knottedgraph-inputs-"))
print(f"input_workdir = {input_workdir}")


def pdb_atom(serial, atom_name, residue_name, chain_id, resseq, x, y, z):
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id:1s}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        "  1.00 20.00           C\n"
    )


def edge_pts(graph):
    return next(iter(graph.edges(data=True)))[2]["pts"]


t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
trefoil_coords = np.column_stack(
    [
        (2.0 + np.cos(3 * t)) * np.cos(2 * t),
        (2.0 + np.cos(3 * t)) * np.sin(2 * t),
        np.sin(3 * t),
    ]
)
coordinate_result = from_coordinate_chain(
    trefoil_coords,
    input_id="closed_trefoil_coordinate_chain",
    closed=True,
    closure="direct",
)

polymer_coords = np.column_stack(
    [
        np.cos(np.linspace(0, 5 * np.pi, 90)),
        np.sin(np.linspace(0, 5 * np.pi, 90)),
        np.linspace(-1.6, 1.6, 90),
    ]
)
dump_path = input_workdir / "open_helix_polymer.dump"
write_lammps_dump(polymer_coords, dump_path, molecule_id=7)
polymer_result = from_lammps_dump(dump_path, molecule_id=7, polymer_id="open_helix_polymer")

nodes_path = input_workdir / "k4_nodes.csv"
edges_path = input_workdir / "k4_edges.csv"
with nodes_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["node_id", "x", "y", "z", "label"])
    for node, data in graph.nodes(data=True):
        writer.writerow([node, *np.round(data["pos"], 6), f"K4 vertex {node}"])
with edges_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["edge_id", "source", "target", "points_json", "label"])
    for u, v, key, data in graph.edges(keys=True, data=True):
        writer.writerow([key, u, v, json.dumps(np.round(data["pts"], 6).tolist()), f"K4 edge {key}"])
spatial_csv_result = from_spatial_graph_csv(
    nodes_path,
    edges_path,
    graph_id="trivalent_k4_from_csv",
    metadata={"source": "generated inside the notebook"},
)

pdb_path = PROJECT_ROOT / "pdb-cache" / "1AOC.pdb"
protein_result = from_protein_ca_backbone(pdb_path, pdb_id="1AOC", chain_id="A", download=False)

dna_path = input_workdir / "mini_dna.pdb"
dna_path.write_text(
    "".join(
        [
            pdb_atom(1, "P", "DA", "A", 1, 0.0, 0.0, 0.0),
            pdb_atom(2, "P", "DT", "A", 2, 0.8, 0.5, 0.2),
            pdb_atom(3, "P", "DG", "A", 3, 1.4, 1.1, 0.7),
            pdb_atom(4, "P", "DC", "A", 4, 2.2, 1.4, 1.2),
        ]
    )
)
dna_result = from_nucleic_acid_backbone(dna_path, pdb_id="DNA1", chain_id="A", download=False)

cif_path = input_workdir / "mini.cif"
cif_path.write_text(
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
            "ATOM P P A A 1 . 0.8 0.4 0.2 C C 2 2",
            "ATOM P P A A 1 . 1.3 1.0 0.7 G G 3 3",
            "#",
        ]
    )
    + "\n"
)
mmcif_result = from_mmcif_backbone(cif_path, pdb_id="CIF1", chain_id="A", atom_name="P")

surface_mesh_path = input_workdir / "torus_surface.ply"
pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.28).triangulate().save(surface_mesh_path)
surface_mesh_result = from_surface_mesh(surface_mesh_path, mesh_id="torus_surface")

adapter_results = [
    ("coordinate_chain", coordinate_result.graph, coordinate_result.issues),
    ("polymer_lammps", polymer_result.graph, polymer_result.issues),
    ("spatial_graph_csv", spatial_csv_result.graph, spatial_csv_result.issues),
    ("protein_ca_pdb", protein_result.graph, protein_result.issues),
    ("nucleic_acid_pdb", dna_result.graph, dna_result.issues),
    ("mmcif_backbone", mmcif_result.graph, mmcif_result.issues),
]
for name, adapter_graph, issues in adapter_results:
    print(name)
    print("  nodes_edges =", (adapter_graph.number_of_nodes(), adapter_graph.number_of_edges()))
    print("  input_kind =", adapter_graph.graph.get("input_kind"))
    print("  issues =", issues)
print("surface_mesh")
print("  points_cells =", (surface_mesh_result.mesh.n_points, surface_mesh_result.mesh.n_cells))
print("  issues =", surface_mesh_result.issues)
"""
            ),
            code(
                r"""
fig = plot_graph_kg(coordinate_result.graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(spatial_csv_result.graph)
fig.show()
"""
            ),
            code(
                r"""
csv_projection = select_projection(spatial_csv_result.graph, num_rotation_samples=12)
csv_result = compute_yamada_polynomial(
    spatial_csv_result.graph,
    Y,
    rotation_angles=csv_projection.rotation_angles,
    return_result=True,
    n_jobs=1,
)
print(f"selected_crossings = {csv_projection.num_crossings}")
print(f"pd_code = {csv_projection.pd_code}")
print_upsilon("G_csv", csv_result.polynomial)
"""
            ),
            code(
                r"""
fig = plot_graph_kg(protein_result.graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(polymer_result.graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_surface_polydata(surface_mesh_result.mesh, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(dna_result.graph)
fig.show()
"""
            ),
            md(
                """
## 3. Inspection Mode: Every Intermediate Object

This section follows the paper's Fig. 1 logic with an awesome non-Hermitian
surface.  The inspected stages are surface, filled mask, skeleton, raw graph,
leaf removal, chain collapse, short-edge contraction, smoothing, planar
diagram, PD code, and invariant.
"""
            ),
            code(
                r"""
from skimage import morphology

from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import awesome_bloch_vector, hopf_link_bloch_vector
from knotted_graph.core import (
    contract_short_edges,
    idx_to_coord,
    remove_leaf_nodes,
    simplify_edges,
    smooth_edges,
)
from knotted_graph.extraction import skeleton_image_to_graph


def graph_indices_to_kspace(index_graph, skeleton):
    graph_kspace = nx.MultiGraph()
    for node, data in index_graph.nodes(data=True):
        attrs = dict(data)
        attrs["pos"] = idx_to_coord(attrs["pos"], spacing=skeleton.spacing, origin=skeleton.origin)
        graph_kspace.add_node(node, **attrs)

    for u, v, key, data in index_graph.edges(keys=True, data=True):
        attrs = dict(data)
        if attrs.get("pts") is not None:
            attrs["pts"] = idx_to_coord(attrs["pts"], spacing=skeleton.spacing, origin=skeleton.origin)
        graph_kspace.add_edge(u, v, key=key, **attrs)

    graph_kspace.graph.update(index_graph.graph)
    return graph_kspace

kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
inspection_gamma = 0.48
inspection_dimension = 72
inspection_model = awesome_bloch_vector(inspection_gamma, k_symbols=(kx, ky, kz))
display_bloch_vector(r"\vec d_{\mathrm{awesome}}(\mathbf{k})", inspection_model)
ske = NodalSkeleton(
    inspection_model,
    k_symbols=(kx, ky, kz),
    span=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    dimension=inspection_dimension,
    axis_scale=(1.0, 1.0, 1.0),
)

surface = ske.exceptional_surface_pv.connectivity("largest")
filled_mask = ske._interior_mask
skeleton_image = morphology.skeletonize(filled_mask, method="lee")

raw_graph_index = skeleton_image_to_graph(skeleton_image)
leaf_removed_graph_index = remove_leaf_nodes(raw_graph_index)
chain_collapsed_graph_index = simplify_edges(leaf_removed_graph_index)

raw_graph = graph_indices_to_kspace(raw_graph_index, ske)
leaf_removed_graph = graph_indices_to_kspace(leaf_removed_graph_index, ske)
chain_collapsed_graph = graph_indices_to_kspace(chain_collapsed_graph_index, ske)
contracted_graph = contract_short_edges(chain_collapsed_graph, min_length=0.40)
processed_graph = smooth_edges(contracted_graph, epsilon=0.08, copy=True)

rng = np.random.default_rng(5)
mask_indices = np.column_stack(np.where(filled_mask))
mask_indices = mask_indices[
    rng.choice(len(mask_indices), size=min(4500, len(mask_indices)), replace=False)
]
filled_mask_points = idx_to_coord(mask_indices, spacing=ske.spacing, origin=ske.origin)
skeleton_points = idx_to_coord(
    np.column_stack(np.where(skeleton_image)),
    spacing=ske.spacing,
    origin=ske.origin,
)

candidate_projections = sorted(
    sample_projections(processed_graph, num_rotation_samples=16),
    key=lambda candidate: (
        "X[" not in candidate.pd_code,
        candidate.num_crossings == 0,
        candidate.num_crossings,
        candidate.rotation_angles,
    ),
)
projection_inspection = candidate_projections[0]
inspection_result = compute_yamada_polynomial(
    processed_graph,
    Y,
    rotation_angles=projection_inspection.rotation_angles,
    return_result=True,
    n_jobs=1,
)

stage_graphs = [
    ("raw graph", raw_graph),
    ("leaf removal", leaf_removed_graph),
    ("chain collapse", chain_collapsed_graph),
    ("edge contraction", contracted_graph),
    ("smoothing", processed_graph),
]

print("input = awesome_bloch_vector(gamma=0.48)")
print("dimension =", inspection_dimension)
print("surface_points_cells =", (surface.n_points, surface.n_cells))
print("filled_mask_voxels =", int(filled_mask.sum()))
print("skeleton_voxels =", int(skeleton_image.sum()))
for name, stage_graph in stage_graphs:
    print(
        f"{name:16s}",
        "nodes_edges =",
        (stage_graph.number_of_nodes(), stage_graph.number_of_edges()),
        "degrees =",
        sorted(dict(stage_graph.degree()).values()),
    )
print("projection_crossings =", projection_inspection.num_crossings)
print_upsilon("G_awesome_inspection", inspection_result.polynomial)
"""
            ),
            code(
                r"""
fig = plot_surface_polydata(surface, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_points_3d(filled_mask_points, size=2)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_points_3d(skeleton_points, size=3)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(raw_graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(leaf_removed_graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(chain_collapsed_graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(contracted_graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(processed_graph)
fig.show()
"""
            ),
            code(
                r"""
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for arc in projection_inspection.arcs:
    ax.plot(*arc.line.xy, color=BLUE, linewidth=2.4)
for vertex in projection_inspection.vertices:
    ax.scatter(*vertex.point.xy, color=RED, s=64, zorder=3)
for crossing in projection_inspection.crossings:
    ax.scatter(*crossing.point.xy, marker="x", color="black", s=58, zorder=4)
ax.set_aspect("equal")
ax.axis("off")
plt.show()
"""
            ),
            code(
                r"""
print(f"rotation_angles = {tuple(round(a, 2) for a in projection_inspection.rotation_angles)}")
print(f"crossings = {projection_inspection.num_crossings}")
print(f"pd_code = {projection_inspection.pd_code}")
print("vertices =", [(v.id, v.key) for v in projection_inspection.vertices])
print(
    "crossing_points =",
    [(x.id, tuple(round(c, 3) for c in tuple(x.point.coords)[0])) for x in projection_inspection.crossings],
)
print(
    "arcs =",
    [(arc.id, arc.start_type, arc.start_id, arc.end_type, arc.end_id) for arc in projection_inspection.arcs],
)
"""
            ),
            md(
                """
The inspection output is deliberately verbose.  It gives the user enough
information to debug the exact diagram used downstream: the selected rotation,
crossings, vertices, arcs, and PD-code string.  The graph sequence also shows
which cleanup operation changes the skeleton at each stage.
"""
            ),
            md(
                """
### 3.1 The Same Pipeline Through The High-Level Convenience Method

The previous cells expose the intermediate objects one by one.  When a user
only wants the standard simplified graph, the high-level method wraps the same
core operations: skeleton image to graph, leaf removal, chain collapse, and
edge smoothing.
"""
            ),
            code(
                r"""
standard_graph = ske.skeleton_graph(simplify=True, smooth_epsilon=2)
standard_graph_kspace = graph_indices_to_kspace(standard_graph, ske)

print("standard high-level graph =", (standard_graph_kspace.number_of_nodes(), standard_graph_kspace.number_of_edges()))
print("manual processed graph =", (processed_graph.number_of_nodes(), processed_graph.number_of_edges()))
"""
            ),
            code(
                r"""
fig = plot_graph_kg(standard_graph_kspace)
fig.show()
"""
            ),
            md(
                """
Publication-grid exports with panel letters and compressed PDF/SVG/PNG output
are deliberately separate from this tutorial.  The reusable library objects are
the generated surface, mask, skeleton, graphs, projection, PD code, and
polynomial displayed above.
"""
            ),
            md(
                """
## 4. Physical Fields From The Same Hamiltonian

A non-Hermitian Hamiltonian produces field data, not only a skeleton graph.  In
`KnottedGraph`, this information is available through the same
`NodalSkeleton.fields_pv` object.  The examples below generate the
Berry-curvature and dispersion views directly from the public application
module.
"""
            ),
            code(
                r"""
field_model = hopf_link_bloch_vector(0.4, k_symbols=(kx, ky, kz))
display_bloch_vector(r"\vec d_{\mathrm{Hopf}}(\mathbf{k})", field_model)
field_ske = NodalSkeleton(
    field_model,
    k_symbols=(kx, ky, kz),
    dimension=36,
    axis_scale=(1.0, 1.0, 1.5),
)
field_surface = field_ske.exceptional_surface_pv.connectivity("largest")
field_volume = field_ske.fields_pv

print("available point-data arrays:")
for name in field_volume.point_data:
    arr = np.asarray(field_volume.point_data[name])
    print(f"  {name:20s} shape={arr.shape} min={np.nanmin(arr):.3g} max={np.nanmax(arr):.3g}")
"""
            ),
            code(
                r"""
points = field_volume.points
berry_vec = np.asarray(field_volume.point_data["berry"])
berry_strength = np.linalg.norm(berry_vec, axis=1)
finite = np.isfinite(berry_strength) & (berry_strength > 0)
candidate = np.flatnonzero(finite)
if len(candidate) > 220:
    ranked = candidate[np.argsort(berry_strength[candidate])[-220:]]
else:
    ranked = candidate

mesh = field_surface.triangulate()
faces = mesh.faces.reshape(-1, 4)[:, 1:]
pts = mesh.points

fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=BLUE,
        opacity=0.22,
        showscale=False,
    )
)
fig.add_trace(
    go.Cone(
        x=points[ranked, 0],
        y=points[ranked, 1],
        z=points[ranked, 2],
        u=berry_vec[ranked, 0],
        v=berry_vec[ranked, 1],
        w=berry_vec[ranked, 2],
        colorscale="Blues",
        showscale=False,
        sizemode="absolute",
        sizeref=0.055,
        opacity=0.78,
    )
)
apply_kg_layout(fig).show()
"""
            ),
            code(
                r"""
dispersion_scalar = np.asarray(field_volume.point_data["log10(|im_disp|+1)"])
grid_shape = field_ske.kx_grid.shape
dispersion_grid = dispersion_scalar.reshape(grid_shape, order="F")
middle = grid_shape[2] // 2

fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.contourf(
    field_ske.kx_grid[:, :, middle],
    field_ske.ky_grid[:, :, middle],
    dispersion_grid[:, :, middle],
    levels=28,
    cmap="Blues",
)
ax.set_aspect("equal")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
plt.show()
"""
            ),
            md(
                """
The surface-mode helper is available under
`knotted_graph.applications.nodal.surface_modes`.  The grid below is deliberately
small so `Run All` stays practical: Hamiltonian to open-boundary batches,
eigenvalues, then a surface-mode plot.
"""
            ),
            code(
                r"""
from poly2graph import eig_batch

from knotted_graph.applications.nodal.surface_modes import H_batch
from knotted_graph.applications.nodal.visualization import plot_surface_modes

sigma_x = sp.ImmutableMatrix([[0, 1], [1, 0]])
sigma_z = sp.ImmutableMatrix([[1, 0], [0, -1]])


def h_hopf_hermitian(kx_value, ky_value, kz_value):
    z = sp.cos(2 * kz_value) + sp.Rational(1, 2) + sp.I * (
        sp.cos(kx_value) + sp.cos(ky_value) + sp.cos(kz_value) - 2
    )
    w = sp.sin(kx_value) + sp.I * sp.sin(ky_value)
    f_value = z**2 - w**2
    return sp.simplify(sp.re(f_value) * sigma_x + sp.im(f_value) * sigma_z)


k_grid_small = np.linspace(-np.pi, np.pi, 8)
chain_length = 4
h_obc_x, _ = H_batch(h_hopf_hermitian, "x", chain_length, k_grid_small, k_grid_small)
h_obc_y, _ = H_batch(h_hopf_hermitian, "y", chain_length, k_grid_small, k_grid_small)
h_obc_z, _ = H_batch(h_hopf_hermitian, "z", chain_length, k_grid_small, k_grid_small)
eig_x, _ = eig_batch(h_obc_x)
eig_y, _ = eig_batch(h_obc_y)
eig_z, _ = eig_batch(h_obc_z)

fig = plot_surface_modes(
    (eig_x, eig_y, eig_z),
    (k_grid_small, k_grid_small, k_grid_small),
    (0.04, 0.07, 0.07),
    nH_coeff=0.0,
)
for ax in fig.axes:
    ax.set_title("")
plt.show()
"""
            ),
            md(
                """
## 5. Projection, PD Codes, And Rigid-Vertex Data

The same spatial graph can be sampled from many viewing directions.  The
selected projection is the one used for PD-code construction and invariant
evaluation, but users may inspect other projections when a particular diagram
is desired.
"""
            ),
            code(
                r"""
projections = sample_projections(graph, num_rotation_samples=12)
summary = sorted(
    [
        (idx, p.num_crossings, tuple(round(a, 2) for a in p.rotation_angles), p.pd_code)
        for idx, p in enumerate(projections)
    ],
    key=lambda row: (row[1], row[0]),
)
for idx, crossings, angles, pd_code in summary[:5]:
    print(f"projection[{idx}] crossings={crossings} angles={angles}")
    print(pd_code)
"""
            ),
            md(
                """
For a rigid-vertex invariant, the local cyclic ordering at graph vertices
matters.  The PD code is therefore not merely a drawing label; it is the
combinatorial object sent to the Yamada engine.
"""
            ),
            md(
                """
## 6. Mathematical Workflows: Yamada Engines And Catalogs

The library exposes complementary routes for crossing-free state evaluation:
the recursive deletion-contraction route and the Negami-polynomial state-sum
route.  For small examples, the notebook compares them directly.  After that,
the same section treats Yamada computation as a mathematical graph-family
workflow: build a named graph, plot it with loops and edge multiplicities
visible, and record both `Upsilon(G;Y)` and its sigma form.
"""
            ),
            code(
                r"""
negami = compute_yamada_polynomial(
    processed_graph,
    Y,
    rotation_angles=projection_inspection.rotation_angles,
    method="negami",
    n_jobs=1,
)
recursive = compute_yamada_polynomial(
    processed_graph,
    Y,
    rotation_angles=projection_inspection.rotation_angles,
    method="recursive",
    n_jobs=1,
)
print_upsilon("G_awesome_inspection, negami", negami)
print_upsilon("G_awesome_inspection, recursive", recursive)
print("same_result =", sp.expand(negami - recursive) == 0)
"""
            ),
            code(
                r"""
from knotted_graph.core import ThetaGraph
from knotted_graph.invariants.yamada import compute_yamada_polynomial_recursive

for s in range(2, 8):
    theta = ThetaGraph(s)
    upsilon = compute_yamada_polynomial_recursive(theta, Y)
    print_upsilon(f"Theta_{s}", upsilon)
"""
            ),
            md(
                """
### Structured Graph Yamada Catalog

The mathematical catalog is a reusable part of the package.  It is useful when
the research question is about graph families themselves rather than a graph
extracted from a physical surface.  Each family has a builder, a layout, sample
parameters, and a short note.
"""
            ),
            code(
                r"""
from knotted_graph.applications.mathematical import (
    GRAPH_FAMILY_CATALOG,
    NOTEBOOK_YAMADA_EXAMPLES,
    build_graph_case,
    graph_summary,
    plot_structured_multigraph,
)
from knotted_graph.invariants.yamada import laurent_y_to_sigma_polynomial

sigma = sp.Symbol("sigma")

for family_name, spec in GRAPH_FAMILY_CATALOG.items():
    print(f"{family_name:24s} sample_args={spec.sample_args}  {spec.note}")
"""
            ),
            md(
                """
The next cell generates the displayed catalog figure directly from the package
builders.  Parallel edges are curved apart, loops are drawn as visible loops,
and node labels are numeric aliases so the figure remains readable.
"""
            ),
            code(
                r"""
import matplotlib.pyplot as plt

catalog_results = {}
cols = 3
rows = math.ceil(len(NOTEBOOK_YAMADA_EXAMPLES) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = np.asarray(axes).reshape(-1)

for ax, (family_name, args, label) in zip(axes, NOTEBOOK_YAMADA_EXAMPLES):
    graph, pos = build_graph_case(family_name, *args)
    plot_structured_multigraph(
        graph,
        pos,
        family_name=family_name,
        family_args=args,
        ax=ax,
        show=False,
    )
    yamada_y = sp.expand(compute_yamada_polynomial_recursive(graph, Y))
    yamada_sigma = laurent_y_to_sigma_polynomial(yamada_y, Y, sigma).as_expr()
    catalog_results[label] = {
        "summary": graph_summary(graph),
        "Y": yamada_y,
        "sigma": sp.expand(yamada_sigma),
    }

for ax in axes[len(NOTEBOOK_YAMADA_EXAMPLES):]:
    ax.set_axis_off()

plt.tight_layout()
plt.show()

for panel_index, label in enumerate(catalog_results, start=1):
    result = catalog_results[label]
    print(f"panel {panel_index}: {label}  {result['summary']}")
    print_upsilon(label, result["Y"])
    print(f"Upsilon_sigma({label}; sigma) = {result['sigma']}")
    print()
"""
            ),
            md(
                """
For a single-family study, users can call the same builder with different
parameters.  This cell treats the cylinder family as a compact example of
scanning a parameterized graph family.
"""
            ),
            code(
                r"""
cylinder_scan = []
for cols in range(3, 7):
    graph, pos = build_graph_case("cylinder", 2, cols)
    yamada_y = sp.expand(compute_yamada_polynomial_recursive(graph, Y))
    yamada_sigma = laurent_y_to_sigma_polynomial(yamada_y, Y, sigma).as_expr()
    cylinder_scan.append((cols, graph_summary(graph), yamada_y, sp.expand(yamada_sigma)))

graph, pos = build_graph_case("cylinder", 2, 6)
plot_structured_multigraph(
    graph,
    pos,
    family_name="cylinder",
    family_args=(2, 6),
)

for cols, summary, yamada_y, yamada_sigma in cylinder_scan:
    label = f"Cylinder(2,{cols})"
    print(f"{label}: {summary}")
    print_upsilon(label, yamada_y)
    print(f"Upsilon_sigma({label}; sigma) = {yamada_sigma}")
    print()
"""
            ),
            code(
                r"""
import csv

dataset_path = DOC_ROOT / "assets/data/structured_graph_yamada_dataset.csv"
with dataset_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

print(f"stored structured rows = {len(rows)}")
for row in rows[:12]:
    print(f"Upsilon({row['graph_name']}{row['varying_params']}; Y) = {row['yamada']}")
"""
            ),
            md(
                """
The stored table remains useful as a reproducibility artifact, but the cells
above show the executable workflow: choose the family, build the graph, inspect
the figure, and compute the invariant in the notation of the paper.
"""
            ),
            md(
                """
## 7. Repulsive-Curve Workflow For Complicated Embeddings

The public package can build protein-derived theta graphs and evaluate small
examples directly.  The full repulsive optimization requires the external
Repulsor driver, so the optimization call is shown as reference code below,
while the initial protein graph and invariant are executable.
"""
            ),
            md(
                """
# To be filled by Kehan

This section is marked for Kehan because it covers protein-derived theta graphs
and repulsive-curve layouts.  The executable examples below are placeholders
showing the current public calls and expected outputs.
"""
            ),
            code(
                r"""
from pathlib import Path

from knotted_graph.layout.repulsive import (
    available_samples,
    build_protein_example,
    curve_network_to_multigraph,
)
from knotted_graph.layout.repulsive.protein_examples import set_special_node_distance

print("available protein examples =", available_samples())

protein_networks = {}
protein_graphs = {}
for sample in available_samples():
    network = build_protein_example(
        sample,
        pdb_cache=PROJECT_ROOT / "pdb-cache",
        total_arc_points=42 if sample == "1aoc" else 54,
    )
    if sample == "1aoc":
        set_special_node_distance(network, target_distance=9.0)
    protein_networks[sample] = network
    protein_graphs[sample] = curve_network_to_multigraph(network)
    print(sample)
    print("  name =", network.name)
    print("  nodes =", network.node_order)
    print("  arcs =", network.arc_order)
    print("  nodes_edges =", (protein_graphs[sample].number_of_nodes(), protein_graphs[sample].number_of_edges()))

network = protein_networks["1aoc"]
protein_graph = protein_graphs["1aoc"]
"""
            ),
            code(
                r"""
fig = plot_graph_kg(protein_graph)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(protein_graphs["3ulk"])
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(protein_graphs["5osq"])
fig.show()
"""
            ),
            code(
                r"""
protein_projection = select_projection(protein_graph, num_rotation_samples=12)
protein_result = compute_yamada_polynomial(
    protein_graph,
    Y,
    rotation_angles=protein_projection.rotation_angles,
    return_result=True,
    n_jobs=1,
)
print(f"selected_crossings = {protein_projection.num_crossings}")
print(f"pd_code = {protein_projection.pd_code}")
print_upsilon("G_1AOC", protein_result.polynomial)
"""
            ),
            md(
                r"""
Reference code for the optional external-driver layout step:

```python
from knotted_graph.layout.repulsive import SolverOptions, relax_spatial_graph

layout = relax_spatial_graph(
    protein_graph,
    workspace="protein-layout",
    solver_options=SolverOptions(steps=100, max_time=20, threads=1),
    save_steps=True,
    keep_workspace=True,
    verify_topology=True,
)

relaxed_graph = layout.graph
```

This should become a fully executable paper cell once the external driver is
installed and the layout workspace is standardized for distribution.
"""
            ),
            md(
                """
The before/after panel records the expected visual purpose of the repulsive
stage: simplify the geometric representative while preserving the graph
incidence and, after successful topology verification, the invariant.

No static before/after image is displayed here.  Once the external Repulsor
driver is available in the environment, run the reference code above and then
plot `protein_graph` and `relaxed_graph` with `plot_graph_kg(...)` in adjacent
Plotly scenes.
"""
            ),
            md(
                r"""
## 8. Application Gallery: Inputs, Graphs, And Invariants

Every application example should answer the same questions:

1. Where does the input come from?
2. What geometric object is produced?
3. What spatial graph is extracted?
4. What PD code and `Upsilon(G;Y)` are obtained?

The nodal examples below call model constructors from
`knotted_graph.applications.nodal.models`.  Each constructor returns the Bloch
vector \(\vec d(\mathbf{k})\) used to form the two-band Hamiltonian
\(H(\mathbf{k})=\vec d(\mathbf{k})\cdot\vec\sigma\).
"""
            ),
            code(
                r"""
from knotted_graph.applications.nodal.models import (
    awesome_bloch_vector,
    solomon_bloch_vector,
    trefoil_bloch_vector,
)

application_models = [
    ("Trefoil nodal model", trefoil_bloch_vector, 0.3),
    ("Solomon nodal model", solomon_bloch_vector, 0.55),
    ("Awesome nodal graph model", awesome_bloch_vector, 0.16),
]

application_rows = []
for name, builder, gamma in application_models:
    bloch_vector = builder(gamma, k_symbols=(kx, ky, kz))
    display_bloch_vector(r"\vec d_{\mathrm{" + name.split()[0] + r"}}(\mathbf{k})", bloch_vector)
    ske_app = NodalSkeleton(
        bloch_vector,
        k_symbols=(kx, ky, kz),
        dimension=48,
        axis_scale=(1.0, 1.0, 1.5),
    )
    surface_app = ske_app.exceptional_surface_pv.connectivity("largest")
    graph_app = ske_app.skeleton_graph(simplify=True, smooth_epsilon=2)
    projection_app = select_projection(graph_app, num_rotation_samples=12)
    result_app = compute_yamada_polynomial(
        graph_app,
        Y,
        rotation_angles=projection_app.rotation_angles,
        return_result=True,
        n_jobs=1,
    )
    application_rows.append((name, gamma, surface_app, graph_app, projection_app, result_app.polynomial))

for name, gamma, surface_app, graph_app, projection_app, polynomial in application_rows:
    print(name)
    print("gamma =", gamma)
    print("surface_points_cells =", (surface_app.n_points, surface_app.n_cells))
    print("nodes_edges =", (graph_app.number_of_nodes(), graph_app.number_of_edges()))
    print("crossings =", projection_app.num_crossings)
    print_upsilon("G", polynomial)
"""
            ),
            code(
                r"""
name, gamma, surface_app, graph_app, projection_app, polynomial = application_rows[0]
fig = plot_surface_polydata(surface_app, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
name, gamma, surface_app, graph_app, projection_app, polynomial = application_rows[0]
fig = plot_graph_kg(graph_app)
fig.show()
"""
            ),
            code(
                r"""
name, gamma, surface_app, graph_app, projection_app, polynomial = application_rows[1]
fig = plot_surface_polydata(surface_app, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
name, gamma, surface_app, graph_app, projection_app, polynomial = application_rows[1]
fig = plot_graph_kg(graph_app)
fig.show()
"""
            ),
            code(
                r"""
name, gamma, surface_app, graph_app, projection_app, polynomial = application_rows[2]
fig = plot_surface_polydata(surface_app, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
name, gamma, surface_app, graph_app, projection_app, polynomial = application_rows[2]
fig = plot_graph_kg(graph_app)
fig.show()
"""
            ),
            md(
                r"""
### Material Fermi-Surface Examples

Material examples should follow the same reproducible path as the nodal
examples: display the Hamiltonian, generate the surface from that Hamiltonian,
plot the surface, extract the graph, plot the graph, compute PD code, and then
print `Upsilon(G;Y)`.

The Hamiltonian constructors used here live in
`knotted_graph.applications.materials`, so users can import the same symbolic
models in their own notebooks instead of copying formulas.

For every material example, the public notebook should eventually show:

$$
H(\mathbf{k})
\longrightarrow
\text{constant-energy surface}
\longrightarrow
G\subset\mathbb{R}^3
\longrightarrow
\operatorname{PD}(G)
\longrightarrow
\Upsilon(G;Y).
$$
"""
            ),
            code(
                r"""
from knotted_graph.applications.materials import (
    H_D6_sympy,
    H_Ti3Al_sympy,
    H_YH3_sympy,
)

H_ti3al = H_Ti3Al_sympy(k_symbols=(kx, ky, kz))
H_d6 = H_D6_sympy(k_symbols=(kx, ky, kz))
H_yh3 = H_YH3_sympy(k_symbols=(kx, ky, kz))

display(Math(r"H_{\mathrm{Ti_3Al}}(\mathbf{k})=" + sp.latex(H_ti3al)))
display(Math(r"H_{D_6}(\mathbf{k})=" + sp.latex(H_d6)))
display(Math(r"H_{\mathrm{YH}_3}(\mathbf{k})=" + sp.latex(H_yh3)))

print("material Hamiltonian constructors are public")
print("public material-surface adapter = pending")
print("next library task: provide a public material-surface adapter in knotted_graph.applications.materials")
"""
            ),
            md(
                r"""
Reference structure for the future public material API:

```python
from knotted_graph.applications.materials import MaterialFermiSurface

material = MaterialFermiSurface.from_hamiltonian(
    name="Ti3Al",
    hamiltonian=H_ti3al,
    k_symbols=(kx, ky, kz),
    energy_window=(-0.25, 0.25),
)

surface = material.surface()
plot_surface_polydata(surface).show()

graph = material.skeleton_graph(simplify=True)
plot_graph_kg(graph).show()

projection = select_projection(graph, num_rotation_samples=12)
result = compute_yamada_polynomial(
    graph,
    Y,
    rotation_angles=projection.rotation_angles,
    return_result=True,
    n_jobs=1,
)
print_upsilon("G_Ti3Al", result.polynomial)
```

This is intentionally not executed yet because the material helper is not a
public package interface.  Showing static `material_*.png` panels here would
hide that API gap rather than helping users reproduce the workflow.
"""
            ),
            md(
                """
## 9. Paper Figure Reproduction Map

This section regenerates representative manuscript-style panels from public
notebook code.  Each plotted output below is generated in the cell that
immediately precedes it.
"""
            ),
            code(
                r"""
from knotted_graph.applications.nodal.models import pq_torus_knot_bloch_vector, threelink_bloch_vector

torus_ske = NodalSkeleton(
    pq_torus_knot_bloch_vector(1, 2, 0.2, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=96,
    axis_scale=(1.0, 1.0, 1.5),
)
torus_surface = torus_ske.exceptional_surface_pv.connectivity("largest")
torus_points = torus_ske.skeleton_coords
torus_graph = torus_ske.skeleton_graph(simplify=True, smooth_epsilon=2)

print("torus surface =", (torus_surface.n_points, torus_surface.n_cells))
print("torus skeleton points =", torus_points.shape)
print("torus graph =", (torus_graph.number_of_nodes(), torus_graph.number_of_edges()))
"""
            ),
            code(
                r"""
fig = plot_surface_polydata(torus_surface, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_points_3d(torus_points, size=3)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(torus_graph)
fig.show()
"""
            ),
            code(
                r"""
planarity_ske = NodalSkeleton(
    threelink_bloch_vector(0.41, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=96,
    axis_scale=(1.0, 1.0, 1.5),
)
planarity_surface = planarity_ske.exceptional_surface_pv.connectivity("largest")
planarity_graph = planarity_ske.skeleton_graph(simplify=True, smooth_epsilon=2)
is_planar = nx.check_planarity(nx.Graph(planarity_graph))[0]

print("three-link surface =", (planarity_surface.n_points, planarity_surface.n_cells))
print("three-link graph =", (planarity_graph.number_of_nodes(), planarity_graph.number_of_edges()))
print("is_planar =", is_planar)
"""
            ),
            code(
                r"""
fig = plot_surface_polydata(planarity_surface, opacity=0.58)
fig.show()
"""
            ),
            code(
                r"""
fig = plot_graph_kg(planarity_graph)
fig.show()
"""
            ),
            md(
                """
## 10. Appendix-Style Figure Workflows

The cells below build manuscript-style scientific panels directly from
`KnottedGraph` objects: surfaces, skeleton points, spatial graphs, projections,
Yamada fingerprints, Berry slices, minor checks, and planarity changes.
"""
            ),
            code(
                r"""
from plotly.subplots import make_subplots
from knotted_graph.applications.nodal.models import (
    awesome_bloch_vector,
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    threelink_bloch_vector,
    trefoil_bloch_vector,
    unknot_bloch_vector,
)

def add_surface_trace(fig, surface, *, row=1, col=1, opacity=0.58, color=BLUE):
    mesh = surface.triangulate()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    pts = mesh.points
    fig.add_trace(
        go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            showscale=False,
        ),
        row=row,
        col=col,
    )


def add_points_trace(fig, points, *, row=1, col=1, size=2.5, color=BLUE):
    points = np.asarray(points)
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=size, color=color),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def add_graph_traces(fig, graph, *, row=1, col=1):
    graph_fig = plot_3D_graph_plotly(graph)
    for trace in graph_fig.data:
        fig.add_trace(trace, row=row, col=col)


def style_plotly_scenes(fig, scene_count, *, width=980, height=660):
    for index in range(1, scene_count + 1):
        scene_name = "scene" if index == 1 else f"scene{index}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=axis_style(),
                    yaxis=axis_style(),
                    zaxis=axis_style(),
                    aspectmode="data",
                    camera=CAMERA,
                )
            }
        )
    fig.update_layout(
        title=None,
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


def polyline_traces_from_slice(polydata):
    points = np.asarray(polydata.points)
    lines = np.asarray(polydata.lines)
    traces = []
    cursor = 0
    while cursor < len(lines):
        n = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n]
        cursor += n + 1
        if n < 2:
            continue
        pts = points[ids]
        traces.append(pts)
    return traces

print("appendix plotting helpers ready")
"""
            ),
            md(
                """
### 10.1 Skeletonization Appendix: Torus `(2,4)`

The original appendix showed two thicknesses for a `(2,4)` torus model.  In the
current public extraction path, `gamma=0.06` still produces a surface and
skeleton points but the simplified graph collapses; `gamma=0.2` produces the
full graph stage.  Showing both is more useful than hiding the fragile case.
"""
            ),
            code(
                r"""
appendix_torus_records = []
for gamma in (0.06, 0.2):
    ske_t24 = NodalSkeleton(
        pq_torus_knot_bloch_vector(2, 4, gamma, k_symbols=(kx, ky, kz), c=0.7, m=2.0),
        k_symbols=(kx, ky, kz),
        dimension=120,
        axis_scale=(1.0, 1.0, 1.5),
    )
    surface_t24 = ske_t24.exceptional_surface_pv.connectivity("largest")
    points_t24 = ske_t24.skeleton_coords
    graph_t24 = None
    graph_error = None
    try:
        graph_t24 = ske_t24.skeleton_graph(simplify=True, smooth_epsilon=2)
    except Exception as exc:
        graph_error = f"{type(exc).__name__}: {exc}"
    appendix_torus_records.append((gamma, surface_t24, points_t24, graph_t24, graph_error))
    print(f"gamma = {gamma}")
    print("  surface_points_cells =", (surface_t24.n_points, surface_t24.n_cells))
    print("  skeleton_points =", points_t24.shape)
    if graph_t24 is None:
        print("  graph_stage =", graph_error)
    else:
        print("  graph_nodes_edges =", (graph_t24.number_of_nodes(), graph_t24.number_of_edges()))
"""
            ),
            code(
                r"""
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    horizontal_spacing=0.02,
)
for col, (gamma, surface_t24, points_t24, graph_t24, graph_error) in enumerate(appendix_torus_records, start=1):
    add_surface_trace(fig, surface_t24, row=1, col=col, opacity=0.42)
    add_points_trace(fig, points_t24, row=1, col=col, size=2.3, color=RED if graph_t24 is None else BLUE)
style_plotly_scenes(fig, 2, width=980, height=520).show()
"""
            ),
            code(
                r"""
gamma, surface_t24, points_t24, graph_t24, graph_error = appendix_torus_records[1]
fig = plot_graph_kg(graph_t24)
fig.show()
"""
            ),
            md(
                """
### 10.2 Yamada Appendix Table

The appendix table compares knot/graph families across parameter choices.  The
cell below keeps all rows from the appendix schedule.  When a row does not
produce a valid graph/projection at this practical notebook resolution, the
failure is printed explicitly; that tells the user to inspect thickness,
resolution, simplification, and projection choice before trusting a polynomial.
"""
            ),
            code(
                r"""
appendix_yamada_specs = [
    ("Hopf link", "Y^2 + 1", hopf_link_bloch_vector, [0.1, 0.2, 0.5], 64),
    ("Trefoil", "Y - 1 + 1/Y", trefoil_bloch_vector, [0.1, 0.19, 0.25], 64),
    ("Torus (1,2)", "1", lambda gamma, k_symbols: pq_torus_knot_bloch_vector(1, 2, gamma, k_symbols=k_symbols), [0.12, 0.5, 0.7], 64),
    ("Solomon", "Y^2 - Y + 1 - 1/Y - 1/Y^2", solomon_bloch_vector, [0.12, 1.0, 2.0], 64),
]

appendix_yamada_rows = []
for family, classical_reference, builder, gammas, dimension in appendix_yamada_specs:
    for gamma in gammas:
        row = {
            "family": family,
            "gamma": gamma,
            "classical_reference": classical_reference,
            "dimension": dimension,
            "status": "ok",
        }
        try:
            ske_row = NodalSkeleton(
                builder(gamma, k_symbols=(kx, ky, kz)),
                k_symbols=(kx, ky, kz),
                dimension=dimension,
                axis_scale=(1.0, 1.0, 1.5),
            )
            graph_row = ske_row.skeleton_graph(simplify=True, smooth_epsilon=2)
            projection_row = select_projection(graph_row, num_rotation_samples=8)
            polynomial_row = compute_yamada_polynomial(
                graph_row,
                Y,
                rotation_angles=projection_row.rotation_angles,
                n_jobs=1,
            )
            row.update(
                nodes=graph_row.number_of_nodes(),
                edges=graph_row.number_of_edges(),
                crossings=projection_row.num_crossings,
                upsilon=str(sp.expand(polynomial_row)),
            )
        except Exception as exc:
            row.update(status="inspect", reason=f"{type(exc).__name__}: {str(exc)[:120]}")
        appendix_yamada_rows.append(row)

for row in appendix_yamada_rows:
    print(f"{row['family']:12s} gamma={row['gamma']:<4} status={row['status']}")
    print(f"  classical_reference = {row['classical_reference']}")
    if row["status"] == "ok":
        print(f"  nodes_edges_crossings = {(row['nodes'], row['edges'], row['crossings'])}")
        print(f"  Upsilon(G; Y) = {row['upsilon']}")
    else:
        print(f"  needs inspection = {row['reason']}")
"""
            ),
            md(
                """
### 10.3 Energy-Isosurface Appendix Gallery

The appendix gallery compared several Hamiltonian families.  This notebook
rebuilds the surfaces and extracted spatial graphs from the model functions,
then prints which panels are planar or non-planar after simplification.
"""
            ),
            code(
                r"""
energy_gallery_specs = [
    ("Unknot", lambda: unknot_bloch_vector(0.1, k_symbols=(kx, ky, kz)), 64),
    ("Hopf link", lambda: hopf_link_bloch_vector(0.2, k_symbols=(kx, ky, kz)), 48),
    ("Trefoil", lambda: trefoil_bloch_vector(0.25, k_symbols=(kx, ky, kz)), 64),
    ("Solomon", lambda: solomon_bloch_vector(1.0, k_symbols=(kx, ky, kz)), 64),
    ("Three-link", lambda: threelink_bloch_vector(0.41, k_symbols=(kx, ky, kz)), 80),
    ("Torus (1,2)", lambda: pq_torus_knot_bloch_vector(1, 2, 0.5, k_symbols=(kx, ky, kz)), 48),
    ("Torus (3,7)", lambda: pq_torus_knot_bloch_vector(3, 7, 0.2, k_symbols=(kx, ky, kz), c=0.7, m=2.0), 96),
    ("Awesome", lambda: awesome_bloch_vector(0.16, k_symbols=(kx, ky, kz)), 64),
]

energy_gallery_records = []
for name, builder, dimension in energy_gallery_specs:
    ske_energy = NodalSkeleton(
        builder(),
        k_symbols=(kx, ky, kz),
        dimension=dimension,
        axis_scale=(1.0, 1.0, 1.5),
    )
    surface_energy = ske_energy.exceptional_surface_pv.connectivity("largest")
    graph_energy = None
    graph_error = None
    is_planar = None
    try:
        graph_energy = ske_energy.skeleton_graph(simplify=True, smooth_epsilon=2)
        is_planar = nx.check_planarity(nx.Graph(graph_energy))[0]
    except Exception as exc:
        graph_error = f"{type(exc).__name__}: {exc}"
    energy_gallery_records.append((name, surface_energy, graph_energy, graph_error, is_planar))
    print(name)
    print("  surface_points_cells =", (surface_energy.n_points, surface_energy.n_cells))
    if graph_energy is None:
        print("  graph_stage =", graph_error)
    else:
        print("  graph_nodes_edges =", (graph_energy.number_of_nodes(), graph_energy.number_of_edges()))
        print("  planar =", is_planar)
"""
            ),
            code(
                r"""
fig = make_subplots(
    rows=2,
    cols=4,
    specs=[[{"type": "scene"} for _ in range(4)], [{"type": "scene"} for _ in range(4)]],
    horizontal_spacing=0.01,
    vertical_spacing=0.02,
)
for index, (name, surface_energy, graph_energy, graph_error, is_planar) in enumerate(energy_gallery_records):
    row = 1 if index < 4 else 2
    col = index % 4 + 1
    add_surface_trace(fig, surface_energy, row=row, col=col, opacity=0.46)
    if graph_energy is not None:
        add_graph_traces(fig, graph_energy, row=row, col=col)
style_plotly_scenes(fig, 8, width=1120, height=760).show()
"""
            ),
            md(
                """
### 10.4 Berry-Curvature Slices And Surface-Plane Intersections

The appendix Berry figure used three slicing planes.  The code below exposes
the actual geometric operation: slice the exceptional surface by
`kx=0`, `ky=0`, and `kz=pi/2`, then plot the intersection curves.
"""
            ),
            code(
                r"""
berry_appendix_ske = NodalSkeleton(
    hopf_link_bloch_vector(0.8, k_symbols=(kx, ky, kz)),
    k_symbols=(kx, ky, kz),
    dimension=64,
    axis_scale=(1.0, 1.0, 1.5),
)
berry_appendix_surface = berry_appendix_ske.exceptional_surface_pv.connectivity("largest")
berry_slice_specs = [
    ("kx=0", (1, 0, 0), (0, 0, 0), RED),
    ("ky=0", (0, 1, 0), (0, 0, 0), "#2ca02c"),
    ("kz=pi/2", (0, 0, 1), (0, 0, np.pi / 2), "#ff7f0e"),
]
berry_slices = []
for label, normal, origin, color in berry_slice_specs:
    sliced = berry_appendix_surface.slice(normal=normal, origin=origin)
    berry_slices.append((label, sliced, color))
    print(label, "points_cells =", (sliced.n_points, sliced.n_cells))
"""
            ),
            code(
                r"""
fig = go.Figure()
mesh = berry_appendix_surface.triangulate()
faces = mesh.faces.reshape(-1, 4)[:, 1:]
pts = mesh.points
fig.add_trace(
    go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=BLUE,
        opacity=0.22,
        showscale=False,
    )
)
for label, sliced, color in berry_slices:
    for segment in polyline_traces_from_slice(sliced):
        fig.add_trace(
            go.Scatter3d(
                x=segment[:, 0],
                y=segment[:, 1],
                z=segment[:, 2],
                mode="lines",
                line=dict(color=color, width=7),
                showlegend=False,
            )
        )
apply_kg_layout(fig).show()
"""
            ),
            md(
                """
### 10.5 Intrinsic-Linkedness Minor Check

The appendix included an awesome-surface panel together with an intrinsic
linkedness/Petersen comparison.  The public API exposes this as a graph-minor
question on the extracted skeleton graph.
"""
            ),
            code(
                r"""
awesome_minor_ske = NodalSkeleton(
    awesome_bloch_vector(0.2, k_symbols=(kx, ky, kz), c=0.5),
    k_symbols=(kx, ky, kz),
    dimension=120,
    axis_scale=(1.0, 1.0, 1.5),
)
awesome_minor_surface = awesome_minor_ske.exceptional_surface_pv.connectivity("largest")
awesome_minor_graph = awesome_minor_ske.skeleton_graph(simplify=True, smooth_epsilon=2)
petersen = nx.petersen_graph()
minor_embedding = awesome_minor_ske.check_minor(petersen, awesome_minor_graph)

print("awesome surface =", (awesome_minor_surface.n_points, awesome_minor_surface.n_cells))
print("awesome graph =", (awesome_minor_graph.number_of_nodes(), awesome_minor_graph.number_of_edges()))
print("degree sequence =", sorted(dict(awesome_minor_graph.degree()).values()))
print("petersen minor found =", bool(minor_embedding))
if minor_embedding:
    print("embedding sizes =", {node: len(chain) for node, chain in minor_embedding.items()})
"""
            ),
            code(
                r"""
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    horizontal_spacing=0.03,
)
add_graph_traces(fig, awesome_minor_graph, row=1, col=1)
pos = nx.spring_layout(petersen, seed=4)
for u, v in petersen.edges():
    fig.add_trace(
        go.Scatter(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            mode="lines",
            line=dict(color=BLUE, width=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
fig.add_trace(
    go.Scatter(
        x=[pos[n][0] for n in petersen.nodes()],
        y=[pos[n][1] for n in petersen.nodes()],
        mode="markers",
        marker=dict(size=10, color=RED),
        showlegend=False,
    ),
    row=1,
    col=2,
)
style_plotly_scenes(fig, 1, width=980, height=520)
fig.update_xaxes(visible=False, row=1, col=2)
fig.update_yaxes(visible=False, row=1, col=2, scaleanchor="x", scaleratio=1)
fig.show()
"""
            ),
            md(
                """
### 10.6 Three-Link Planarity Evolution

The planarity appendix compared three `gamma` values.  This cell regenerates
all three from the Hamiltonian model, extracts graphs, and checks planarity.
"""
            ),
            code(
                r"""
three_link_planarity_records = []
for gamma in (0.116, 0.41, 0.5):
    ske_three = NodalSkeleton(
        threelink_bloch_vector(gamma, k_symbols=(kx, ky, kz), c=0.5),
        k_symbols=(kx, ky, kz),
        dimension=80,
        axis_scale=(1.0, 1.0, 1.5),
    )
    surface_three = ske_three.exceptional_surface_pv.connectivity("largest")
    graph_three = ske_three.skeleton_graph(simplify=True, smooth_epsilon=2)
    planar_three = nx.check_planarity(nx.Graph(graph_three))[0]
    three_link_planarity_records.append((gamma, surface_three, graph_three, planar_three))
    print(f"gamma = {gamma}")
    print("  surface_points_cells =", (surface_three.n_points, surface_three.n_cells))
    print("  graph_nodes_edges =", (graph_three.number_of_nodes(), graph_three.number_of_edges()))
    print("  planar =", planar_three)
"""
            ),
            code(
                r"""
fig = make_subplots(
    rows=2,
    cols=3,
    specs=[[{"type": "scene"} for _ in range(3)], [{"type": "scene"} for _ in range(3)]],
    horizontal_spacing=0.01,
    vertical_spacing=0.02,
)
for col, (gamma, surface_three, graph_three, planar_three) in enumerate(three_link_planarity_records, start=1):
    add_surface_trace(fig, surface_three, row=1, col=col, opacity=0.46)
    add_graph_traces(fig, graph_three, row=2, col=col)
style_plotly_scenes(fig, 6, width=1080, height=720).show()
"""
            ),
            md(
                """
## 11. Advanced Projection And Yamada Diagnostics

This section gives more detailed diagnostic workflows for users who want to
understand exactly how a projection, PD code, Yamada state expansion, and
projection-sampling study are produced.
"""
            ),
            md(
                """
### 11.1 PD Code Emergence From The Selected Projection

Select a projection, plot its planar arcs, label vertices/crossings/arcs, and
print the exact PD terms consumed by the Yamada computation.
"""
            ),
            code(
                r"""
trefoil_projection_candidates = sorted(
    sample_projections(coordinate_result.graph, num_rotation_samples=8),
    key=lambda candidate: (candidate.num_crossings, candidate.rotation_angles),
)
pd_emergence_projection = trefoil_projection_candidates[0]

print(f"rotation_angles = {tuple(round(a, 2) for a in pd_emergence_projection.rotation_angles)}")
print(f"crossings = {pd_emergence_projection.num_crossings}")
print("pd_terms:")
for term in sorted(pd_emergence_projection.pd_code.split(";")):
    print(" ", term)
print("vertices =", [(vertex.id, vertex.key) for vertex in pd_emergence_projection.vertices])
print(
    "crossings =",
    [
        (crossing.id, tuple(round(c, 3) for c in tuple(crossing.point.coords)[0]))
        for crossing in pd_emergence_projection.crossings
    ],
)
print(
    "arcs =",
    [
        (arc.id, arc.start_type, arc.start_id, arc.end_type, arc.end_id)
        for arc in pd_emergence_projection.arcs
    ],
)
"""
            ),
            code(
                r"""
fig, ax = plt.subplots(figsize=(6.0, 5.0))
for arc in pd_emergence_projection.arcs:
    ax.plot(*arc.line.xy, color=BLUE, linewidth=2.5, solid_capstyle="round")
    midpoint = arc.line.interpolate(0.5, normalized=True)
    ax.text(midpoint.x, midpoint.y, f"a{arc.id}", fontsize=8, color=BLUE)
for vertex in pd_emergence_projection.vertices:
    ax.scatter(*vertex.point.xy, color=RED, s=72, zorder=3)
    ax.text(vertex.point.x, vertex.point.y, f"v{vertex.id}", fontsize=9, color=RED)
for crossing in pd_emergence_projection.crossings:
    ax.scatter(*crossing.point.xy, marker="x", color="black", s=72, zorder=4)
    ax.text(crossing.point.x, crossing.point.y, f"x{crossing.id}", fontsize=9, color="black")
ax.set_aspect("equal")
ax.axis("off")
plt.show()
"""
            ),
            md(
                """
### 11.2 Yamada Route Details: States Behind `Upsilon(G;Y)`

The public call is `compute_yamada_polynomial(...)`.  The cell below also shows
the internal state route as an inspection tool for debugging and explanatory
figures.
"""
            ),
            code(
                r"""
from knotted_graph.invariants.yamada import Yamada, compute_yamada_from_states

yamada_inspector = Yamada(
    pd_emergence_projection.vertices,
    pd_emergence_projection.crossings,
    pd_emergence_projection.arcs,
)
state_graphs, exponents = yamada_inspector._build_state_graphs()
state_sum = compute_yamada_from_states(state_graphs, exponents, Y, method="negami", n_jobs=1)
public_result = compute_yamada_polynomial(
    coordinate_result.graph,
    Y,
    rotation_angles=pd_emergence_projection.rotation_angles,
    n_jobs=1,
)

print(f"crossings = {pd_emergence_projection.num_crossings}")
print(f"state_count = {len(state_graphs)}")
for index, (state_graph, exponent) in enumerate(zip(state_graphs[:6], exponents[:6])):
    print(
        f"state[{index}] exponent={exponent} "
        f"nodes_edges={(state_graph.number_of_nodes(), state_graph.number_of_edges())}"
    )
print_upsilon("state_sum", state_sum)
print_upsilon("public_call", public_result)
print("same_result =", sp.expand(state_sum - public_result) == 0)
"""
            ),
            code(
                r"""
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
for axis, state_index in zip(axes, range(3)):
    state_graph = state_graphs[state_index]
    simple_state = nx.Graph(state_graph)
    positions = nx.spring_layout(simple_state, seed=10 + state_index)
    nx.draw_networkx_edges(simple_state, positions, ax=axis, edge_color=BLUE, width=2.0)
    nx.draw_networkx_nodes(simple_state, positions, ax=axis, node_color=RED, node_size=90)
    axis.set_aspect("equal")
    axis.axis("off")
plt.show()
"""
            ),
            md(
                """
### 11.3 Trefoil Yamada Calculation From A Coordinate Chain

Coordinates become a graph, a projection is selected, and the polynomial is
printed in the paper notation.
"""
            ),
            code(
                r"""
fig = plot_graph_kg(coordinate_result.graph)
fig.show()
"""
            ),
            code(
                r"""
trefoil_public_result = compute_yamada_polynomial(
    coordinate_result.graph,
    Y,
    rotation_angles=pd_emergence_projection.rotation_angles,
    return_result=True,
    n_jobs=1,
)
print("input_kind =", coordinate_result.graph.graph.get("input_kind"))
print("nodes_edges =", (coordinate_result.graph.number_of_nodes(), coordinate_result.graph.number_of_edges()))
print("selected_crossings =", trefoil_public_result.projection.num_crossings)
print(f"pd_code = {trefoil_public_result.projection.pd_code}")
print_upsilon("G_trefoil", trefoil_public_result.polynomial)
"""
            ),
            md(
                """
### 11.4 Randomized Projection And Scaling Diagnostics

This diagnostic asks whether different sampled directions and geometric
scalings preserve the symbolic fingerprint.  Many projections are sampled for
crossing counts, and only the lowest-crossing projection at each scale is sent
to the Yamada engine.
"""
            ),
            code(
                r"""
projection_diagnostics = []
for x_scale in (0.75, 1.0, 1.25, 1.5):
    scaled_coords = trefoil_coords * np.array([x_scale, 1.0, 1.0])
    scaled_result = from_coordinate_chain(
        scaled_coords,
        input_id=f"trefoil_x_scale_{x_scale}",
        closed=True,
        closure="direct",
    )
    candidates = sorted(
        sample_projections(scaled_result.graph, num_rotation_samples=8),
        key=lambda candidate: (candidate.num_crossings, candidate.rotation_angles),
    )
    crossing_counts = [candidate.num_crossings for candidate in candidates]
    selected = candidates[0]
    polynomial = compute_yamada_polynomial(
        scaled_result.graph,
        Y,
        rotation_angles=selected.rotation_angles,
        n_jobs=1,
    )
    projection_diagnostics.append(
        {
            "x_scale": x_scale,
            "min_crossings": min(crossing_counts),
            "median_crossings": float(np.median(crossing_counts)),
            "max_crossings": max(crossing_counts),
            "selected_angles": tuple(round(a, 2) for a in selected.rotation_angles),
            "polynomial": sp.expand(polynomial),
        }
    )

for row in projection_diagnostics:
    print(f"x_scale = {row['x_scale']}")
    print("  crossing_range =", (row["min_crossings"], row["median_crossings"], row["max_crossings"]))
    print("  selected_angles =", row["selected_angles"])
    print_upsilon("G_scaled", row["polynomial"])
"""
            ),
            code(
                r"""
scales = [row["x_scale"] for row in projection_diagnostics]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.plot(scales, [row["min_crossings"] for row in projection_diagnostics], color=BLUE, marker="o", label="min")
ax.plot(scales, [row["median_crossings"] for row in projection_diagnostics], color="black", marker="s", label="median")
ax.plot(scales, [row["max_crossings"] for row in projection_diagnostics], color=RED, marker="^", label="max")
ax.set_xlabel("x scale")
ax.set_ylabel("projection crossings")
ax.legend(frameon=False)
plt.show()
"""
            ),
            md(
                """
## 12. Library Improvements Exposed By This Notebook

The notebook is runnable, but it also exposes the next API improvements needed
to make the library feel fully public-facing:

- Add `inspect_spatial_graph(...)` returning a pipeline object with input graph,
  simplified graph, projections, selected projection, PD code, and Yamada result.
- Promote `NodalSkeletonMultiBand` or an equivalent material-surface adapter into
  `knotted_graph.applications`.
- Promote the shared plotting style helpers into `knotted_graph.visualization`.
- Promote the appendix subplot helpers (`add_surface_trace`, `add_graph_traces`,
  `style_plotly_scenes`, and slicing utilities) into public visualization
  helpers so users do not have to repeat notebook boilerplate.
- Promote a public `yamada_state_summary(...)` or `Yamada.inspect_states()`
  helper, because state inspection is valuable for paper figures and debugging.
- Add a public volumetric/surface skeletonization adapter that covers custom
  mask workflows without requiring users to copy image-processing utilities.
- Provide bundled example datasets through `knotted_graph.examples` instead of
  relying on local file paths.
- Add a standard `PipelineResult.to_notebook()` or `.summary()` method so every
  application can present intermediate objects consistently.
"""
            ),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
            "knotted_graph": {
                "source": "paper companion",
                "generated_by": "scripts/build_paper_companion_notebook.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_metadata(source: str) -> dict[str, Any]:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
        "knotted_graph": {
            "source": source,
            "generated_by": "scripts/build_paper_companion_notebook.py",
        },
    }


def make_notebook(cells: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": notebook_metadata(source),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def section_number(cell: dict[str, Any]) -> int | None:
    if cell.get("cell_type") != "markdown":
        return None
    source = "".join(cell.get("source", ""))
    match = re.match(r"##\s+(\d+)\.", source)
    if match:
        return int(match.group(1))
    return None


def split_sections(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    preface: list[dict[str, Any]] = []
    sections: dict[int, list[dict[str, Any]]] = {}
    current: int | None = None

    for cell in cells:
        number = section_number(cell)
        if number is not None:
            current = number
            sections.setdefault(current, []).append(cell)
            continue

        if current is None:
            preface.append(cell)
        else:
            sections.setdefault(current, []).append(cell)

    return preface, sections


def guide_chapter_intro(title: str, filename: str, dependencies: list[int]) -> dict[str, Any]:
    if dependencies:
        dependency_text = "\n".join(
            f"- includes prerequisite cells from `{SECTION_NOTEBOOKS[number][0]}`"
            for number in dependencies
        )
    else:
        dependency_text = "- self-contained after the shared setup cells"

    return md(
        f"""
# {title}

This notebook is one chapter of the runnable `KnottedGraph` user guide.  It is
generated into `User_guide/{filename}` so users can open the specific workflow
they need without navigating one very large notebook.

{dependency_text}
"""
    )


def nodal_imports_cell() -> dict[str, Any]:
    return code(
        r"""
from knotted_graph.applications.nodal import NodalSkeleton
from knotted_graph.applications.nodal.models import (
    awesome_bloch_vector,
    hopf_link_bloch_vector,
    pq_torus_knot_bloch_vector,
    solomon_bloch_vector,
    threelink_bloch_vector,
    trefoil_bloch_vector,
    unknot_bloch_vector,
)

print("nodal application imports ready")
""",
        tags=["setup"],
    )


def section_notebooks(full_notebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    preface, sections = split_sections(full_notebook["cells"])
    setup_cells = sections.get(0, [])
    notebooks: dict[str, dict[str, Any]] = {}

    for number, (filename, title) in SECTION_NOTEBOOKS.items():
        dependencies = SECTION_DEPENDENCIES.get(number, [])
        chapter_cells = [guide_chapter_intro(title, filename, dependencies)]

        if number == 0:
            chapter_cells.extend(preface)
            chapter_cells.extend(setup_cells)
        else:
            chapter_cells.extend(setup_cells)
            if number in NODAL_SECTION_IMPORTS:
                chapter_cells.append(nodal_imports_cell())
            for dependency_number in dependencies:
                chapter_cells.extend(sections.get(dependency_number, []))
            chapter_cells.extend(sections.get(number, []))

        notebooks[filename] = make_notebook(
            chapter_cells,
            source=f"user guide chapter {number}: {title}",
        )

    return notebooks


def split_cells_at_markdown(cells: list[dict[str, Any]], marker: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", ""))
        if marker in source:
            return cells[:index], cells[index:]
    return cells, []


def application_notebooks(full_notebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _preface, sections = split_sections(full_notebook["cells"])
    setup_cells = sections.get(0, [])
    application_cells = sections.get(8, [])
    nodal_cells, material_cells = split_cells_at_markdown(
        application_cells,
        "### Material Fermi-Surface Examples",
    )

    notebooks: dict[str, dict[str, Any]] = {}
    notebooks["08a_application_nodal_skeletons.ipynb"] = make_notebook(
        [
            guide_chapter_intro(
                "Application: Non-Hermitian Nodal Skeletons",
                "08a_application_nodal_skeletons.ipynb",
                [],
            ),
            *setup_cells,
            nodal_imports_cell(),
            *nodal_cells,
        ],
        source="user guide application: nodal skeletons",
    )
    notebooks["08b_application_material_fingerprints.ipynb"] = make_notebook(
        [
            guide_chapter_intro(
                "Application: Material Fermi-Surface Fingerprints",
                "08b_application_material_fingerprints.ipynb",
                [],
            ),
            *setup_cells,
            nodal_imports_cell(),
            *material_cells,
        ],
        source="user guide application: material fingerprints",
    )
    return notebooks


def user_guide_notebook() -> dict[str, Any]:
    cells = [
        md(
            """
# KnottedGraph User Guide

This folder is the notebook home for the library.  Open this file first, then
choose the notebook matching the workflow you want to learn.  Each referenced
notebook contains code followed by its generated output; the root project folder
does not carry a separate tutorial notebook.
"""
        ),
        md(
            """
## Recommended Reading Order

1. [Quick Start](01_quick_start.ipynb)  
   The shortest path: surface, spatial graph, projection, PD code, and
   `Upsilon(G;Y)`.

2. [Input Adapters](02_input_adapters.ipynb)  
   Coordinate chains, polymers, CSV spatial graphs, biomolecules, mmCIF files,
   and surface meshes.

3. [Inspection Mode](03_inspection_mode.ipynb)  
   Surface, mask, skeleton points, raw graph, simplified graph, selected
   projection, PD code, and Yamada result.

4. [Projection, PD Codes, And Yamada](05_projection_pd_yamada.ipynb)  
   Projection sampling, rigid-vertex data, and invariant calculation.

5. [Application Gallery](08_application_gallery.ipynb)  
   Application overview.  For individual application notebooks, open
   [Nodal Skeletons](08a_application_nodal_skeletons.ipynb),
   [Material Fingerprints](08b_application_material_fingerprints.ipynb), or
   [Proteins And Repulsive Curves](07_repulsive_curves_and_proteins.ipynb).

6. [Mathematical Workflows](06_mathematical_workflows.ipynb)  
   Structured graph families, catalog plots, Laurent Yamada polynomials, and
   sigma-form conversions.
"""
        ),
        md(
            """
## Notebook Map

| Notebook | Purpose |
| --- | --- |
| [00_setup_preflight.ipynb](00_setup_preflight.ipynb) | Shared imports, paths, plotting style, and notation helpers. |
| [01_quick_start.ipynb](01_quick_start.ipynb) | Main end-to-end pipeline. |
| [02_input_adapters.ipynb](02_input_adapters.ipynb) | Data/file inputs to package graph or mesh objects. |
| [03_inspection_mode.ipynb](03_inspection_mode.ipynb) | Full intermediate-object inspection. |
| [04_physical_fields.ipynb](04_physical_fields.ipynb) | Hamiltonian fields, Berry views, and surface-mode plots. |
| [05_projection_pd_yamada.ipynb](05_projection_pd_yamada.ipynb) | Projection choices, PD code, and Yamada engines. |
| [06_mathematical_workflows.ipynb](06_mathematical_workflows.ipynb) | Graph-family catalog and mathematical Yamada experiments. |
| [07_repulsive_curves_and_proteins.ipynb](07_repulsive_curves_and_proteins.ipynb) | Protein theta graphs and repulsive-curve workflow placeholder. |
| [08_application_gallery.ipynb](08_application_gallery.ipynb) | Application overview notebook. |
| [08a_application_nodal_skeletons.ipynb](08a_application_nodal_skeletons.ipynb) | Non-Hermitian nodal skeleton application examples. |
| [08b_application_material_fingerprints.ipynb](08b_application_material_fingerprints.ipynb) | Material Hamiltonians and the material-fingerprint API gap. |
| [09_paper_figure_map.ipynb](09_paper_figure_map.ipynb) | Paper figure reproduction map. |
| [10_appendix_workflows.ipynb](10_appendix_workflows.ipynb) | Appendix-style surface, graph, Berry, planarity, and minor examples. |
| [11_advanced_diagnostics.ipynb](11_advanced_diagnostics.ipynb) | PD-code emergence, Yamada states, and projection diagnostics. |
| [12_library_improvements.ipynb](12_library_improvements.ipynb) | API improvements exposed by the notebooks. |
| [website_examples.ipynb](website_examples.ipynb) | Website companion notebook, generated by the website notebook builder. |
"""
        ),
        md(
            """
## Ownership Notes

# To be filled by Zhaoyun

Input types and input adapters should be finalized by Zhaoyun.  This includes
coordinate chains, spatial graph CSV files, PDB/mmCIF biomolecular inputs,
polymer trajectories, and surface meshes.

# To be filled by Kehan

Protein-derived theta graphs and repulsive-curve workflows should be finalized
by Kehan.  This includes the protein example selection, repulsive layout
before/after panels, and topology-preservation checks.
"""
        ),
        md(
            """
## How To Use This Folder

Use `user_guide.ipynb` as the router.  Open one chapter notebook at a time when
you want to learn a specific workflow.  When editing examples, keep code and
its generated figure/output together in the same notebook section.
"""
        ),
    ]

    return make_notebook(cells, source="user guide index")


def main() -> None:
    coverage_rows = collect_source_coverage()
    full_notebook = notebook()
    guide = user_guide_notebook()
    chapters = section_notebooks(full_notebook)
    chapters.update(application_notebooks(full_notebook))
    USER_GUIDE_DIR.mkdir(parents=True, exist_ok=True)
    USER_GUIDE_OUT.write_text(json.dumps(guide, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for filename, chapter in chapters.items():
        (USER_GUIDE_DIR / filename).write_text(
            json.dumps(chapter, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for notebook_path in LEGACY_ROOT_NOTEBOOKS:
        if notebook_path.exists():
            notebook_path.unlink()
    print(f"source_coverage_rows_checked = {len(coverage_rows)}")
    print(f"wrote {USER_GUIDE_OUT}")
    print(f"user_guide_cells = {len(guide['cells'])}")
    print(f"wrote_chapter_notebooks = {len(chapters)}")
    for filename, chapter in chapters.items():
        print(
            f"  {filename}: cells={len(chapter['cells'])} "
            f"code={sum(c['cell_type'] == 'code' for c in chapter['cells'])} "
            f"markdown={sum(c['cell_type'] == 'markdown' for c in chapter['cells'])}"
        )


if __name__ == "__main__":
    main()
