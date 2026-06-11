"""Main-text Task 2 input-format figure.

This script renders the advisor-feedback version of the main 3x3 figure:
one visually representative example per user-facing input type.  The appendix
figures can show broader domain diversity; this figure prioritizes format
diversity.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import pyvista as pv
from PIL import Image


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
ROOT = EXAMPLES_DIR.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PANEL_DIR = FIGURE_DIR / "main_text_panels"
SUMMARY_PATH = DATA_DIR / "main_text_input_gallery_summary.json"
SELECTION_PATH = HERE / "main_text_selection.json"
FIGURE_STEM = "main_text_input_gallery"

sys.path.insert(0, str(HERE))
for relative in [
    "coordinate_chains",
    "spatial_graphs",
    "volumetric_fields",
]:
    sys.path.insert(0, str(EXAMPLES_DIR / relative))

from volumetric_field_adapter import build_surface_from_scalar_field_file, write_npz_scalar_field
from spatial_graph_adapter import build_spatial_graph_from_json, write_spatial_graph_json
from knotted_graph.inputs import (
    from_coordinate_chain,
    from_gromacs_gro,
    from_lammps_dump,
    from_mmcif_backbone,
    from_protein_ca_backbone,
    from_spatial_graph_csv,
    from_surface_mesh,
    validate_spatial_graph,
    write_gro_coords,
    write_lammps_dump,
)
from knotted_graph.inputs.coordinate_chain import write_xyz_coords

from plot_publication_style_gallery import (
    EDGE_COLOR,
    FERMI_COLOR,
    NODE_COLOR,
    SURFACE_COLOR,
    VOLUME_COLOR,
    add_point_cloud,
    add_raw_spatial_graph,
    add_source_beads_and_bonds,
    add_source_backbone_trace,
    add_spatial_graph,
    crop_white,
    edge_points,
    load_mmcif_atom_points,
    load_pdb_atom_points,
    make_plotter,
    point_span,
    polyline_mesh,
    save_image_array,
    set_camera,
)
from plot_publication_style_gallery_sets import (
    make_gyroid_field,
    make_nodal_fermi_mesh,
)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "figure.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

BIOLOGY_SEGMENT_COLORS = ["#2563A8", "#1B8A8F", "#4B9D3A", "#D99B1B", "#B33F62"]
SOURCE_GREY = "#a8afb3"


def label_with_format(source_name: str, input_format: str) -> str:
    return f"{source_name} ({input_format})"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for old_panel in PANEL_DIR.glob("*.png"):
        old_panel.unlink()


def write_image(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    return path


def render_scene(draw_fn, image_path: Path) -> tuple[np.ndarray, Path, list[str]]:
    plotter = make_plotter()
    result = draw_fn(plotter)
    if len(result) == 2:
        meshes, points = result
        issues: list[str] = []
    else:
        meshes, points, issues = result
    set_camera(plotter, points)
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    cropped = crop_white(image, threshold=249, pad=10)
    write_image(cropped, image_path)
    return cropped, image_path, issues


def render_source(draw_fn, image_path: Path) -> tuple[np.ndarray, Path]:
    plotter = make_plotter(window_size=(900, 760))
    meshes, points = draw_fn(plotter)
    set_camera(plotter, points)
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    cropped = crop_white(image, threshold=249, pad=8)
    write_image(cropped, image_path)
    return cropped, image_path


def add_tube(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    radius: float,
    color: str,
    opacity: float = 1.0,
    n_sides: int = 32,
) -> pv.PolyData:
    tube = polyline_mesh(np.asarray(points, dtype=float)).tube(radius=radius, n_sides=n_sides, capping=True)
    plotter.add_mesh(
        tube,
        color=color,
        opacity=opacity,
        smooth_shading=True,
        specular=0.45,
        specular_power=24,
        metallic=0.08,
    )
    return tube


def add_endpoint_nodes(plotter: pv.Plotter, points: np.ndarray, *, radius: float) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    meshes: list[pv.DataSet] = []
    for endpoint in (pts[0], pts[-1]):
        sphere = pv.Sphere(radius=radius, center=endpoint, theta_resolution=32, phi_resolution=16)
        plotter.add_mesh(sphere, color=NODE_COLOR, smooth_shading=True, specular=0.45)
        meshes.append(sphere)
    return meshes


def add_segmented_curve(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    colors: list[str] = BIOLOGY_SEGMENT_COLORS,
    closed: bool = False,
    direct_closure: bool = False,
    add_endpoints: bool = True,
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    _, _, _, span = point_span(pts)
    radius = 0.018 * span
    meshes: list[pv.DataSet] = []
    split_indices = np.linspace(0, pts.shape[0] - 1, len(colors) + 1).astype(int)
    for index, color in enumerate(colors):
        start = int(split_indices[index])
        stop = int(split_indices[index + 1]) + 1
        segment = pts[start:stop]
        if segment.shape[0] < 2:
            continue
        meshes.append(add_tube(plotter, segment, radius=radius, color=color))
    if direct_closure and pts.shape[0] >= 2:
        closure = np.vstack([pts[-1], pts[0]])
        meshes.append(add_tube(plotter, closure, radius=0.72 * radius, color="#656b70", opacity=0.58, n_sides=24))
    if add_endpoints and not closed:
        meshes.extend(add_endpoint_nodes(plotter, pts, radius=2.5 * radius))
    return meshes


def add_curve(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    color: str = EDGE_COLOR,
    closed: bool = False,
    direct_closure: bool = False,
    add_endpoints: bool = False,
) -> list[pv.DataSet]:
    pts = np.asarray(points, dtype=float)
    _, _, _, span = point_span(pts)
    radius = 0.018 * span
    render_pts = np.vstack([pts, pts[0]]) if direct_closure and pts.shape[0] >= 2 else pts
    meshes = [add_tube(plotter, render_pts, radius=radius, color=color)]
    if add_endpoints and not closed:
        meshes.extend(add_endpoint_nodes(plotter, pts, radius=2.5 * radius))
    return meshes


def add_surface_safe(plotter: pv.Plotter, mesh: pv.PolyData, *, color: str) -> list[pv.DataSet]:
    mesh = mesh.triangulate().clean()
    if mesh.n_cells > 6000:
        try:
            mesh = mesh.decimate_pro(0.55, preserve_topology=True)
        except Exception:
            pass
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=0.94,
        smooth_shading=True,
        specular=0.45,
        specular_power=22,
        metallic=0.08,
    )
    return [mesh]


def make_torus_knot(n_points: int = 240, p: int = 2, q: int = 3, scale: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    major = 1.6 + 0.45 * np.cos(q * t)
    return scale * np.column_stack(
        [
            major * np.cos(p * t),
            major * np.sin(p * t),
            0.75 * np.sin(q * t),
        ]
    )


def make_lammps_polymer_curve(n_points: int = 280, scale: float = 1.0) -> np.ndarray:
    """Closed simulation-style polymer curve distinct from the cinquefoil panel."""
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    points = np.column_stack(
        [
            1.10 * np.sin(2.0 * t) + 0.52 * np.sin(5.0 * t + 0.55),
            1.05 * np.sin(3.0 * t + 0.45) + 0.40 * np.cos(7.0 * t - 0.30),
            0.82 * np.sin(4.0 * t + 1.10) + 0.30 * np.sin(9.0 * t),
        ]
    )
    points -= points.mean(axis=0)
    points /= np.max(np.linalg.norm(points, axis=1))
    return scale * points


def make_engineering_network_payload() -> dict:
    """Curved node/edge CSV example for spatial topology in engineering systems."""
    nodes = {
        "pump": [-1.65, -0.45, 0.00],
        "controller": [-0.45, 0.85, 0.35],
        "heat_exchanger": [0.95, 0.55, -0.10],
        "battery": [1.55, -0.70, 0.25],
        "sensor": [-0.10, -1.05, -0.35],
    }

    def arc(source: str, target: str, lift: float, side: float, bend: float) -> list[list[float]]:
        start = np.asarray(nodes[source], dtype=float)
        end = np.asarray(nodes[target], dtype=float)
        t = np.linspace(0.0, 1.0, 80)
        base = (1.0 - t)[:, None] * start + t[:, None] * end
        tangent = end - start
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(normal) < 1e-8:
            normal = np.array([1.0, 0.0, 0.0])
        normal = normal / np.linalg.norm(normal)
        base += side * np.sin(np.pi * t)[:, None] * normal
        base += bend * np.sin(2.0 * np.pi * t)[:, None] * np.array([0.0, 0.0, 1.0])
        base[:, 2] += lift * np.sin(np.pi * t)
        return base.tolist()

    edge_specs = [
        ("pump", "controller", 0.80, 0.25, 0.08, "coolant_loop"),
        ("pump", "sensor", -0.35, -0.20, 0.04, "signal"),
        ("controller", "heat_exchanger", 0.25, -0.32, -0.10, "control_bus"),
        ("controller", "battery", -0.45, 0.40, 0.08, "power"),
        ("sensor", "heat_exchanger", 0.65, 0.30, -0.05, "return_pipe"),
        ("heat_exchanger", "battery", 0.25, -0.16, 0.05, "thermal_link"),
        ("pump", "battery", 1.10, -0.55, 0.12, "overhead_cable"),
    ]
    return {
        "graph_id": "engineering_network_csv",
        "nodes": [
            {"node_id": node_id, "pos": pos, "label": node_id.replace("_", " ").title(), "type": "component"}
            for node_id, pos in nodes.items()
        ],
        "edges": [
            {
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "type": edge_id,
                "points_json": arc(source, target, lift, side, bend),
            }
            for source, target, lift, side, bend, edge_id in edge_specs
        ],
    }


def curved_edge_path(
    start,
    end,
    *,
    lift: float = 0.0,
    side: float = 0.0,
    twist: float = 0.0,
    n_points: int = 84,
) -> list[list[float]]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, n_points)
    pts = (1.0 - t)[:, None] * start + t[:, None] * end
    tangent = end - start
    norm = np.linalg.norm(tangent)
    if norm < 1e-10:
        return pts.tolist()
    tangent = tangent / norm
    side_axis = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side_axis) < 1e-8:
        side_axis = np.array([1.0, 0.0, 0.0])
    side_axis = side_axis / np.linalg.norm(side_axis)
    pts += side * np.sin(np.pi * t)[:, None] * side_axis
    pts[:, 2] += lift * np.sin(np.pi * t)
    pts += twist * np.sin(2.0 * np.pi * t)[:, None] * np.array([0.0, 0.0, 1.0])
    return pts.tolist()


def make_json_spatial_graph_payload() -> dict:
    nodes = {
        "left": [-1.45, -0.10, 0.00],
        "right": [1.45, 0.06, 0.00],
        "top": [0.02, 0.94, 0.48],
        "bottom": [-0.06, -0.98, -0.38],
        "hub": [0.10, 0.00, 0.16],
    }
    edge_specs = [
        ("left", "right", 0.88, 0.44, 0.08, "upper_route"),
        ("left", "right", -0.70, -0.48, -0.08, "lower_route"),
        ("left", "top", 0.24, -0.18, 0.04, "left_top"),
        ("top", "right", -0.18, -0.16, -0.05, "top_right"),
        ("right", "bottom", 0.30, -0.20, 0.05, "right_bottom"),
        ("bottom", "left", -0.10, -0.14, 0.06, "bottom_left"),
        ("top", "bottom", 0.46, 0.28, -0.05, "vertical_cross_route"),
        ("left", "hub", -0.18, 0.10, 0.04, "left_hub"),
        ("hub", "right", 0.20, 0.10, -0.04, "hub_right"),
    ]
    return {
        "graph_id": "theta_style_spatial_graph_json",
        "metadata": {"domain": "abstract spatial graph", "source_format": "json"},
        "nodes": [{"id": node_id, "pos": pos, "label": node_id} for node_id, pos in nodes.items()],
        "edges": [
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "points": curved_edge_path(nodes[source], nodes[target], lift=lift, side=side, twist=twist),
            }
            for source, target, lift, side, twist, edge_id in edge_specs
        ],
    }


def write_neuron_swc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        (1, 1, 0.00, 0.00, 0.00, 0.18, -1),
        (2, 3, -0.34, 0.46, 0.24, 0.07, 1),
        (3, 3, -0.78, 0.95, 0.44, 0.055, 2),
        (4, 3, -1.18, 1.38, 0.18, 0.042, 3),
        (5, 3, -0.42, 1.22, 0.84, 0.044, 3),
        (6, 3, 0.45, 0.44, -0.10, 0.07, 1),
        (7, 3, 0.94, 0.84, -0.42, 0.055, 6),
        (8, 3, 1.40, 1.20, -0.12, 0.042, 7),
        (9, 3, 1.18, 0.88, -0.88, 0.042, 7),
        (10, 2, -0.16, -0.50, -0.28, 0.075, 1),
        (11, 2, -0.58, -0.98, -0.58, 0.055, 10),
        (12, 2, -1.00, -1.42, -0.36, 0.043, 11),
        (13, 2, 0.42, -0.92, -0.46, 0.055, 10),
        (14, 2, 0.88, -1.38, -0.74, 0.043, 13),
    ]
    lines = ["# synthetic SWC-style 3D neuron morphology for input-format gallery"]
    lines.extend(" ".join(str(value) for value in record) for record in records)
    path.write_text("\n".join(lines) + "\n")


def spatial_graph_from_swc(path: Path) -> tuple[nx.MultiGraph, list[str]]:
    graph = nx.MultiGraph()
    graph.graph.update({"input_kind": "neuron_morphology_swc", "source_format": "swc", "source_path": str(path)})
    parent_by_node: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"SWC line {line_number}: expected at least 7 columns.")
        node_id, swc_type = parts[0], parts[1]
        x, y, z, radius = (float(value) for value in parts[2:6])
        parent = parts[6]
        graph.add_node(node_id, pos=np.array([x, y, z], dtype=float), swc_type=swc_type, radius=radius)
        parent_by_node[node_id] = parent
    for node_id, parent in parent_by_node.items():
        if parent == "-1":
            continue
        if parent not in graph:
            raise ValueError(f"SWC node {node_id!r} references unknown parent {parent!r}.")
        source_pos = graph.nodes[parent]["pos"]
        target_pos = graph.nodes[node_id]["pos"]
        graph.add_edge(parent, node_id, key=f"{parent}_{node_id}", pts=np.vstack([source_pos, target_pos]), swc_edge=True)
    return graph, validate_spatial_graph(graph)


def write_graphml_spatial_network(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = {
        "n1": [-1.28, -0.76, -0.28],
        "n2": [-1.05, 0.72, 0.24],
        "n3": [-0.18, -0.08, 0.92],
        "n4": [0.18, -0.96, 0.10],
        "n5": [0.72, 0.76, -0.30],
        "n6": [1.32, -0.38, 0.32],
    }
    graph = nx.MultiGraph()
    for node_id, pos in nodes.items():
        graph.add_node(node_id, x=pos[0], y=pos[1], z=pos[2], label=node_id)
    edge_specs = [
        ("n1", "n2", 0.22, 0.18, 0.04, "frame_12"),
        ("n2", "n3", 0.36, -0.16, 0.05, "frame_23"),
        ("n3", "n6", -0.28, -0.22, -0.04, "frame_36"),
        ("n6", "n4", 0.18, 0.20, 0.04, "frame_64"),
        ("n4", "n1", -0.16, -0.18, -0.04, "frame_41"),
        ("n1", "n5", 0.70, -0.42, 0.06, "overpass_15"),
        ("n2", "n6", -0.58, 0.34, -0.08, "underpass_26"),
        ("n5", "n4", 0.40, -0.18, 0.04, "brace_54"),
        ("n3", "n5", 0.12, 0.16, 0.02, "brace_35"),
    ]
    for source, target, lift, side, twist, edge_id in edge_specs:
        graph.add_edge(
            source,
            target,
            key=edge_id,
            edge_id=edge_id,
            points_json=json.dumps(curved_edge_path(nodes[source], nodes[target], lift=lift, side=side, twist=twist)),
        )
    nx.write_graphml(graph, path)


def spatial_graph_from_graphml(path: Path) -> tuple[nx.MultiGraph, list[str]]:
    raw = nx.read_graphml(path, force_multigraph=True)
    graph = nx.MultiGraph()
    graph.graph.update({"input_kind": "spatial_network_graphml", "source_format": "graphml", "source_path": str(path)})
    for node_id, data in raw.nodes(data=True):
        pos = np.array([float(data["x"]), float(data["y"]), float(data["z"])], dtype=float)
        attrs = {key: value for key, value in data.items() if key not in {"x", "y", "z"}}
        graph.add_node(str(node_id), pos=pos, **attrs)
    for edge_index, edge in enumerate(raw.edges(keys=True, data=True)):
        source, target, key, data = edge
        points_json = data.get("points_json")
        if points_json:
            pts = np.asarray(json.loads(points_json), dtype=float)
        else:
            pts = np.vstack([graph.nodes[str(source)]["pos"], graph.nodes[str(target)]["pos"]])
        edge_key = str(data.get("edge_id", key if key is not None else f"edge_{edge_index}"))
        attrs = {attr_key: value for attr_key, value in data.items() if attr_key not in {"points_json", "edge_id"}}
        graph.add_edge(str(source), str(target), key=edge_key, pts=pts, **attrs)
    return graph, validate_spatial_graph(graph)


def write_spatial_graph_csv_payload(payload: dict, nodes_path: Path, edges_path: Path) -> None:
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_path.open("w") as handle:
        handle.write("node_id,x,y,z,label,type\n")
        for node in payload["nodes"]:
            x, y, z = node["pos"]
            handle.write(f"{node['node_id']},{x},{y},{z},{node['label']},{node['type']}\n")
    with edges_path.open("w") as handle:
        handle.write("edge_id,source,target,label,type,points_json\n")
        for edge in payload["edges"]:
            handle.write(
                f"{edge['edge_id']},{edge['source']},{edge['target']},"
                f"{edge['edge_id'].replace('_', ' ').title()},{edge['type']},"
                f"\"{json.dumps(edge['points_json'])}\"\n"
            )


def scalar_slices_image(values: np.ndarray, output_path: Path) -> tuple[np.ndarray, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.35), dpi=240)
    slices = [
        values[:, :, values.shape[2] // 2].T,
        values[:, values.shape[1] // 2, :].T,
        values[values.shape[0] // 2, :, :].T,
    ]
    for ax, slice_data in zip(axes, slices):
        ax.imshow(slice_data, cmap="Greys", origin="lower")
        ax.contour(slice_data, levels=[0.0], colors=[EDGE_COLOR], linewidths=0.65, origin="lower")
        ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    save_image_array(image, output_path)
    return image, output_path


def assemble_figure(panels: list[dict]) -> tuple[Path, Path, Path]:
    labels = [f"({chr(ord('a') + i)})" for i in range(len(panels))]
    fig = plt.figure(figsize=(17.5, 14.0), facecolor="white")
    outer = fig.add_gridspec(
        3,
        3,
        left=0.016,
        right=0.984,
        top=0.985,
        bottom=0.015,
        wspace=0.040,
        hspace=0.110,
    )
    for i, panel in enumerate(panels):
        r = i // 3
        c = i % 3
        cell_bbox = outer[r, c].get_position(fig)
        frame_outset = 0.004
        fig.add_artist(
            Rectangle(
                (cell_bbox.x0 - frame_outset, cell_bbox.y0 - frame_outset),
                cell_bbox.width + 2.0 * frame_outset,
                cell_bbox.height + 2.0 * frame_outset,
                transform=fig.transFigure,
                fill=False,
                edgecolor="#c4c9ce",
                linewidth=1.18,
                zorder=30,
                clip_on=False,
            )
        )
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[0.13, 1.0], hspace=0.04)
        title_ax = fig.add_subplot(inner[0, 0])
        body = inner[1, 0].subgridspec(1, 2, width_ratios=[0.78, 0.22], wspace=0.035)
        image_ax = fig.add_subplot(body[0, 0])
        source_ax = fig.add_subplot(body[0, 1])
        for ax in (title_ax, image_ax, source_ax):
            ax.axis("off")

        title_ax.text(
            0.035,
            0.52,
            labels[i],
            transform=title_ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="center",
        )
        title_ax.text(
            0.53,
            0.52,
            panel["title"],
            transform=title_ax.transAxes,
            fontsize=15.0,
            fontweight="semibold",
            ha="center",
            va="center",
        )
        image_ax.imshow(panel["image"])
        image_ax.set_box_aspect(1)
        source_ax.imshow(panel["source_image"])
        source_ax.set_box_aspect(1)
        for spine in source_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.75)
        source_ax.text(
            0.07,
            0.92,
            "source",
            transform=source_ax.transAxes,
            fontsize=8,
            fontweight="semibold",
            ha="left",
            va="top",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
        )

    base = FIGURE_DIR / FIGURE_STEM
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(svg_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(pdf_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    plt.close(fig)
    return png_path, svg_path, pdf_path


def main() -> None:
    ensure_dirs()
    selection = json.loads(SELECTION_PATH.read_text())
    panels: list[dict] = []
    summary: list[dict] = []

    def add_panel(title: str, output_stem: str, input_type: str, domain: str, input_format: str, draw_fn, source_fn, notes=None) -> None:
        image, panel_path, issues = render_scene(draw_fn, PANEL_DIR / f"{output_stem}.png")
        source_image, source_path = render_source(source_fn, PANEL_DIR / f"{output_stem}_source.png")
        panels.append({"title": title, "image": image, "source_image": source_image})
        summary.append(
            {
                "title": title,
                "input_type": input_type,
                "domain": domain,
                "input_format": input_format,
                "panel_path": str(panel_path),
                "source_view_path": str(source_path),
                "success": panel_path.exists() and panel_path.stat().st_size > 0 and source_path.exists() and source_path.stat().st_size > 0,
                "issues": issues,
                "notes": notes or [],
            }
        )

    protein = from_protein_ca_backbone("1J85", chain_id="A", data_dir=DATA_DIR / "proteins", save_coords=True)

    def draw_protein(plotter):
        pts = edge_points(protein.graph)
        meshes = add_segmented_curve(plotter, pts, direct_closure=True, add_endpoints=True)
        return meshes, pts, protein.issues

    def source_protein(plotter):
        atom_points = load_pdb_atom_points(protein.pdb_path, chain_id=protein.chain_id, model_id=protein.model_id)
        meshes = []
        meshes.extend(add_point_cloud(plotter, atom_points, color=SOURCE_GREY, opacity=0.30, point_size=5.5))
        meshes.extend(add_segmented_curve(plotter, protein.coords, direct_closure=False, add_endpoints=False))
        return meshes, np.vstack([atom_points, protein.coords])

    add_panel(
        label_with_format("Protein Backbone", "PDB"),
        "a_protein_backbone_pdb",
        "PDB",
        "protein",
        ".pdb",
        draw_protein,
        source_protein,
        notes=["1J85 shown as a conservative protein-backbone PDB example; knotted-protein provenance not claimed in this figure."],
    )

    rna = from_mmcif_backbone("1EHZ", chain_id="A", atom_name="P", data_dir=DATA_DIR / "mmcif", save_coords=True)

    def draw_rna(plotter):
        pts = edge_points(rna.graph)
        meshes = add_segmented_curve(plotter, pts, direct_closure=True, add_endpoints=True)
        return meshes, pts, rna.issues

    def source_rna(plotter):
        atom_points = load_mmcif_atom_points(rna.cif_path, chain_id=rna.chain_id, model_id=rna.model_id)
        meshes = []
        meshes.extend(add_point_cloud(plotter, atom_points, color=SOURCE_GREY, opacity=0.30, point_size=5.5))
        meshes.extend(add_segmented_curve(plotter, rna.coords, direct_closure=False, add_endpoints=False))
        return meshes, np.vstack([atom_points, rna.coords])

    add_panel(label_with_format("tRNA", "mmCIF"), "b_trna_mmcif", "mmCIF", "RNA", ".cif", draw_rna, source_rna)

    main_text_data = DATA_DIR / "main_text"
    gro_path = main_text_data / "ring_polymer.gro"
    write_gro_coords(make_torus_knot(n_points=230, p=1, q=2, scale=0.88), gro_path, title="ring polymer main-text source snapshot")
    ring = from_gromacs_gro(gro_path, closed=True, closure="direct", polymer_id="ring_polymer_gro")

    def draw_gro(plotter):
        pts = edge_points(ring.graph)
        meshes = add_curve(plotter, pts, closed=True)
        return meshes, pts, ring.issues

    def source_gro(plotter):
        meshes = add_source_beads_and_bonds(plotter, ring.coords)
        return meshes, ring.coords

    add_panel(label_with_format("Ring Polymer", "GRO"), "c_ring_polymer_gro", "GRO", "polymer", ".gro", draw_gro, source_gro)

    dump_path = main_text_data / "polymer_lammps.dump"
    write_lammps_dump(make_lammps_polymer_curve(n_points=280, scale=1.15), dump_path, molecule_id=7)
    polymer = from_lammps_dump(dump_path, molecule_id=7, closed=True, closure="direct", polymer_id="polymer_lammps")

    def draw_lammps(plotter):
        pts = edge_points(polymer.graph)
        meshes = add_curve(plotter, pts, closed=True)
        return meshes, pts, polymer.issues

    def source_lammps(plotter):
        meshes = add_source_beads_and_bonds(plotter, polymer.coords)
        return meshes, polymer.coords

    add_panel(
        label_with_format("Polymer", "LAMMPS"),
        "d_polymer_lammps",
        "LAMMPS dump",
        "polymer",
        "LAMMPS dump",
        draw_lammps,
        source_lammps,
    )

    xyz_path = DATA_DIR / "coordinate_chains" / "cinquefoil_coordinate_chain.xyz"
    if not xyz_path.exists():
        write_xyz_coords(make_torus_knot(n_points=260, p=2, q=5, scale=1.05), xyz_path, comment="cinquefoil coordinate chain")
    cinquefoil = from_coordinate_chain(xyz_path, closed=True, closure="direct", input_id="cinquefoil_coordinate_chain")

    def draw_xyz(plotter):
        pts = edge_points(cinquefoil.graph)
        meshes = add_curve(plotter, pts, closed=True)
        return meshes, pts, cinquefoil.issues

    def source_xyz(plotter):
        pts = edge_points(cinquefoil.graph)
        meshes = add_source_beads_and_bonds(plotter, pts)
        return meshes, pts

    add_panel(label_with_format("Cinquefoil Knot", "XYZ"), "e_cinquefoil_xyz", "XYZ coordinate chain", "coordinate knot", ".xyz", draw_xyz, source_xyz)

    nodes_path = DATA_DIR / "spatial_graphs" / "engineering_network_nodes.csv"
    edges_path = DATA_DIR / "spatial_graphs" / "engineering_network_edges.csv"
    write_spatial_graph_csv_payload(make_engineering_network_payload(), nodes_path, edges_path)
    spatial = from_spatial_graph_csv(nodes_path, edges_path, graph_id="engineering_network_csv")

    def draw_csv(plotter):
        meshes, pts = add_spatial_graph(plotter, spatial.graph)
        return meshes, pts, spatial.issues

    def source_csv(plotter):
        return add_raw_spatial_graph(plotter, spatial.graph)

    add_panel(
        label_with_format("Engineering Network", "CSV"),
        "f_engineering_network_csv",
        "Spatial Graph CSV",
        "engineering spatial network",
        "node/edge CSV",
        draw_csv,
        source_csv,
        notes=["Rendered with curved points_json edge paths, not straight source-target edges."],
    )

    json_path = DATA_DIR / "main_text" / "theta_spatial_graph.json"
    write_spatial_graph_json(make_json_spatial_graph_payload(), json_path)
    json_graph = build_spatial_graph_from_json(json_path)

    def draw_json_graph(plotter):
        meshes, pts = add_spatial_graph(plotter, json_graph.graph)
        return meshes, pts, json_graph.issues

    def source_json_graph(plotter):
        return add_raw_spatial_graph(plotter, json_graph.graph)

    add_panel(
        label_with_format("Spatial Graph", "JSON"),
        "g_spatial_graph_json",
        "Spatial Graph JSON",
        "abstract spatial graph",
        ".json",
        draw_json_graph,
        source_json_graph,
        notes=["Friendly JSON spatial graph converted to networkx.MultiGraph(pos/pts)."],
    )

    swc_path = DATA_DIR / "main_text" / "neuron_morphology.swc"
    write_neuron_swc(swc_path)
    swc_graph, swc_issues = spatial_graph_from_swc(swc_path)

    def draw_swc_graph(plotter):
        meshes, pts = add_spatial_graph(plotter, swc_graph)
        return meshes, pts, swc_issues

    def source_swc_graph(plotter):
        return add_raw_spatial_graph(plotter, swc_graph)

    add_panel(
        label_with_format("Neuron Morphology", "SWC"),
        "h_neuron_morphology_swc",
        "Neuron Morphology SWC",
        "neuroscience morphology",
        ".swc",
        draw_swc_graph,
        source_swc_graph,
        notes=["SWC morphology nodes and parent links converted to 3D spatial graph edges."],
    )

    graphml_path = DATA_DIR / "main_text" / "spatial_network.graphml"
    write_graphml_spatial_network(graphml_path)
    graphml_graph, graphml_issues = spatial_graph_from_graphml(graphml_path)

    def draw_graphml_graph(plotter):
        meshes, pts = add_spatial_graph(plotter, graphml_graph)
        return meshes, pts, graphml_issues

    def source_graphml_graph(plotter):
        return add_raw_spatial_graph(plotter, graphml_graph)

    add_panel(
        label_with_format("Spatial Network", "GraphML"),
        "i_spatial_network_graphml",
        "Spatial Network GraphML",
        "graph/network exchange",
        ".graphml",
        draw_graphml_graph,
        source_graphml_graph,
        notes=["GraphML node attributes x/y/z and edge points_json converted to networkx.MultiGraph(pos/pts)."],
    )

    png_path, svg_path, pdf_path = assemble_figure(panels)
    summary.extend(
        [
            {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
            {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
            {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
        ]
    )
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "selection_path": str(SELECTION_PATH),
                "figure_goal": selection.get("figure_goal"),
                "scope_clarification": selection.get("scope_clarification"),
                "panels": summary,
            },
            indent=2,
        )
        + "\n"
    )

    print("Main-text Task 2 input-format figure")
    print(f"Panel directory: {PANEL_DIR}")
    print(f"Final PNG: {png_path}")
    print(f"Final SVG: {svg_path}")
    print(f"Final PDF: {pdf_path}")
    print(f"Summary path: {SUMMARY_PATH}")
    for item in summary:
        if "title" in item:
            print(f"Panel: {item['title']} success={item['success']} issues={item['issues'] or 'none'}")


if __name__ == "__main__":
    main()
